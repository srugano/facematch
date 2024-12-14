import json
import logging
import sys
from collections import ChainMap
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

from celery import Task, chord

from recognizeapp.c import app
from recognizeapp.utils import (
    Dataset,
    chop_microseconds,
    dedupe_images,
    encode_faces,
    generate_report,
)

WORKERS = 5
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def notify_status(counter: int, filepath: str, task: Task, size: int, **kwargs):
    """Update task status."""
    task.update_state(state="PROGRESS", meta=json.dumps({"file": filepath, "counter": counter}))
    task.send_event("task-progress", current=counter, total=size)


def convert_dict_keys_to_str(input_dict: Dict[Path, Any]) -> Dict[str, Any]:
    """Convert dictionary keys from Path to str."""
    return {str(k): v for k, v in input_dict.items()}


@app.task(bind=True)
def encode_chunk(
    self: Task,
    files: List[str],
    chunk_id: str,
    config: Dict[str, Any],
    pre_encodings: Dict[str, Any],
) -> Dict[str, Any]:
    """Encode faces in a chunk of files."""
    # Initialize logger
    logger.info(f"Processing chunk {chunk_id} with {len(files)} files.")

    try:
        ds = Dataset(config)
        size = len(files)
        callback = partial(notify_status, task=self, size=size)

        logger.debug(f"Encoding configuration for chunk {chunk_id}: {ds.get_encoding_config()}")
        result = encode_faces(files, ds.get_encoding_config(), pre_encodings, progress=callback)

        logger.info(f"Successfully processed chunk {chunk_id}. Encoded {len(result)} files.")
        return result

    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id}: {e}", exc_info=True)
        raise


@app.task(bind=True)
def dedupe_chunk(
    self: Task,
    files: List[str],
    ids: str,
    config: Dict[str, Any],
    pre_findings: Dict[str, Any],
) -> Dict[str, Any]:
    """Deduplicate faces in a chunk of files."""
    ds = Dataset(config)
    callback = partial(notify_status, task=self, size=len(files))

    encoded = convert_dict_keys_to_str(ds.get_encoding())  # Convert keys to str
    return dedupe_images(
        files,
        encoded,
        ds.get_dedupe_config(),
        pre_findings=pre_findings,
        progress=callback,
    )


@app.task(bind=True)
def get_findings(self: Task, results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate and save findings."""
    # dictionaries = [result[0] for result in results if isinstance(result, tuple) and isinstance(result[0], dict)]
    ds = Dataset(config)
    findings = dict(ChainMap(*results))
    ds.update_findings(findings)

    end_time = datetime.now()
    config["sys"]["encode_end_time"] = int(round(end_time.timestamp()))
    encode_start_time = datetime.fromtimestamp(config["sys"]["encode_start_time"])
    dedupe_start = datetime.fromtimestamp(config["sys"]["dedupe_start_time"])
    dedupe_end = end_time

    total = dedupe_end - encode_start_time
    elapsed1 = dedupe_start - encode_start_time
    elapsed2 = dedupe_end - dedupe_start

    results = {
        "Files": len(ds.get_files()),
        "Config": ds.get_dedupe_config(),
        "Findings": len(findings),
        "Start Time": str(encode_start_time),
        "End Time": str(dedupe_end),
        "Total Elapsed": str(chop_microseconds(total)),
        "Encoding Elapsed": str(chop_microseconds(elapsed1)),
        "Deduplication Elapsed": str(chop_microseconds(elapsed2)),
    }
    ds.save_run_info(results)
    save_report.delay(config)
    return results


@app.task(bind=True)
def get_encodings(self: Task, results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate and save encodings."""
    ds = Dataset(config)
    dictionaries = [result[0] for result in results if isinstance(result, tuple) and isinstance(result[0], dict)]
    encodings = dict(ChainMap(*dictionaries))
    ds.update_encodings(encodings)

    end_time = datetime.now()
    config["sys"]["end_time"] = int(round(end_time.timestamp()))
    start_time = datetime.fromtimestamp(config["sys"]["encode_start_time"])
    elapsed = end_time - start_time

    results = {
        "Encoded": len(encodings),
        "Start Time": str(start_time),
        "Encoding End Time": str(end_time),
        "Encoding Elapsed Time": str(chop_microseconds(elapsed)),
    }
    return results


@app.task(bind=True)
def save_report(self: Task, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate and save a report."""
    ds = Dataset(config)
    report_file = Path(ds._get_filename("_report", ".html"))
    generate_report(ds.path, ds.get_findings(), ds.get_perf(), report_file)
    logger.info("Report available at: ", report_file.absolute())
    return {**config, "Report available at: ": str(report_file.absolute())}


def get_chunks(elements: List[Any], processes: int = WORKERS) -> List[List[Any]]:
    """Divide elements into chunks for parallel processing."""
    if processes <= 0:  # Fallback to 1 process if no valid number of workers is provided
        processes = 1
    chunk_size = max(1, len(elements) // processes)
    return [elements[i : i + chunk_size] for i in range(0, len(elements), chunk_size)]


@app.task(bind=True)
def deduplicate_dataset(self: Task, config: Dict[str, Any]) -> Dict[str, Any]:
    """Deduplicate the dataset."""
    logger.info("Starting deduplicate_dataset task.")
    logger.info(f"Configuration received: {config}")

    # Ensure `sys` key is present in the config
    config.setdefault("sys", {})
    config["sys"]["dedupe_start_time"] = int(round(datetime.now().timestamp()))

    try:
        # Load dataset and prepare encodings
        ds = Dataset(config)
        logger.info(f"Loaded dataset from path: {config['path']}")

        encoded = convert_dict_keys_to_str(ds.get_encoding())
        logger.info(f"Loaded {len(encoded)} encodings.")

        existing_findings = ds.get_findings()
        logger.info(f"Loaded {len(existing_findings)} existing findings.")

        # Prepare chunks for deduplication
        chunks = get_chunks(list(encoded.keys()))
        size = len(chunks)
        logger.info(f"Divided encodings into {size} chunks for parallel deduplication.")

        # Prepare deduplication tasks
        tasks = [dedupe_chunk.s(chunk, f"{n}/{size}", config, existing_findings) for n, chunk in enumerate(chunks)]
        logger.info("Prepared deduplication tasks.")

        # Create a chord for deduplication tasks
        dd = chord(tasks)(get_findings.s(config=config))
        logger.info("Submitted deduplication tasks to Celery.")

        return {"status": "deduplication_started", "async_result": str(dd)}

    except Exception as e:
        logger.error("An error occurred in deduplicate_dataset.", exc_info=True)
        raise


@app.task(bind=True)
def process_dataset(self: Task, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process the dataset for encoding and deduplication."""
    logger.info("Starting process_dataset task.")
    logger.info(f"Configuration received: {config}")

    # Ensure `sys` key exists
    config.setdefault("sys", {"encode_start_time": int(datetime.now().timestamp())})

    try:
        ds = Dataset(config)
        files = ds.get_files()
        chunks = get_chunks(files)

        # Prepare encoding tasks
        tasks = [
            encode_chunk.s(chunk, f"{n}/{len(chunks)}", config, ds.get_encoding()) for n, chunk in enumerate(chunks)
        ]
        logger.info("Prepared encoding tasks.")

        # Define the workflow with a chord and subsequent tasks
        chord_workflow = chord(tasks)(process_encodings_and_continue.s(config))
        logger.info(f"Submitted workflow with chord: {chord_workflow.id}")

        return {"status": "submitted", "async_result": str(chord_workflow)}

    except Exception as e:
        logger.error("An error occurred in process_dataset.", exc_info=True)
        raise


@app.task(bind=True)
def process_encodings_and_continue(self: Task, results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Process encodings, deduplicate, and trigger report generation."""
    logger.info("Received encoding results. Preparing deduplication and report tasks.")

    try:
        # Chain deduplication and report tasks
        workflow = (deduplicate_dataset.s(config=config) | save_report.s(config=config)).apply_async()
        logger.info(f"Workflow for deduplication and reporting submitted: {workflow.id}")

    except Exception as e:
        logger.error("An error occurred while processing encodings.", exc_info=True)
        raise

    return {"status": "workflow_started"}
