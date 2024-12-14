import json
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

WORKERS = 16


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
    ds = Dataset(config)
    size = len(files)
    callback = partial(notify_status, task=self, size=size)

    return encode_faces(files, ds.get_encoding_config(), pre_encodings, progress=callback)


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
    # Extract only the first element (dictionary) from each tuple in results
    dictionaries = [result[0] for result in results if isinstance(result, tuple) and isinstance(result[0], dict)]
    ds = Dataset(config)
    findings = dict(ChainMap(*dictionaries))
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
    content = generate_report(ds.path, ds.get_findings(), ds.get_perf())
    report_file = Path(ds._get_filename("_report", ".html"))
    report_file.write_text(content, encoding="utf-8")
    print("Report available at: ", report_file.absolute())
    return {**config, "Report available at: ": str(report_file.absolute())}


def get_chunks(elements: List[Any]) -> List[List[Any]]:
    """Divide elements into chunks for parallel processing."""
    processes = min(len(elements), WORKERS)
    chunk_size = max(1, len(elements) // processes)
    return [elements[i : i + chunk_size] for i in range(0, len(elements), chunk_size)]


@app.task(bind=True)
def deduplicate_dataset(self: Task, config: Dict[str, Any]) -> Dict[str, Any]:
    """Deduplicate the dataset."""
    ds = Dataset(config)
    encoded = convert_dict_keys_to_str(ds.get_encoding())
    existing_findings = ds.get_findings()

    now = datetime.now()
    # config["sys"]["dedupe_start_time"] = int(round(now.timestamp()))
    chunks = get_chunks(list(encoded.keys()))
    size = len(chunks)

    tasks = [dedupe_chunk.s(chunk, f"{n}/{size}", config, existing_findings) for n, chunk in enumerate(chunks)]
    dd = chord(tasks)(get_findings.s(config=config))
    return {"ds": str(ds), "async_result": str(dd)}


@app.task(bind=True)
def process_dataset(self: Task, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process the dataset for encoding and deduplication."""
    now = datetime.now()
    config["sys"] = {"encode_start_time": int(round(now.timestamp()))}

    ds = Dataset(config)
    existing_encoded = convert_dict_keys_to_str(ds.get_encoding())
    files = ds.get_files()

    chunks = get_chunks(files)
    size = len(chunks)
    tasks = [encode_chunk.s(chunk, f"{n}/{size}", config, existing_encoded) for n, chunk in enumerate(chunks)]
    dd = chord(tasks)(get_encodings.s(config=config))
    return {
        "dataset": str(ds),
        "async_result": str(dd),
        "start_time": str(now),
        "chunks": len(chunks),
    }
