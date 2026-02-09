import json
from collections import ChainMap
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union
from celery import signals

from celery import Task, chord
from celery.canvas import Signature
from celery.utils.imports import qualname

from recognizeapp.c import app, DedupeTask
from recognizeapp.utils import (
    Dataset,
    chop_microseconds,
    dedupe_images,
    encode_faces,
    generate_html_report,
    FindingType,
    EncodingType,
    get_chunks,
    generate_csv_report,
)

CHUNK_WORKERS = 30
CHUNK_SIZE = 100


#
#
def notify_status(counter: int, filepath: str, task: Task, size: int, **kwargs):
    """Update task status."""
    custom_state = f"step_{counter}"
    task.update_state(state="PROGRESS", meta={"counter": counter, "filepath": filepath})
    signals.task_prerun.send(
        sender=task.name, task_id=task.request.id, custom_state=custom_state
    )

    # task.update_state(state="PROGRESS", meta=json.dumps({"file": filepath, "counter": counter}))
    # task.send_event("task-custom_state", current=counter, total=filepath)
    # with task.app.events.default_dispatcher() as dispatcher:
    #     dispatcher.send('task-custom_state', field1='value1', field2='value2')
    # print(counter, filepath)


def shadow_name(task, args, kwargs, options):
    try:
        s: Signature = options["chord"]
        group: str = options["group_id"].split("-")[-1]
        chunk = int(options["group_index"])
        name = f"{qualname(s.type)}({group})-{chunk:03}"
        return name
    except Exception as e:
        return str(e)


#
@app.task(bind=True, base=DedupeTask, shadow_name=shadow_name)
def encode_chunk(
    self: DedupeTask,
    files: List[str],
    config: Dict[str, Any],
) -> tuple[EncodingType, int, int]:
    """Encode faces in a chunk of files."""
    ds = Dataset(config)
    size = len(files)
    callback = partial(notify_status, task=self, size=size)
    pre_encodings = ds.get_encoding()
    return encode_faces(
        files, ds.get_encoding_config(), pre_encodings, progress=callback
    )


@app.task(bind=True, base=DedupeTask)
def dedupe_chunk(
    self: Task,
    files: List[str],
    config: Dict[str, Any],
) -> FindingType:
    """Deduplicate faces in a chunk of files."""
    ds = Dataset(config)
    dedupe_threshold = ds.get_dedupe_threshold()
    callback = None
    size = len(files)
    callback = partial(notify_status, task=self, size=size)

    encoded = ds.get_encoding()  # Convert keys to str
    return dedupe_images(
        files,
        encoded,
        dedupe_threshold,
        ds.get_dedupe_config(),
        progress=callback,
    )


@app.task(bind=True, base=DedupeTask)
def callback_findings(
    self: Task, results: list[FindingType], config: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate and save findings."""
    # Extract only the first element (dictionary) from each tuple in results
    ds = Dataset(config)
    findings: FindingType = []
    for d in results:
        findings.extend(d)
    # sort by similarity
    findings = sorted(findings, key=lambda x: -x[2])
    ds.update_findings(findings)

    # end_time = datetime.now()
    # config["sys"]["encode_end_time"] = int(round(end_time.timestamp()))
    # encode_start_time = datetime.fromtimestamp(config["sys"]["encode_start_time"])
    # dedupe_start = datetime.fromtimestamp(config["sys"]["dedupe_start_time"])
    # dedupe_end = end_time

    # total = dedupe_end - encode_start_time
    # elapsed1 = dedupe_start - encode_start_time
    # elapsed2 = dedupe_end - dedupe_start
    #
    results = {
        "Files": len(ds.get_files()),
        "Config": ds.get_dedupe_config(),
        "Findings": len(findings),
        # "Start Time": str(encode_start_time),
        # "End Time": str(dedupe_end),
        # "Total Elapsed": str(chop_microseconds(total)),
        # "Encoding Elapsed": str(chop_microseconds(elapsed1)),
        # "Deduplication Elapsed": str(chop_microseconds(elapsed2)),
    }
    ds.save_run_info(results)
    save_report.delay(config)
    return results


@app.task(bind=True, base=DedupeTask)
def callback_encodings(
    self: Task, results: tuple[EncodingType, int, int], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Aggregate and save encodings."""
    ds = Dataset(config)
    encodings = dict(ChainMap(*[result[0] for result in results]))
    ds.update_encodings(encodings)
    #
    # end_time = datetime.now()
    # config["sys"]["end_time"] = int(round(end_time.timestamp()))
    # start_time = datetime.fromtimestamp(config["sys"]["encode_start_time"])
    # elapsed = end_time - start_time
    deduplicate_dataset.delay(config)

    results = {
        "Encoded": len(encodings),
        # "Start Time": str(start_time),
        # "Encoding End Time": str(end_time),
        # "Encoding Elapsed Time": str(chop_microseconds(elapsed)),
    }
    return results


@app.task(bind=True, base=DedupeTask)
def save_report(self: Task, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate and save a report."""
    ds = Dataset(config)
    threshold = config["report_threshold"]
    edges = config["edges"]
    content1, __ = generate_html_report(
        ds.get_findings(),
        {},
        threshold=threshold,
        symmetric=config["symmetric"],
        edges=edges,
    )
    content2, __ = generate_csv_report(
        ds.get_findings(), threshold=config["report_threshold"], edges=config["edges"]
    )
    if edges:
        report_file = Path(ds._get_filename(f"report-{edges}", ".html"))
        csv_file = Path(ds._get_filename(f"report-{edges}", ".csv"))
    else:
        report_file = Path(ds._get_filename(f"report-{threshold}", ".html"))
        csv_file = Path(ds._get_filename(f"report-{threshold}", ".csv"))

    report_file.write_text(content1, encoding="utf-8")
    csv_file.write_text(content2, encoding="utf-8")
    return {**config, "Report available at: ": str(report_file.absolute())}


#
# def get_chunks(elements: List[Any]) -> List[List[Any]]:
#     """Divide elements into chunks for parallel processing."""
#     processes = min(len(elements), CHUNK_PAGINATION)
#     chunk_size = max(1, len(elements) // processes)
#     return [elements[i : i + chunk_size] for i in range(0, len(elements), chunk_size)]


@app.task(bind=True, base=DedupeTask)
def deduplicate_dataset(self: Task, config: Dict[str, Any]) -> Dict[str, Any]:
    """Deduplicate the dataset."""
    ds = Dataset(config)
    encoded = ds.get_encoding()
    # existing_findings = ds.get_findings()
    #
    # now = datetime.now()
    # # config["sys"]["dedupe_start_time"] = int(round(now.timestamp()))
    chunks = get_chunks(list(encoded.keys()), CHUNK_WORKERS)
    size = len(chunks)
    #
    tasks = [dedupe_chunk.s(chunk, config) for n, chunk in enumerate(chunks)]
    chord_id = chord(tasks)(callback_findings.s(config=config))
    return {"ds": str(ds), "chord_id": str(chord_id), "chunks": size}


@app.task(bind=True, base=DedupeTask)
def process_dataset(self: Task, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process the dataset for encoding and deduplication."""
    now = datetime.now()
    config["sys"] = {"encode_start_time": int(round(now.timestamp()))}

    ds = Dataset(config)
    if config["reset"]:
        ds.reset()
    files = ds.get_files()

    # chunks = get_chunks(files, CHUNK_WORKERS)
    chunks = get_chunks(files, (len(files) // CHUNK_SIZE) + 1)
    size = len(chunks)
    tasks = [encode_chunk.s(chunk, config) for n, chunk in enumerate(chunks)]
    chord_id = chord(tasks)(callback_encodings.s(config=config))

    return {
        "dataset": str(ds),
        "chord_id": str(chord_id),
        "start_time": str(now),
        "chunks": len(chunks),
    }
