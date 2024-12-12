import json
from collections import ChainMap
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

from celery import chord, Task

from recognizeapp.c import app
from recognizeapp.utils import (
    Dataset,
    chop_microseconds,
    dedupe_images,
    encode_faces,
    generate_report,
)

WORKERS = 16


#
def notify_status(counter, filepath, task: Task, size, **kwargs):
    print(f"-> {counter} {filepath}")
    task.update_state(
        state="PROGRESS", meta=json.dumps({"file": filepath, "counter": counter})
    )
    task.send_event("task-progress", current=counter, total=size)


@app.task(bind=True)
def encode_chunk(self: Task, files: list[str], ids, config, pre_encodings):
    ds = Dataset(config)
    size = len(files)

    callback = partial(notify_status, task=self, size=size)

    return encode_faces(
        files, ds.get_encoding_config(), pre_encodings, progress=callback
    )


@app.task(bind=True)
def dedupe_chunk(self, files: list[str], ids, config, pre_findings):
    ds = Dataset(config)

    callback = partial(notify_status, task=self, size=len(files))

    return dedupe_images(
        files,
        ds.get_encoding(),
        ds.get_dedupe_config(),
        pre_findings=pre_findings,
        progress=callback,
    )


@app.task(bind=True)
def get_findings(self, results, config):
    ds = Dataset(config)
    findings = dict(ChainMap(*results))
    ds.update_findings(findings)
    end_time = datetime.now()
    config["sys"]["encode_end_time"] = int(round(end_time.timestamp()))
    encode_start_time = datetime.fromtimestamp(config["sys"]["encode_start_time"])
    encode_end_time = datetime.fromtimestamp(config["sys"]["encode_end_time"])
    dedupe_start = datetime.fromtimestamp(config["sys"]["dedupe_start_time"])
    dedupe_end = datetime.fromtimestamp(config["sys"]["encode_end_time"])

    total: timedelta = dedupe_end - encode_start_time
    elapsed1: timedelta = encode_end_time - encode_start_time
    elapsed2: timedelta = dedupe_end - dedupe_start

    results = {
        "Files": len(ds.get_files()),
        "Config": {**ds.get_dedupe_config()},
        "Findings": len(findings),
        "Start Time": str(encode_start_time),
        "End Time": str(dedupe_end),
        "Total Elapsed": str(chop_microseconds(total)),
        "Encoding Elapsed": str(chop_microseconds(elapsed1)),
        "Deduplication Elapsed": str(chop_microseconds(elapsed2)),
    }
    ds.save_run_info(results)
    print(results)
    save_report.delay(config)
    return results


@app.task(bind=True)
def get_encodings(self, results, config):
    ds = Dataset(config)
    encodings = dict(ChainMap(*results))
    ds.update_encodings(encodings)

    end_time = datetime.now()
    config["sys"]["end_time"] = int(round(end_time.timestamp()))
    start_time = datetime.fromtimestamp(config["sys"]["encode_start_time"])
    elapsed: timedelta = end_time - start_time
    results = {
        "Encoded": len(encodings),
        "Start Time": str(start_time),
        "Encoding End Time": str(end_time),
        "Encoding Elapsed Time": str(chop_microseconds(elapsed)),
    }
    print(results)
    deduplicate_dataset.delay(config)
    return results


@app.task(bind=True)
def save_report(self, config):
    ds = Dataset(config)
    content = generate_report(ds.path, ds.get_findings(), ds.get_perf())
    report_file = Path(ds._get_filename("_report", ".html"))
    report_file.write_text(content)
    print("Report available at: ", Path(report_file).absolute())
    return {**config, "Report available at: ": str(Path(report_file).absolute())}


def get_chunks(elements: list[Any]) -> list[list[Any]]:
    processes = min(len(elements), WORKERS)
    chunk_size = len(elements) // processes
    chunks = [elements[i : i + chunk_size] for i in range(0, len(elements), chunk_size)]
    return chunks


@app.task(bind=True)
def deduplicate_dataset(self, config):
    ds = Dataset(config)
    encoded = ds.get_encoding()
    existing_findings = ds.get_findings()
    now = datetime.now()
    config["sys"]["dedupe_start_time"] = int(round(now.timestamp()))
    chunks = get_chunks(list(encoded.keys()))
    size = len(chunks)

    tasks = [
        dedupe_chunk.s(chunk, "%s/%s" % (n, size), config, existing_findings)
        for n, chunk in enumerate(chunks)
    ]
    dd = chord(tasks)(get_findings.s(config=config))

    result = {"ds": str(ds), "async_result": str(dd)}
    print(result)
    return result


@app.task(bind=True)
def process_dataset(self, config):
    now = datetime.now()
    config["sys"] = {"encode_start_time": int(round(now.timestamp()))}

    ds = Dataset(config)
    existing_encoded = ds.get_encoding()
    files = ds.get_files()

    chunks = get_chunks(files)
    size = len(chunks)
    tasks = [
        encode_chunk.s(chunk, f"{n}/{size}", config, existing_encoded)
        for n, chunk in enumerate(chunks)
    ]

    dd = chord(tasks)(get_encodings.s(config=config))

    result = {
        "dataset": str(ds),
        "async_result": str(dd),
        "start_time": str(now),
        "chunks": len(chunks),
    }
    print(f"process results: {result}")
    return result
