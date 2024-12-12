from collections import ChainMap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from celery import chord

from recognizeapp.c import app
from recognizeapp.utils import (Dataset, chop_microseconds, dedupe_images,
                                encode_faces, generate_report)

WORKERS = 4


@app.task(bind=True)
def encode_chunk(self, files: list[str], config, pre_encodings):
    ds = Dataset(config)

    def callback(args):
        self.update_state(state='PROGRESS', meta={'file': str(args)})

    return encode_faces(files, ds.get_encoding_config(), pre_encodings, progress=callback)


@app.task(bind=True)
def dedupe_chunk(self, files: list[str], config, pre_findings):
    ds = Dataset(config)

    def callback(args):
        self.update_state(state='PROGRESS', meta={'file': str(args)})

    return dedupe_images(
        files, ds.get_encoding(), ds.get_dedupe_config(), pre_findings=pre_findings, progress=callback
    )


@app.task(bind=True)
def get_findings(self, results, config):
    ds = Dataset(config)
    findings = dict(ChainMap(*results))
    filename = ds.update_findings(findings)

    end_time = datetime.now()
    start_time = datetime.fromtimestamp(config["start_time"])
    elapsed: timedelta = end_time - start_time

    results = {
        "Findings": len(findings),
        "Start Time": str(start_time),
        "End Time": str(end_time),
        "Elapsed": str(chop_microseconds(elapsed)),
        "db": filename,
    }
    print(results)
    save_report.delay(config)
    return results


@app.task(bind=True)
def save_report(self, config):
    ds = Dataset(config)
    content = generate_report(ds.path, ds.get_findings(), {})
    report_file = Path(ds._get_filename("_report", ".html"))
    report_file.write_text(content)
    print("Report available at: ", Path(report_file).absolute())


@app.task(bind=True)
def get_encodings(self, results, config):
    ds = Dataset(config)
    encodings = dict(ChainMap(*results))
    filename = ds.update_encodings(encodings)
    end_time = datetime.now()
    start_time = datetime.fromtimestamp(config["start_time"])
    elapsed: timedelta = end_time - start_time
    results = {
        "Encoded": len(encodings),
        "Start Time": str(start_time),
        "End Time": str(end_time),
        "Elapsed": str(chop_microseconds(elapsed)),
        "db": filename,
    }
    print(results)
    deduplicate_dataset.delay(config)
    return results


def get_chunks(elements: list[Any]) -> list[list[Any]]:
    processes = min(len(elements), WORKERS)
    chunk_size = len(elements) // processes
    chunks = [elements[i: i + chunk_size] for i in range(0, len(elements), chunk_size)]
    return chunks


@app.task(bind=True)
def deduplicate_dataset(self, config):
    ds = Dataset(config)
    encoded = ds.get_encoding()
    existing_findings = ds.get_findings()

    chunks = get_chunks(list(encoded.keys()))

    tasks = [dedupe_chunk.s(chunk, config, existing_findings) for chunk in chunks]
    dd = chord(tasks)(get_findings.s(config=config))

    result = {"ds": str(ds)}
    print(result)
    return result


@app.task(bind=True)
def process_dataset(self, config):
    now = datetime.now()
    config["start_time"] = int(round(now.timestamp()))

    ds = Dataset(config)
    existing_encoded = ds.get_encoding()
    files = ds.get_files()

    chunks = get_chunks(files)

    tasks = [encode_chunk.s(chunk, config, existing_encoded) for chunk in chunks]

    dd = chord(tasks)(get_encodings.s(config=config))

    result = {"async_result": str(dd), "start_time": str(now)}
    print(f"process results: {result}")
    return result
