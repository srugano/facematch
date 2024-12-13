import multiprocessing
import sys
import tracemalloc
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Union

import click
from jinja2 import Template

from recognizeapp.utils import (Dataset, dedupe_images, encode_faces,
                                generate_report, get_chunks)

NO_FACE_DETECTED = "NO FACE DETECTED"


def process_files(config, num_processes):
    ds = Dataset(config)
    start_time = datetime.now()
    tracemalloc.start()
    total_files = ds.get_files()
    chunks = get_chunks(total_files, num_processes)

    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            partial_enc = pool.map(
                partial(
                    encode_faces, options=ds.get_encoding_config(), pre_encodings=ds.get_encoding()
                ),
                chunks,
            )
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
    encodings = {}
    added = exiting = 0
    for d, a, e in partial_enc:
        added += a
        exiting += e
        encodings.update(d)

    # encodings = encode_faces(args, depface_options, pre_encodings=pre_encodings)
    encoding_time = datetime.now()
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            partial_find = pool.map(
                partial(
                    dedupe_images,
                    options=ds.get_dedupe_config(),
                    encodings=encodings,
                    pre_findings=ds.get_findings(),
                ),
                chunks,
            )
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
    findings = {}
    for d in partial_find:
        findings.update(d)

    end_time = datetime.now()
    snapshot1 = tracemalloc.take_snapshot()
    top_stats = snapshot1.statistics("traceback")

    metrics = {
        "Encoding Time": str(encoding_time - start_time),
        "Deduplication Time": str(end_time - encoding_time),
        "Total Time": str(end_time - start_time),
        "RAM Mb": str(top_stats[0].size / 1024 / 1024),
        "Processes": num_processes,
        "------": "--------",
        "Total Files": len(total_files),
        "New Images": added,
        "Database": len(encodings),
        "Findings": len(findings),
        "======": "======",
        **config["options"],
    }
    return encodings, findings, metrics


@click.command()
@click.argument("path", type=click.Path())
@click.option("-t", "--threshold", type=float, default=0.0)
@click.option("--reset", is_flag=True)
@click.option("--queue", is_flag=True)
@click.option(
    "-p",
    "--processes",
    type=int,
    default=multiprocessing.cpu_count(),
    help="The number of processes to use."
)
@click.option(
    "--model-name",
    type=click.Choice(
        [
            "VGG-Face",
            "Facenet",
            "Facenet512",
            "OpenFace",
            "DeepFace",
            "DeepID",
            "Dlib",
            "ArcFace",
            "SFace",
            "GhostFaceNet",
        ]
    ),
    default="VGG-Face",
)
@click.option(
    "--detector-backend",
    type=click.Choice(
        [
            "opencv",
            "retinaface",
            "mtcnn",
            "ssd",
            "dlib",
            "mediapipe",
            "yolov8",
            "centerface",
            "skip",
        ]
    ),
    default="retinaface")
@click.option(
    "--distance-metric",
    type=click.Choice(
        [
            "cosine",
            "euclidean",
            "euclidean_l12"
        ]
    ),
    default="cosine"
)
def cli(path, processes, reset, queue, **depface_options):
    patterns = ("*.png", "*.jpg", "*.jpeg")
    files = [
        str(f.absolute())
        for f in Path(path).iterdir()
        if any(f.match(p) for p in patterns)
    ]

    processes = min(len(files), processes)
    ds = Dataset({"path": path, "options": depface_options})

    click.echo(f"Processing {len(files)} files")

    if reset:
        ds.reset()
    else:
        ds.storage(ds.findings_db_name).unlink(True)

    config = {"options": {**depface_options}, "path": path}

    if queue:
        from .tasks import process_dataset

        res = process_dataset.delay(config)
    else:
        click.echo(f"Spawn {processes} processes")

        # initialize to evalue perf
        # DeepFace.build_model(depface_options["model_name"], "facial_recognition")
        # DeepFace.build_model(depface_options["detector_backend"], "face_detector")
        pre_encodings = ds.get_encoding()
        click.echo(f"Found {len(pre_encodings)} existing encodings")

        pre_findings = ds.get_findings()
        encodings, findings, metrics = process_files(config, processes)

        for k, v in metrics.items():
            click.echo(f"{k:<25}: {v}")

        ds.update_findings(findings)
        ds.update_encodings(encodings)
        ds.save_run_info(metrics)

        content = generate_report(ds.path, ds.get_findings(), ds.get_perf())
        report_file = Path(ds._get_filename("_report", ".html"))
        report_file.write_text(content)
        click.echo(f"Report saved to {report_file.absolute()}")
