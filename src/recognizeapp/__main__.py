import multiprocessing
import tracemalloc
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Union

import click

from recognizeapp.utils import (
    Dataset,
    dedupe_images,
    encode_faces,
    get_chunks,
    show_progress, generate_html_report, generate_csv_report,
)

NO_FACE_DETECTED = "NO FACE DETECTED"

# Compatibility dictionary
MODEL_BACKEND_COMPATIBILITY = {
    "VGG-Face": ["opencv", "mtcnn", "retinaface", "ssd"],
    "Facenet": ["mtcnn", "retinaface", "mediapipe"],
    "Facenet512": ["mtcnn", "retinaface", "mediapipe"],
    "OpenFace": ["opencv", "mtcnn"],
    "DeepFace": ["mtcnn", "ssd", "dlib"],
    "DeepID": ["mtcnn", "opencv"],
    "Dlib": ["dlib", "mediapipe"],
    "ArcFace": ["mtcnn", "retinaface"],
    "SFace": ["mtcnn", "retinaface", "mediapipe"],
    "GhostFaceNet": ["mtcnn", "retinaface", "centerface", "yolov8"],
}

DEFAULT_BACKENDS = {
    "VGG-Face": "retinaface",
    "Facenet": "mtcnn",
    "Facenet512": "mtcnn",
    "OpenFace": "mtcnn",
    "DeepFace": "retinaface",
    "DeepID": "mtcnn",
    "Dlib": "dlib",
    "ArcFace": "retinaface",
    "SFace": "retinaface",
    "GhostFaceNet": "mtcnn",
}


def validate_model_backend(model: str, backend: str) -> bool:
    """Validate if the model-backend pair is compatible."""
    compatible_backends = MODEL_BACKEND_COMPATIBILITY.get(model, [])
    if backend not in compatible_backends:
        return False
    return True


def suggest_backend(model: str) -> str:
    """Suggest a default backend for a given model."""
    return DEFAULT_BACKENDS.get(model, "mtcnn")


def validate_and_adjust_options(options: Dict[str, Any]) -> Dict[str, Any]:
    model = options["model_name"]
    backend = options["detector_backend"]
    if not validate_model_backend(model, backend):
        click.echo(
            f"Error: Incompatible model-backend pair: Model '{model}' does not support Backend '{backend}'.\n"
            f"choose one of {",".join(MODEL_BACKEND_COMPATIBILITY.get(model, []))}",
            err=True,
        )
        raise click.Abort()
    return options


def process_files(config, num_processes, threshold, progress=None, reset=False):
    ds = Dataset(config)
    if reset:
        ds.reset()
    pre_findings = ds.get_findings()
    pre_encoding = ds.get_encoding()

    tracemalloc.start()
    total_files = ds.get_files()
    chunks = get_chunks(total_files, num_processes)

    start_time = datetime.now()
    # Encode faces in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            partial_enc = pool.map(
                partial(
                    encode_faces,
                    options=ds.get_encoding_config(),
                    pre_encodings=pre_encoding,
                    progress=progress,
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
    encoding_time = datetime.now()
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            partial_find = pool.map(
                partial(
                    dedupe_images,
                    dedupe_threshold=threshold,
                    options=ds.get_dedupe_config(),
                    encodings=encodings,
                    pre_findings=pre_findings,
                    progress=progress,
                ),
                chunks,
            )
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
    findings = []
    for d in partial_find:
        findings.extend(d)
    findings = sorted(findings, key=lambda x: -x[2])

    # Capture performance metrics
    end_time = datetime.now()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("traceback")

    metrics = {
        "timing": {
            "Encoding Time": str(encoding_time - start_time).split(".")[0],
            "Deduplication Time": str(end_time - encoding_time).split(".")[0],
            "Total Time": str(end_time - start_time).split(".")[0],
        },
        "perfs": {
            "RAM Mb": str(top_stats[0].size / 1024 / 1024),
            "Processes": num_processes,
        },
        "info": {
            "Total Files": len(total_files),
            "New Images": added,
            "Database": len(encodings),
            "Findings": len(findings),
            "Threshold": threshold,
        },
        "options": config["options"],
    }
    return encodings, findings, metrics


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--reset", is_flag=True, help="Reset the dataset (clear encodings.")
@click.option("--progress", is_flag=True, help="Show progress in cli.")
@click.option("--threshold", "-t", type=float, default=None, help="Model threshold")
@click.option(
    "--dedupe-threshold", type=float, default=0.4, help="Similarity hard limit"
)
@click.option(
    "--report-threshold", type=float, default=0.4, help="Similarity soft limit"
)
@click.option("--queue", is_flag=True, help="Queue the task for background processing.")
@click.option("--report", is_flag=True, help="Generate a report after processing.")
@click.option(
    "-p",
    "--processes",
    type=int,
    default=multiprocessing.cpu_count(),
    help="Number of processes to use for parallel execution.",
)
@click.option(
    "--model-name",
    type=click.Choice(MODEL_BACKEND_COMPATIBILITY.keys()),
    default="VGG-Face",
    help="Name of the face recognition model to use.",
)
@click.option(
    "--detector-backend",
    type=click.Choice(DEFAULT_BACKENDS.values()),
    default="retinaface",
    help="Face detection backend to use.",
)
@click.option("--symmetric", is_flag=True, help="")
@click.option("--edges", type=int, default=0, help="")
def cli(
        path,
        processes,
        reset,
        queue,
        report,
        symmetric,
        dedupe_threshold,
        report_threshold,
        progress,
        edges,
        **depface_options,
):
    # Filter image files in the provided folder
    depface_options = validate_and_adjust_options(depface_options)
    ds = Dataset({"path": path, "options": depface_options})
    process_start = datetime.now()
    # Reset dataset if requested
    # if reset:
    #     ds.reset()
    # else:
    #     ds.storage(ds.findings_db_name).unlink(True)
    if progress:
        progress = show_progress
    else:
        progress = None

    files = ds.get_files()
    processes = min(len(files), processes)
    config = {"options": {**depface_options}, "path": path}

    if queue:
        from recognizeapp.tasks import process_dataset

        config = {"options": {**depface_options}, "path": path, "dedupe_threshold": dedupe_threshold, "reset":reset}
        process_dataset.delay(config)
    else:
        click.echo(f"Spawn {processes} processes")
        click.echo(f"Found {len(ds.get_encoding())} existing encodings")
        click.secho(f"Encoding configuration", fg="green")
        for k, v in ds.get_encoding_config().items():
            click.echo(f"{k:<25}: {v}")

        click.secho(f"Deduplication configuration", fg="green")
        for k, v in ds.get_dedupe_config().items():
            click.echo(f"{k:<25}: {v}")

        encodings, findings, metrics = process_files(
            config, processes, progress=progress, threshold=dedupe_threshold, reset=reset
        )
        process_end = datetime.now()
        process_time = process_end - process_start
        metrics["timing"]["Overall Time"] = str(process_time).split(".")[0]

        for section, data in metrics.items():
            click.secho(section, fg="yellow")
            for k, v in data.items():
                click.echo(f"  {k:<25}: {v}")

        ds.update_findings(findings)
        ds.update_encodings(encodings)
        ds.save_run_info(metrics)

        click.secho("Files:", fg="yellow")
        click.echo(f"Encoding saved to {ds.storage(ds.encoding_db_name).absolute()}")
        click.echo(f"Findings saved to {ds.storage(ds.findings_db_name).absolute()}")
        if report:
            content, opts = generate_html_report(
                ds.get_findings(),
                ds.get_perf(),
                symmetric=symmetric,
                threshold=report_threshold,
                edges=edges,
            )
            content1, __ = generate_csv_report(
                ds.get_findings(),
                ds.get_perf(),
                symmetric=symmetric,
                threshold=report_threshold,
                edges=edges,
            )
            if edges:
                report_file = Path(ds._get_filename(f"report-{edges}", ".html"))
                csv_file = Path(ds._get_filename(f"report-{edges}", ".csv"))
            else:
                report_file = Path(ds._get_filename(f"report-{report_threshold}", ".html"))
                csv_file = Path(ds._get_filename(f"report-{report_threshold}", ".csv"))

            report_file.write_text(content)
            csv_file.write_text(content1)

            click.echo(f"Report saved to {report_file.absolute()}")
            click.secho("Report", fg="yellow")
            for k, v in opts.items():
                click.echo(f"  {k:<25}: {v}")
