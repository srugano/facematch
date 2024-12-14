import multiprocessing
import tracemalloc
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict

import click

from recognizeapp.utils import (
    Dataset,
    dedupe_images,
    encode_faces,
    generate_report,
    get_chunks,
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
            f"Error: Incompatible model-backend pair: Model '{model}' does not support Backend '{backend}'.",
            err=True,
        )
        raise click.Abort()
    return options


from tqdm import tqdm


def process_files(config, num_processes, pre_findings):
    ds = Dataset(config)
    start_time = datetime.now()
    tracemalloc.start()
    total_files = ds.get_files()
    chunks = get_chunks(total_files, num_processes)

    encodings = {}
    added = exiting = 0

    # Encode faces in parallel with tqdm progress tracking
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            with tqdm(total=len(chunks), desc="Encoding chunks", unit="chunk") as pbar:
                for result in pool.imap_unordered(
                    partial(
                        encode_faces,
                        options=ds.get_encoding_config(),
                        pre_encodings=ds.get_encoding(),
                    ),
                    chunks,
                ):
                    pbar.update(1)
                    d, a, e = result
                    added += a
                    exiting += e
                    encodings.update(d)
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()

    encoding_time = datetime.now()
    findings = {}

    # Deduplicate faces in parallel with tqdm progress tracking
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            with tqdm(total=len(chunks), desc="Deduplicating chunks", unit="chunk") as pbar:
                for result in pool.imap_unordered(
                    partial(
                        dedupe_images,
                        options=ds.get_dedupe_config(),
                        encodings=encodings,
                        pre_findings=pre_findings,
                    ),
                    chunks,
                ):
                    pbar.update(1)
                    findings.update(result)
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()

    # Capture performance metrics
    end_time = datetime.now()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("traceback")

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
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--reset", is_flag=True, help="Reset the dataset (clear encodings and findings).")
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
    help="Name of the face recognition model to use.",
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
            "centerface",
            "skip",
        ]
    ),
    default="retinaface",
    help="Face detection backend to use.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose output for debugging.")
def cli(path, processes, reset, queue, report, verbose, **depface_options):
    """
    CLI to process a folder of images to detect and deduplicate faces.

    :param path: Path to the folder containing images.
    :param processes: Number of processes to use.
    :param reset: Reset the dataset (clear encodings and findings).
    :param queue: Queue the task for background processing.
    :param report: Generate a report after processing.
    """
    # Filter image files in the provided folder
    depface_options = validate_and_adjust_options(depface_options)

    patterns = ("*.png", "*.jpg", "*.jpeg")
    files = [str(f.absolute()) for f in Path(path).iterdir() if any(f.match(p) for p in patterns)]
    # Handle empty directories
    if not files:
        click.echo("No image files found in the provided directory. Exiting.", err=True)
        return

    # Ensure number of processes doesn't exceed the number of files
    processes = min(len(files), processes)

    if verbose:
        click.echo(f"Model: {depface_options['model_name']}")
        click.echo(f"Backend: {depface_options['detector_backend']}")
        click.echo(f"Process: {processes}")

    ds = Dataset({"path": path, "options": depface_options})
    report_file = Path(path) / f"_report_{depface_options['model_name']}_{depface_options['detector_backend']}.html"
    click.echo(f"Processing {len(files)} files in {path}")

    # Reset dataset if requested
    if reset:
        ds.reset()
    else:
        ds.storage(ds.findings_db_name).unlink(True)

    config = {"options": {**depface_options}, "path": path}

    if queue:
        from recognizeapp.tasks import process_dataset

        config = {"options": {**depface_options}, "path": path}
        process_dataset.delay(config)
    else:
        click.echo(f"Spawn {processes} processes")
        pre_encodings = ds.get_encoding()
        click.echo(f"Found {len(pre_encodings)} existing encodings")
        pre_findings = ds.get_findings()
        encodings, findings, metrics = process_files(config, processes, pre_findings)
        for k, v in metrics.items():
            click.echo(f"{k:<25}: {v}")

        ds.update_findings(findings)
        ds.update_encodings(encodings)
        ds.save_run_info(metrics)

        if report:
            generate_report(ds.path, ds.get_findings(), ds.get_perf(), report_file)


if __name__ == "__main__":
    cli()
