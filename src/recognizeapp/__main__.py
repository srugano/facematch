import multiprocessing
import tracemalloc
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict

import click

from recognizeapp.utils import (Dataset, dedupe_images, encode_faces,
                                generate_report)

NO_FACE_DETECTED = "NO FACE DETECTED"

# Compatibility dictionary
MODEL_BACKEND_COMPATIBILITY = {
    "VGG-Face": ["opencv", "mtcnn", "retinaface", "ssd"],
    "Facenet": ["mtcnn", "retinaface", "mediapipe"],
    "Facenet512": ["mtcnn", "retinaface", "mediapipe"],
    "OpenFace": ["opencv", "mtcnn"],
    "DeepFace": ["mtcnn", "retinaface", "ssd", "dlib"],
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


def process_files(
    files,
    threshold=0.5,
    num_processes=1,
    depface_options=None,
    pre_encodings=None,
    pre_findings=None,
):
    """
    Process a list of image files to encode faces and find duplicates.

    :param files: List of file paths to process.
    :param threshold: Similarity threshold for duplicate detection.
    :param num_processes: Number of parallel processes.
    :param depface_options: Options for face detection/recognition.
    :param pre_encodings: Pre-existing encodings (if available).
    :param pre_findings: Pre-existing findings (if available).
    :return: Encodings, findings, and performance metrics.
    """
    start_time = datetime.now()
    tracemalloc.start()

    # Divide files into chunks for multiprocessing
    if num_processes > 1:
        chunk_size = max(1, len(files) // num_processes)
        args = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
    else:
        args = [sorted(files)]

    # Encode faces in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            partial_enc = pool.map(
                partial(
                    encode_faces, options=depface_options, pre_encodings=pre_encodings
                ),
                args,
            )
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
    encodings = {}
    for d in partial_enc:
        encodings.update(d)
    encoding_time = datetime.now()

    # Deduplicate faces in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            partial_find = pool.map(
                partial(
                    dedupe_images,
                    options=depface_options,
                    encodings=encodings,
                    pre_findings=pre_findings,
                ),
                args,
            )
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
    findings = {}
    for d in partial_find:
        findings.update(d)

    # Capture performance metrics
    end_time = datetime.now()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("traceback")

    metrics = {
        "Encoding Time": str(encoding_time - start_time),
        "Deduplication Time": str(end_time - encoding_time),
        "Total Time": str(end_time - start_time),
        "RAM Mb": top_stats[0].size / 1024 / 1024,
        "Total Files": len(files),
        "Duplicates": len(findings),
        "Threshold": threshold,
        "Processes": num_processes,
        **depface_options,
    }

    click.echo(f"Encoding Time:      {metrics['Encoding Time']}")
    click.echo(f"Deduplication Time: {metrics['Deduplication Time']}")
    click.echo(f"Total Time:         {metrics['Total Time']}")
    return encodings, findings, metrics


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=0.5,
    help="Similarity threshold for duplicate detection.",
)
@click.option(
    "--reset", is_flag=True, help="Reset the dataset (clear encodings and findings)."
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
            "yolov8",
            "centerface",
            "skip",
        ]
    ),
    default="retinaface",
    help="Face detection backend to use.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose output for debugging.")
def cli(path, processes, threshold, reset, queue, report, verbose, **depface_options):
    """
    CLI to process a folder of images to detect and deduplicate faces.

    :param path: Path to the folder containing images.
    :param processes: Number of processes to use.
    :param threshold: Similarity threshold for duplicates.
    :param reset: Reset the dataset (clear encodings and findings).
    :param queue: Queue the task for background processing.
    :param report: Generate a report after processing.
    """
    # Filter image files in the provided folder
    depface_options = validate_and_adjust_options(depface_options)

    patterns = ("*.png", "*.jpg", "*.jpeg")
    files = [
        str(f.absolute())
        for f in Path(path).iterdir()
        if any(f.match(p) for p in patterns)
    ]
    # Handle empty directories
    if not files:
        click.echo("No image files found in the provided directory. Exiting.", err=True)
        return

    # Ensure number of processes doesn't exceed the number of files
    processes = min(len(files), processes)

    if verbose:
        click.echo(f"Model: {depface_options['model_name']}")
        click.echo(f"Backend: {depface_options['detector_backend']}")
        click.echo(f"Threshold: {threshold}")
        click.echo(f"Process: {processes}")

    ds = Dataset({"path": path, "options": depface_options})
    report_file = (
        Path(path)
        / f"_report_{depface_options['model_name']}_{depface_options['detector_backend']}.html"
    )
    click.echo(f"Processing {len(files)} files in {path}")

    # Reset dataset if requested
    if reset:
        ds.reset()
    pre_encodings = ds.get_encoding()
    pre_findings = ds.get_findings()

    # Process dataset in queue mode
    if queue:
        from recognizeapp.tasks import process_dataset

        config = {"options": {**depface_options, "threshold": threshold}, "path": path}
        process_dataset.delay(config)
    else:
        encodings, findings, metrics = process_files(
            files,
            threshold=threshold,
            num_processes=processes,
            depface_options=depface_options,
            pre_encodings=pre_encodings,
            pre_findings=pre_findings,
        )
        metrics["Processes"] = processes
        ds.update_findings(findings)
        ds.update_encodings(encodings)
        ds.save_run_info(metrics)

    # Generate report if the `--report` flag is set
    if report:
        generate_report(Path(path).absolute(), findings, metrics, report_file)


if __name__ == "__main__":
    cli()
