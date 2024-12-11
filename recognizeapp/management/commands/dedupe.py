import json
from pathlib import Path

from django.core.management.base import BaseCommand

from recognizeapp.utils import generate_report, process_files


class Command(BaseCommand):
    help = "Process a folder of images to detect and match faces."

    def add_arguments(self, parser):
        parser.add_argument("-f", "--folder", type=str, required=True, help="Path to the folder containing images.")

        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default="retinaface",
            choices=["retinaface", "dnn"],
            help="Face detection model to use ('retinaface' or 'dnn').",
        )
        parser.add_argument(
            "-n", "--num-processes", type=int, default=4, help="Number of processes to use for encoding."
        )
        parser.add_argument("-s", "--skip-encoding", action="store_true", help="Skip encoding images.")
        parser.add_argument(
            "-r",
            "--report",
            action="store_true",
            help="Generate a report using existing findings.",
        )
        parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            default=0.5,
            help="Tolerance threshold for detecting duplicates. Default is 0.5.",
        )

    def handle(self, *args, **options):
        folder_path = options["folder"]
        threshold = options["threshold"]
        model = options["model"].lower()
        skip_encoding = options["skip_encoding"]
        num_processes = options["num_processes"]
        generate_only_report = options["report"]

        self.stdout.write(f"Processing folder: {folder_path}")
        self.stdout.write(f"Using model: {model}")
        self.stdout.write(f"Tolerance threshold: {threshold}")
        self.stdout.write(f"Number of processes: {num_processes}")
        self.stdout.write(f"Skip encoding: {'Yes' if skip_encoding else 'No'}")
        self.stdout.write(f"Generate report only: {'Yes' if generate_only_report else 'No'}")

        try:
            if generate_only_report:
                generate_report(folder_path, model=model)
                self.stdout.write(f"Report generated successfully in {folder_path}.")
            else:
                process_path(
                    folder_path=folder_path,
                    threshold=threshold,
                    model=model,
                    skip_encoding=skip_encoding,
                    num_processes=num_processes,
                )
                self.stdout.write(f"Processing completed successfully in {folder_path}.")
        except Exception as e:
            self.stderr.write(f"Error while processing: {e}")


def process_path(folder_path, threshold=0.4, model="retinaface", num_processes=4, skip_encoding=False):
    """
    Process a folder of images to find duplicates based on face encodings.

    :param folder_path: Path to the folder containing images.
    :param threshold: Tolerance threshold for duplicates.
    :param model: Face detection model to use ('retinaface' or 'dnn').
    :param num_processes: Number of parallel processes to use.
    :param skip_encoding: Whether to skip encoding the images.
    """
    patterns = ("*.png", "*.jpg", "*.jpeg")
    files = [str(f.absolute()) for f in Path(folder_path).iterdir() if any(f.match(p) for p in patterns)]
    findings_file = Path(folder_path) / f"_findings_{model}.json"
    if not findings_file.exists():
        findings_file.write_text(json.dumps({}))

    process_files(
        files=files,
        metric="cosine",
        working_dir=folder_path,
        threshold=threshold,
        skip_encoding=skip_encoding,
        num_processes=num_processes,
        existing_findings=findings_file,
    )
