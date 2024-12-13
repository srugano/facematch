import datetime
import json
import multiprocessing
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from deepface import DeepFace
from jinja2 import Template

NO_FACE_DETECTED = "NO_FACE_DETECTED"


class Dataset:
    """
    A class to manage dataset-related operations, such as encoding and deduplication.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = {**config}
        self.path: Path = Path(self.config.pop("path")).absolute()
        self.options: Dict[str, Any] = config["options"]
        self.storage: Callable[[Path], Path] = Path  # Default to Path to avoid mistakes

    def __str__(self):
        return f"<Dataset: {self.path.name}>"

    def _get_filename(self, prefix: str, suffix: str = ".json") -> Path:
        """Generate a file name based on the prefix and dataset options."""
        parts = [self.options["model_name"], self.options["detector_backend"]]
        extra = "_".join(parts)
        return self.storage(self.path) / f"_{prefix}_{extra}{suffix}"

    @cached_property
    def encoding_db_name(self) -> Path:
        """Return the file name for the encoding database."""
        return self._get_filename("encoding")

    @cached_property
    def findings_db_name(self) -> Path:
        """Return the file name for the findings database."""
        return self._get_filename("findings")

    @cached_property
    def silenced_db_name(self) -> Path:
        """Return the file name for the silenced database."""
        return self._get_filename("silenced")

    @cached_property
    def runinfo_db_name(self) -> Path:
        """Return the file name for performance metrics."""
        return self._get_filename("perf")

    def reset(self) -> None:
        """Delete all related database files to reset the dataset."""
        self.storage(self.encoding_db_name).unlink(missing_ok=True)
        self.storage(self.findings_db_name).unlink(missing_ok=True)
        self.storage(self.silenced_db_name).unlink(missing_ok=True)

    def get_encoding(self) -> Dict[Path, Union[str, List[float]]]:
        """Load the encodings from the encoding database file."""
        if self.storage(self.encoding_db_name).exists():
            return json.loads(self.storage(self.encoding_db_name).read_text())
        return {}

    def get_findings(self) -> Dict[Path, Any]:
        """Load the findings from the findings database file."""
        if self.storage(self.findings_db_name).exists():
            return json.loads(self.storage(self.findings_db_name).read_text())
        return {}

    def get_perf(self) -> Dict[Path, Any]:
        """Load performance metrics from the performance database file."""
        if self.storage(self.runinfo_db_name).exists():
            return json.loads(self.storage(self.runinfo_db_name).read_text())
        return {}

    def get_silenced(self) -> Dict[Path, Any]:
        """Load the silenced findings from the silenced database file."""
        if self.storage(self.silenced_db_name).exists():
            return json.loads(self.storage(self.silenced_db_name).read_text())
        return {}

    def get_files(self) -> List[str]:
        """Retrieve all valid image files from the dataset directory."""
        patterns = ("*.png", "*.jpg", "*.jpeg")
        return [str(f.absolute()) for f in self.storage(self.path).iterdir() if any(f.match(p) for p in patterns)]

    def update_findings(self, findings: Dict[str, Any]) -> Path:
        """Update the findings database with new findings."""
        self.storage(self.findings_db_name).write_text(json.dumps(findings))
        return self.findings_db_name

    def update_encodings(self, encodings: Dict[str, Any]) -> Path:
        """Update the encoding database with new encodings."""
        self.storage(self.encoding_db_name).write_text(json.dumps(encodings))
        return self.encoding_db_name

    def save_run_info(self, info: Dict[str, Any]) -> None:
        """Save performance metrics to the performance database."""
        self.storage(self.runinfo_db_name).write_text(json.dumps(info))

    def get_encoding_config(self) -> Dict[str, Union[str, int, float, bool]]:
        """Retrieve encoding configuration options."""
        return {
            "model_name": self.options["model_name"],
            "detector_backend": self.options["detector_backend"],
        }

    def get_dedupe_config(self) -> Dict[str, Union[str, int, float, bool]]:
        """Retrieve deduplication configuration options."""
        return {
            "model_name": self.options["model_name"],
            "detector_backend": self.options["detector_backend"],
        }


def chop_microseconds(delta: datetime.timedelta) -> datetime.timedelta:
    """Remove microseconds from a timedelta."""
    return delta - datetime.timedelta(microseconds=delta.microseconds)


def encode_faces(
    files: list[str], options=None, pre_encodings=None, progress=None
) -> tuple[dict[str, Union[str, list[float]]], int, int]:

    if not callable(progress):
        progress = lambda *a: True

    results = {}
    if pre_encodings:
        results.update(pre_encodings)
    added = existing = 0
    for n, file in enumerate(files):
        progress(n, file)
        if file in results:
            existing += 1
            continue
        try:
            result = DeepFace.represent(file, **(options or {}))
            if len(result) > 1:
                raise ValueError("More than one face detected")
            results[file] = result[0]["embedding"]
            added += 1
        except TypeError as e:
            results[file] = str(e)
        except ValueError:
            results[file] = NO_FACE_DETECTED
    return results, added, existing


def get_chunks(elements: list[Any], max_len=multiprocessing.cpu_count()) -> list[list[Any]]:
    processes = min(len(elements), max_len)
    chunk_size = len(elements) // processes
    chunks = [elements[i : i + chunk_size] for i in range(0, len(elements), chunk_size)]
    return chunks


def dedupe_images(
    files: List[str],
    encodings: Dict[str, Union[str, List[float]]],
    options: Optional[Dict[str, Any]] = None,
    pre_findings: Optional[Dict[str, Any]] = None,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, List[Union[str, float]]]:
    """Find duplicate images based on face encodings."""
    if not callable(progress):
        progress = lambda *a: None
    findings: defaultdict = defaultdict(list)
    if pre_findings:
        findings.update(pre_findings)
    for n, file1 in enumerate(files):
        progress(n, file1)
        enc1 = encodings[file1]
        # Skip if the encoding is NO_FACE_DETECTED or not a list of floats
        if enc1 == NO_FACE_DETECTED:
            findings[file1].append([NO_FACE_DETECTED, 99])
            continue
        # Skip if comparing the same file or enc2 is not valid
        for file2, enc2 in encodings.items():
            if file1 == file2:
                continue
            if file2 in [x[0] for x in findings.get(file1, [])]:
                continue
            if file2 in findings:
                continue
            if enc2 == NO_FACE_DETECTED:
                continue
            res = DeepFace.verify(enc1, enc2, **(options or {}))
            findings[file1].append([file2, res["distance"]])
    return findings


def generate_report(
    working_dir: Path,
    findings: Dict[str, List[Union[str, float]]],
    metrics: Dict[str, Any],
    report_file: Path,
    save_to_file: bool = True,
) -> None:
    """Generate an HTML report from findings and metrics."""

    def _resolve(p: Union[Path, str]) -> Union[Path, str]:
        if p == NO_FACE_DETECTED:
            return NO_FACE_DETECTED
        return Path(p).absolute().relative_to(working_dir)

    template_path = Path(__file__).parent / "report.html"
    template = Template(template_path.read_text())

    results = []
    for img, duplicates in findings.items():
        for dup in duplicates:
            results.append([_resolve(img), _resolve(dup[0]), dup[1]])

    results = sorted(results, key=lambda x: x[2])
    rendered_content = template.render(metrics=metrics, findings=results)
    if save_to_file:
        report_file.write_text(rendered_content, encoding="utf-8")
        print(f"Report successfully saved to {report_file}")


def distance_to_similarity(distance):
    return 1 - distance
