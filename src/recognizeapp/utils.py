import datetime
import json
import multiprocessing
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, Union

from jinja2 import Template

NO_FACE_DETECTED = "NO_FACE_DETECTED"


class Dataset:
    def __init__(self, config):
        self.config = {**config}
        self.path = Path(self.config.pop("path")).absolute()
        self.options = config["options"]
        self.storage = Path  # to avoid mistakes

    def __str__(self):
        return f"<Dataset: {Path(self.path).name}"

    def _get_filename(self, prefix, suffix=".json") -> str:
        parts = [self.options["model_name"], self.options["detector_backend"]]
        extra = "_".join(parts)
        return str(self.storage(self.path) / f"_{prefix}_{extra}{suffix}")

    @cached_property
    def encoding_db_name(self) -> str:
        return self._get_filename("encoding")

    @cached_property
    def findings_db_name(self) -> str:
        return self._get_filename("findings")

    @cached_property
    def silenced_db_name(self) -> str:
        return self._get_filename("silenced")

    @cached_property
    def runinfo_db_name(self) -> str:
        return self._get_filename("perf")

    def reset(self):
        self.storage(self.encoding_db_name).unlink(True)
        self.storage(self.findings_db_name).unlink(True)
        self.storage(self.silenced_db_name).unlink(True)

    def get_encoding(self):
        if self.storage(self.encoding_db_name).exists():
            return json.loads(self.storage(self.encoding_db_name).read_text())
        else:
            return {}

    def get_findings(self):
        if self.storage(self.findings_db_name).exists():
            return json.loads(self.storage(self.findings_db_name).read_text())
        else:
            return {}

    def get_perf(self):
        if self.storage(self.runinfo_db_name).exists():
            return json.loads(self.storage(self.runinfo_db_name).read_text())
        else:
            return {}

    def get_silenced(self):
        if self.storage(self.silenced_db_name).exists():
            return json.loads(self.storage(self.silenced_db_name).read_text())
        else:
            return {}

    def get_files(self):
        patterns = ("*.png", "*.jpg", "*.jpeg")
        files = [
            str(f.absolute())
            for f in self.storage(self.path).iterdir()
            if any(f.match(p) for p in patterns)
        ]
        return files

    def update_findings(self, findings):
        self.storage(self.findings_db_name).write_text(json.dumps(findings))
        return self.findings_db_name

    def update_encodings(self, encodings):
        self.storage(self.encoding_db_name).write_text(json.dumps(encodings))
        return self.encoding_db_name

    def save_run_info(self, info):
        self.storage(self.runinfo_db_name).write_text(json.dumps(info))

    def get_encoding_config(self) -> dict[str, Union[str, int, float, bool]]:
        return {
            "model_name": self.options["model_name"],
            "detector_backend": self.options["detector_backend"],
        }

    def get_dedupe_config(self) -> dict[str, Union[str, int, float, bool]]:
        return {
            "threshold": self.options["threshold"],
            "model_name": self.options["model_name"],
            "detector_backend": self.options["detector_backend"],
        }


def chop_microseconds(delta):
    return delta - datetime.timedelta(microseconds=delta.microseconds)


def encode_faces(
    files: list[str], options=None, pre_encodings=None, progress=None
) -> tuple[dict[str, Union[str, list[float]]], int, int]:
    from deepface import DeepFace

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


def get_chunks(
    elements: list[Any], max_len=multiprocessing.cpu_count()
) -> list[list[Any]]:
    processes = min(len(elements), max_len)
    chunk_size = len(elements) // processes
    chunks = [elements[i : i + chunk_size] for i in range(0, len(elements), chunk_size)]
    return chunks


def dedupe_images(
    files: list[str],
    encodings: dict[str, Union[str, list[float]]],
    options: dict[str, Any] = None,
    pre_findings=None,
    progress=None,
):
    from deepface import DeepFace

    if not callable(progress):
        progress = lambda *a: True

    findings = defaultdict(list)
    if pre_findings:
        findings.update(pre_findings)
    for n, file1 in enumerate(files):
        progress(n, file1)
        enc1 = encodings[file1]
        if enc1 == NO_FACE_DETECTED:
            findings[file1].append([NO_FACE_DETECTED, 99])
            continue
        for file2, enc2 in encodings.items():
            if file1 == file2:
                continue
            if file2 in [x[0] for x in findings.get(file1, [])]:
                continue
            if file2 in findings:
                continue
            if enc2 == NO_FACE_DETECTED:
                continue
            print(111.1, options)
            res = DeepFace.verify(enc1, enc2, **(options or {}))
            findings[file1].append([file2, res["distance"]])
    return findings


def generate_report(working_dir, findings, metrics):
    global manager

    def _resolve(p):
        if p == NO_FACE_DETECTED:
            return NO_FACE_DETECTED
        return Path(p).absolute().relative_to(working_dir)

    template = (Path(__file__).parent / "report.html").read_text()

    results = []
    for img, duplicates in findings.items():
        for dup in duplicates:
            pair_key = sorted([img, dup[0]])
            results.append([_resolve(img), _resolve(dup[0]), dup[1], pair_key])

    results = sorted(results, key=lambda x: x[2])
    return Template(template).render(metrics=metrics, findings=results)


def distance_to_similarity(distance):
    return 1 - distance
