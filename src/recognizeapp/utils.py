import datetime
import gzip
import json
import logging
import multiprocessing
import pickle
import sys
import zipfile
from collections import defaultdict
from functools import cached_property
from hashlib import md5
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from jinja2 import Template

logger = logging.getLogger(__name__)
NO_FACE_DETECTED = "NO_FACE_DETECTED"
MULTIPLE_FACE_DETECTED = "MULTIPLE_FACE_DETECTED"
FILE_ERROR = "GENERIC_ERROR"
ERRORS = [NO_FACE_DETECTED, MULTIPLE_FACE_DETECTED, FILE_ERROR]

NO_ENCODING = 999

EncodingType = dict[str, Union[str, list[float]]]  # {file: encoding}
FindingRecord = tuple[str, str, float]  # [file1, file2, similarity]
FindingType = list[Optional[FindingRecord]]
SilencedType = list[str, str]  # [file1, file2]


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj.name)

        return super().default(obj)


class Dataset:
    """
    A class to manage dataset-related operations, such as encoding and deduplication.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = {**config}
        self.path = Path(self.config.pop("path")).absolute()
        self.options = config["options"]
        self.storage = Path  # to avoid mistakes
        self._encoding = None
        self._findings = None

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
        self.storage(self.runinfo_db_name).unlink(True)

    def get_encoding(self) -> EncodingType:
        if self._encoding is None:
            try:
                self._encoding = json.loads(
                    self.storage(self.encoding_db_name).read_text()
                )
                # with zipfile.ZipFile(Path(self.encoding_db_name).absolute(), mode="r") as zip_file:
                #     json.loads(zip_file.read("data.json"))
            except (json.JSONDecodeError, FileNotFoundError):
                self._encoding = {}
        return self._encoding

    def get_findings(self) -> FindingType:
        if self._findings is None:
            try:
                self._findings = json.loads(
                    self.storage(self.findings_db_name).read_text()
                )
            except (json.JSONDecodeError, FileNotFoundError):
                self._findings = {}
        return self._findings

    def get_perf(self):
        if self.storage(self.runinfo_db_name).exists():
            return json.loads(self.storage(self.runinfo_db_name).read_text())
        else:
            return {}

    def get_silenced(self) -> SilencedType:
        if self.storage(self.silenced_db_name).exists():
            return json.loads(self.storage(self.silenced_db_name).read_text())
        else:
            return {}

    def get_files(self) -> list[str]:
        patterns = ("*.png", "*.jpg", "*.jpeg")
        files = [
            str(f.absolute())
            for f in self.storage(self.path).iterdir()
            if any(f.match(p) for p in patterns)
        ]
        return files

    def update_findings(self, findings):
        self.storage(self.findings_db_name).write_text(json.dumps(findings))
        self._findings = None
        return self.findings_db_name

    def update_encodings(self, encodings):
        # with zipfile.ZipFile(Path(self.encoding_db_name).absolute(), mode="w", compression=zipfile.ZIP_DEFLATED,
        #                      compresslevel=9) as zip_file:
        #     zip_file.writestr("data.json", data=json.dumps(encodings))
        #
        self.storage(self.encoding_db_name).write_text(json.dumps(encodings, cls=PathEncoder))
        self._encodings = None
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

    def get_dedupe_threshold(self):
        return self.config["dedupe_threshold"]


def show_progress(a, b):
    sys.stdout.write(".")
    sys.stdout.flush()


def chop_microseconds(delta):
    return delta - datetime.timedelta(microseconds=delta.microseconds)


def encode_faces(
        files: list[str], options=None, pre_encodings=None, progress=None
) -> tuple[EncodingType, int, int]:
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
                results[file] = MULTIPLE_FACE_DETECTED
            else:
                results[file] = result[0]["embedding"]
                added += 1
        except TypeError as e:
            logger.exception(e)
            results[file] = FILE_ERROR
        except ValueError:
            results[file] = NO_FACE_DETECTED
    return results, added, existing


def get_chunks(
        elements: list[Any], max_len=multiprocessing.cpu_count()
) -> list[list[Any]]:
    processes = min(len(elements), max_len)
    chunk_size = len(elements) // processes
    chunks = [elements[i: i + chunk_size] for i in range(0, len(elements), chunk_size)]
    return chunks


def dedupe_images(
        files: list[str],
        encodings: dict[str, Union[str, list[float]]],
        dedupe_threshold: float,
        options: dict[str, Any] = None,
        pre_findings=None,
        progress=None,
) -> FindingType:
    from deepface import DeepFace

    if not callable(progress):
        progress = lambda *a: True

    findings = defaultdict(list)
    if pre_findings:
        findings.update(pre_findings)
    config = options or {}
    config["silent"] = True
    for n, file1 in enumerate(files):
        progress(n, file1)
        enc1 = encodings[file1]
        if enc1 in ERRORS:
            findings[file1].append([enc1, NO_ENCODING])
            continue
        for file2, enc2 in encodings.items():
            if file1 == file2:
                continue
            if file2 in [x[0] for x in findings.get(file1, [])]:
                continue
            if file2 in findings:
                continue
            if enc2 in ERRORS:
                continue
            res = DeepFace.verify(enc1, enc2, **config)
            similarity = float(1 - res["distance"])
            if similarity >= dedupe_threshold:
                findings[file1].append([file2, similarity])
    results: FindingType = []
    for img, duplicates in findings.items():
        for dup in duplicates:
            results.append((img, dup[0], dup[1]))
    return results


def _generate_report(
        findings,
        metrics,
        symmetric=False,
        threshold=0.4,
        edges=0,
        template_name="report.html",
) -> tuple[str, dict]:
    template = (Path(__file__).parent / template_name).read_text()

    results = []
    added = []

    def _get_pair_key(f1, f2):
        return "=="

    if edges > 0:
        threshold = 0.0
        errors = []
        top = 0
        bottom = 0
        for file, other, similarity in findings:
            pair_key = _get_pair_key(file, other)
            if pair_key in added:
                continue
            if similarity == NO_ENCODING:
                errors.append([Path(file).name, other, similarity, pair_key])
            else:
                added.append(pair_key)
                results.append(
                    [Path(file).name, Path(other).name, similarity, pair_key]
                )
                top += 1
            if top >= edges:
                break
        results.append(["-", "-", "-", "-"])
        for file, other, similarity in findings[::-1]:
            pair_key = _get_pair_key(file, other)
            if pair_key in added:
                continue
            added.append(pair_key)
            results.append([Path(file).name, Path(other).name, similarity, pair_key])
            bottom += 1
            if bottom >= edges:
                break

        results.append(["-", "-", "-", "-"])
        results.extend(errors)
    else:
        for file, other, similarity in findings:
            # pair_key = str(md5(("_".join(sorted([file, other]))).encode()).hexdigest())
            pair_key = _get_pair_key(file, other)
            # if not symmetric and (pair_key in added):
            #     continue
            # added.append(pair_key)
            if similarity >= threshold:
                results.append([Path(file).name, Path(other).name, similarity, pair_key])
    options = {
        "edges": edges,
        "threshold": threshold,
        "symmetric": symmetric,
        "findings": len(findings),
        "results": len(results),
    }
    metrics["report"] = options
    return (
        Template(template).render(metrics={"Report": options}, findings=results, edges=edges),
        options
    )


def generate_html_report(
        findings: FindingType, metrics: dict[str, Any], symmetric=False, threshold=0.4, edges=0
) -> tuple[str, dict]:
    return _generate_report(
        findings,
        metrics,
        symmetric=symmetric,
        threshold=threshold,
        edges=edges,
        template_name="report.html",
    )


def generate_csv_report(
        working_dir, findings, metrics, symmetric=False, threshold=0.4, edges=0
) -> tuple[str, dict]:
    return _generate_report(
        findings,
        metrics,
        symmetric=symmetric,
        threshold=threshold,
        edges=edges,
        template_name="csv.html",
    )


def distance_to_similarity(distance):
    return 1 - distance
