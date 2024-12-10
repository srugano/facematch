import json
import logging
import pickle
import signal
import sys
from datetime import datetime
from collections import defaultdict
from functools import wraps, partial
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

import face_recognition
import numpy as np
from django.utils.translation import gettext as _

from recognizeapp.tasks import nightly_face_encoding_task
from recognizeapp.utils import get_face_detections_retinaface, get_face_detections, encode_faces, cosine_similarity, \
    euclidean_distance

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

import subprocess

from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections


#
#
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


NO_FACE_DETECTED = "NO FACE DETECTED"


def encode_images(files: list[str], existing_encoding):
    face_data = {**existing_encoding}

    for image_path in files:
        image_path_str = str(image_path)
        if image_path_str not in face_data:
            image_path, regions = get_face_detections(image_path_str)
            if regions:
                _, encodings = encode_faces(image_path_str, regions)
                if encodings:
                    face_data[image_path_str] = encodings
                    logger.info(f'{image_path_str} processed\n')
            else:
                face_data[image_path_str] = NO_FACE_DETECTED
        else:
            logger.debug(f'{image_path_str} found\n')
    return face_data


def dedupe_images(files: list[str], encodings, threshold, existing_findings, metric="cosine"):
    findings = defaultdict(list)
    findings.update(existing_findings or {})

    for path1 in files:
        encodings1 = encodings[path1]
        if path1 in findings:
            continue
        if encodings1 == NO_FACE_DETECTED:
            findings[path1].append([NO_FACE_DETECTED, 99])
            continue
        for path2, encodings2 in encodings.items():
            if path2 in [x[0] for x in findings.get(path1, [])]:
                continue
            if path2 in findings:
                continue
            if path1 == path2:
                continue
            if encodings2 == NO_FACE_DETECTED:
                continue
            for encoding1 in encodings1:
                for encoding2 in encodings2:
                    if metric == "cosine":
                        try:
                            similarity = cosine_similarity(encoding1, encoding2)
                        except Exception as e:
                            findings[path1].append((f"{path2}: {str(e)}", 99))
                            break
                        if similarity >= 1 - threshold:
                            findings[path1].append((path2, similarity.tolist()))
                            findings[path2].append((path1, similarity.tolist()))
                            break
                    elif metric == "euclidean":
                        distances = euclidean_distance(encodings1, encoding2)
                        if distances <= threshold and distances > 1e-6:
                            findings[path1].append((path2, distances))
                            findings[path2].append((path1, distances))
                else:
                    continue  # only executed if the inner loop did NOT break
                break  # only executed if the inner loop DID break
    return findings


def process_files(files: list[str], working_dir, metric="cosine", threshold=0.4, num_processes=4, skip_encoding=False,
                  out=sys.stdout):
    encoding_file = Path(working_dir) / "_encoding.json"
    existing_findings = Path(working_dir) / f"_findings_{metric}.json"
    results_findings = Path(working_dir) / f"_results_{metric}.json"
    start_time = datetime.now()
    if encoding_file.exists():
        face_data = json.loads(encoding_file.read_text())
    else:
        face_data = {}
    chunk_size = len(files) // num_processes
    args = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

    if skip_encoding:
        encodings = face_data
        elapsed_time = "---"
        partial_time = datetime.now()
    else:
        with Pool(processes=num_processes) as pool:
            try:
                partial_sums = pool.map(partial(encode_images, existing_encoding=face_data), args)
            except KeyboardInterrupt:
                pool.terminate()
                pool.close()
        encodings = {}
        for d in partial_sums:
            encodings.update(d)
        encoding_file.write_text(json.dumps(encodings, cls=NumpyEncoder))
        partial_time = datetime.now()
        elapsed_time = partial_time - start_time
    out.write(f"Encoding Time:      {elapsed_time} \n")
    # -----
    # -----
    # -----
    if existing_findings.exists():
        findings = json.loads(existing_findings.read_text())
    else:
        findings = {}

    with Pool(processes=num_processes) as pool:
        try:
            partial_sums = pool.map(partial(dedupe_images, encodings=encodings,
                                            metric=metric,
                                            threshold=threshold, existing_findings=findings), args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
    output = {}
    for d in partial_sums:
        output.update(d)

    existing_findings.write_text(json.dumps(output, cls=NumpyEncoder))
    end_time = datetime.now()
    dedup_time = end_time - partial_time
    total_time = end_time - start_time
    out.write(f"Deduplication Time: {dedup_time} \n")
    out.write(f"Total Time:         {total_time} \n")

    results_findings.write_text(json.dumps({"Total Time": str(total_time),
                                            "Deduplication Time": str(dedup_time),
                                            "Encoding Time": str(end_time),
                                            "Total Files": len(files),
                                            "Duplicates": len(findings),
                                            "Threshold": threshold,
                                            "Processes": num_processes,
                                            "Metric": metric
                                            }, ))


def process_path(folder_path, threshold=0.4, num_processes=4, metric="cosine", skip_encoding=False):
    patterns = ("*.png", "*.jpg", "*.jpeg")
    files = [str(f.absolute()) for f in Path(folder_path).iterdir() if any(f.match(p) for p in patterns)]

    return process_files(files, metric=metric, working_dir=folder_path, threshold=threshold,
                         skip_encoding=skip_encoding,
                         num_processes=num_processes)


def generate_report(working_dir, metric="cosine"):
    def _resolve(p):
        if p == NO_FACE_DETECTED:
            return NO_FACE_DETECTED
        return Path(p).relative_to(working_dir)

    existing_findings = Path(working_dir) / f"_findings_{metric}.json"
    if existing_findings.exists():
        findings = json.loads(existing_findings.read_text())
    else:
        raise ValueError(f"{working_dir} does contain _findings_{metric}.json")
    template = (Path(__file__).parent / "report.html").read_text()
    report_file = Path(working_dir) / f"_report_{metric}.html"
    results = []
    for img, duplicates in findings.items():
        for dup in duplicates:
            pair_key = sorted([img, dup[0]])
            results.append([_resolve(img), _resolve(dup[0]), dup[1], pair_key])

    results = sorted(results, key=lambda x: x[2])
    from django.template import Template, Context
    report_file.write_text(Template(template).render(Context({"findings": results})))

    # for file, duplicates in findings.items():
    #     print(Path(file).relative_to(working_dir))


class Command(BaseCommand):
    help = (
        "Runs the command-line client for specified database, or the "
        "default database if none is provided."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "image_path",
        )
        parser.add_argument("--metric", default="cosine")
        parser.add_argument("--reset", action="store_true")
        parser.add_argument("--no-encoding", action="store_true")
        parser.add_argument("--processes", "-p", type=int, default=4)
        parser.add_argument("--threshold", "-t", type=float, default=0.5)

    def handle(self, image_path, metric, reset, processes, no_encoding, threshold, **options):
        if reset:
            (Path(image_path) / "_encoding.json").unlink(missing_ok=True)
            (Path(image_path) / f"_findings_{metric}.json").unlink(missing_ok=True)
        process_path(image_path, threshold=threshold, metric=metric, num_processes=processes, skip_encoding=no_encoding)
        generate_report(image_path, metric=metric)
