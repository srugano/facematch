import json
import logging
import os
import sys
from datetime import datetime
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import psutil

from recognizeapp.utils import get_face_detections_retinaface, get_face_detections, encode_faces, cosine_similarity, \
    euclidean_distance, get_face_detections_dnn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


from django.core.management.base import BaseCommand



#
#
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


NO_FACE_DETECTED = "NO FACE DETECTED"
NO_ENCODING = "NO ENCODING"


def get_face_detections(engine, image_path):
    if engine == "dnn":
        return get_face_detections_dnn(image_path)
    elif engine == "retinaface":
        return get_face_detections_retinaface(image_path)
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def encode_images(files: list[str], existing_encoding, engine):
    face_data = {**existing_encoding}

    for image_path in files:
        image_path_str = str(image_path)
        if image_path_str not in face_data:
            image_path, regions = get_face_detections(engine, image_path_str)
            if regions:
                _, encodings = encode_faces(image_path_str, regions)
                if encodings:
                    face_data[image_path_str] = encodings
                    logger.info(f'{image_path_str} processed\n')
                else:
                    face_data[image_path_str] = NO_ENCODING
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
        if encodings1 == NO_ENCODING:
            findings[path1].append([NO_ENCODING, 99])
            continue
        for path2, encodings2 in encodings.items():
            if path2 in [x[0] for x in findings.get(path1, [])]:
                continue
            if path2 in findings:
                continue
            if path1 == path2:
                continue
            if encodings2 in [NO_FACE_DETECTED, NO_ENCODING]:
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
                        distances = euclidean_distance(encoding1, encoding2)
                        if distances <= threshold and distances > 1e-6:
                            findings[path1].append((path2, distances))
                            findings[path2].append((path1, distances))
                else:
                    continue  # only executed if the inner loop did NOT break
                break  # only executed if the inner loop DID break
    return findings


def process_files(files: list[str], working_dir, algo="cosine", engine="dnn", threshold=0.4, num_processes=4,
                  skip_encoding=False,
                  out=sys.stdout):
    encoding_file = Path(working_dir) / f"_{engine}_encoding.json"
    existing_findings = Path(working_dir) / f"_findings_{engine}_{algo}.json"
    results_findings = Path(working_dir) / f"_results_{engine}_{algo}.json"
    start_time = datetime.now()
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 ** 2)

    if encoding_file.exists():
        face_data = json.loads(encoding_file.read_text())
    else:
        face_data = {}
    if num_processes > 1:
        chunk_size = len(files) // num_processes
        args = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    else:
        args = sorted(files)

    if skip_encoding:
        encodings = face_data
        encoding_tima = "---"
        partial_time = datetime.now()
    else:
        with Pool(processes=num_processes) as pool:
            try:
                partial_sums = pool.map(partial(encode_images, engine=engine, existing_encoding=face_data), args)
            except KeyboardInterrupt:
                pool.terminate()
                pool.close()
        encodings = {}
        for d in partial_sums:
            encodings.update(d)
        encoding_file.write_text(json.dumps(encodings, cls=NumpyEncoder))
        partial_time = datetime.now()
        encoding_tima = partial_time - start_time
    out.write(f"Encoding Time:      {encoding_tima} \n")
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
                                            metric=algo,
                                            threshold=threshold, existing_findings=findings), args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
    output = {}
    for d in partial_sums:
        output.update(d)
    end_time = datetime.now()
    ram_after = process.memory_info().rss / (1024 ** 2)
    ram_used = ram_after - ram_before

    existing_findings.write_text(json.dumps(output, cls=NumpyEncoder))
    dedup_time = end_time - partial_time
    total_time = end_time - start_time
    out.write(f"Deduplication Time: {dedup_time} \n")
    out.write(f"Total Time:         {total_time} \n")
    metrics = {"Encoding Time": "-",
               "Deduplication Time": str(dedup_time),
               "Total Time": str(total_time),
               "RAM": ram_used,
               "Total Files": len(files),
               "Duplicates": len(output),
               "Threshold": threshold,
               "Processes": num_processes,
               "Algorythm": algo,
               "Engine": engine,
               }
    if not skip_encoding:
        metrics["Encoding Time"] = str(encoding_tima)
    results_findings.write_text(json.dumps(metrics))


def process_path(folder_path, threshold=0.4, num_processes=4, algo="cosine", engine="dnn", skip_encoding=False):
    patterns = ("*.png", "*.jpg", "*.jpeg")
    files = [str(f.absolute()) for f in Path(folder_path).iterdir() if any(f.match(p) for p in patterns)]
    if len(files) < num_processes:
        num_processes = len(files)
    return process_files(files, algo=algo, working_dir=folder_path, threshold=threshold,
                         engine=engine,
                         skip_encoding=skip_encoding,
                         num_processes=num_processes)


def generate_report(working_dir, algo="cosine", name="_report", engine="dnn", limit=100):
    def _resolve(p):
        if p == NO_FACE_DETECTED:
            return NO_FACE_DETECTED
        elif p == NO_ENCODING:
            return NO_ENCODING
        return Path(p).relative_to(working_dir)

    existing_results = Path(working_dir) / f"_results_{engine}_{algo}.json"

    if existing_results.exists():
        metrics = json.loads(existing_results.read_text())
    else:
        metrics = {}

    existing_findings = Path(working_dir) / f"_findings_{engine}_{algo}.json"
    if existing_findings.exists():
        findings = json.loads(existing_findings.read_text())
    else:
        raise ValueError(f"{working_dir} does contain _findings_{engine}_{algo}.json")
    template = (Path(__file__).parent / "report.html").read_text()
    report_file = Path(working_dir) / f"{name}_{engine}_{algo}.html"
    results = []
    for img, duplicates in findings.items():
        for dup in duplicates:
            pair_key = sorted([img, dup[0]])
            if dup[1] < limit:
                results.append([_resolve(img), _resolve(dup[0]), dup[1], pair_key])

    results = sorted(results, key=lambda x: x[2])
    from django.template import Template, Context
    report_file.write_text(Template(template).render(Context({"findings": results, "metrics": metrics})))
    print("Report available at: ", report_file.absolute())


class Command(BaseCommand):
    help = (
        "Runs the command-line client for specified database, or the "
        "default database if none is provided."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "image_path",
        )
        parser.add_argument("-a", "--algo", choices=["cosine", "euclidean"], default="cosine")
        parser.add_argument("-e", "--engine", choices=["dnn", "retinaface"], default="retinaface")
        parser.add_argument("--reset", action="store_true")
        parser.add_argument("--no-encoding", "-ne", action="store_true")
        parser.add_argument("--no-dedupe", "-nd", action="store_true")
        parser.add_argument("--processes", "-p", type=int, default=4)
        parser.add_argument("--threshold", "-t", type=float, default=0.5)

        parser.add_argument("--report", "-r", default="_report")
        parser.add_argument("--limit", "-l", type=int, default=100)

    def handle(self, image_path, algo, reset, processes, no_encoding, no_dedupe, threshold, report, limit, engine,
               **options):
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
        if reset:
            (Path(image_path) / f"_encoding_{engine}.json").unlink(missing_ok=True)
            (Path(image_path) / f"_findings_{engine}_{algo}.json").unlink(missing_ok=True)

        self.stdout.write(f"Algorythm {algo}")
        self.stdout.write(f"Engine {engine}")

        if not (no_dedupe and no_encoding):
            process_path(image_path,
                         threshold=threshold,
                         algo=algo,
                         engine=engine,
                         num_processes=processes,
                         skip_encoding=no_encoding)
        generate_report(image_path, algo=algo, engine=engine, name=report, limit=limit)
