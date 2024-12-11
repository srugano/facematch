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

import cv2
import face_recognition
import numpy as np
import psutil
from insightface.app import FaceAnalysis

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


def encode_faces(engine, image_path, face_regions):
    """
    Generate face encodings for detected face regions using the configured model.

    :param image_path: Path to the image file.
    :param face_regions: List of bounding boxes [(x1, y1, x2, y2)].
    :return: Tuple (image_path, encodings).
    """
    model_choice = engine
    encodings = []

    if model_choice in ["cnn", "small", "dnn"]:
        image = face_recognition.load_image_file(image_path)
        face_locations = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in face_regions]
        face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations, model=model_choice)
        encodings = [np.array(encoding) for encoding in face_encodings if len(encoding) == 128]

    elif model_choice == "retinaface":
        from django.conf import settings
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
            root=settings.ML_MODELS,
        )
        app.prepare(ctx_id=0)  # Use CPU; set ctx_id=0 for GPU
        image = cv2.imread(image_path)
        faces = app.get(image)
        for face in faces:
            if face.embedding is not None:
                encodings.append(face.embedding)
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")

    return image_path, encodings


def encode_images(files: list[str], existing_encoding, engine):
    face_data = {**existing_encoding}
    for image_path in files:
        image_path_str = str(image_path)
        if image_path_str not in face_data:
            image_path, regions = get_face_detections(engine, image_path_str)
            if regions:
                _, encodings = encode_faces(engine, image_path_str, regions)
                if encodings:
                    face_data[image_path_str] = encodings
                else:
                    face_data[image_path_str] = NO_ENCODING
            else:
                face_data[image_path_str] = NO_FACE_DETECTED
        else:
            logger.debug(f'{image_path_str} found\n')
    return face_data


def dedupe_images(files: list[str], encodings, threshold, existing_findings):
    _findings = defaultdict(list)
    _findings.update(existing_findings or {})

    for path1 in files:
        encodings1 = encodings[path1]
        if path1 in _findings.keys():
            continue

        if encodings1 == NO_FACE_DETECTED:
            _findings[path1].append([NO_FACE_DETECTED, 99])
            continue
        if encodings1 == NO_ENCODING:
            _findings[path1].append([NO_ENCODING, 99])
            continue
        for path2, encodings2 in encodings.items():
            if path2 in [x[0] for x in _findings.get(path1, [])]:
                continue
            if path2 in _findings:
                continue
            if path1 == path2:
                continue
            if encodings2 in [NO_FACE_DETECTED, NO_ENCODING]:
                continue
            for encoding1 in encodings1:
                for encoding2 in encodings2:
                    try:
                        similarity = cosine_similarity(encoding1, encoding2)
                    except Exception as e:
                        _findings[path1].append((f"{path2}: {str(e)}", 99))
                        break
                    if similarity >= 1 - threshold:
                        _findings[path1].append((path2, similarity.tolist()))
                        _findings[path2].append((path1, similarity.tolist()))
                        break

                else:
                    continue  # only executed if the inner loop did NOT break
                break  # only executed if the inner loop DID break

    return _findings


def process_files(files: list[str], working_dir, algo="cosine", engine="dnn", threshold=0.4, num_processes=4,
                  skip_encoding=False,
                  out=sys.stdout):
    encoding_file = Path(working_dir) / f"_encoding_{engine}.json"
    existing_findings = Path(working_dir) / f"_findings_{engine}_{algo}.json"
    results_findings = Path(working_dir) / f"_perfs_{engine}_{algo}.json"
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
        args = [sorted(files)]

    if skip_encoding:
        encodings = face_data
        encoding_tima = "---"
        partial_time = datetime.now()
    else:
        with Pool(processes=num_processes) as pool:
            try:
                partial_enc = pool.map(partial(encode_images, engine=engine, existing_encoding=face_data), args)
            except KeyboardInterrupt:
                pool.terminate()
                pool.close()
        encodings = {}
        for d in partial_enc:
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
            partial_enc = pool.map(partial(dedupe_images, encodings=encodings,
                                            threshold=threshold, existing_findings=findings), args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
    output = {}
    for d in partial_enc:
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


def process_path(folder_path, threshold=0.4, num_processes=4,  engine="dnn", skip_encoding=False):
    patterns = ("*.png", "*.jpg", "*.jpeg")
    files = [str(f.absolute()) for f in Path(folder_path).iterdir() if any(f.match(p) for p in patterns)]
    if len(files) < num_processes:
        num_processes = len(files)
    return process_files(files, working_dir=folder_path, threshold=threshold,
                         engine=engine,
                         skip_encoding=skip_encoding,
                         num_processes=num_processes)


def generate_report(working_dir, name="_report", engine="dnn", limit=100):
    def _resolve(p):
        if p == NO_FACE_DETECTED:
            return NO_FACE_DETECTED
        elif p == NO_ENCODING:
            return NO_ENCODING
        return Path(p).relative_to(working_dir)

    existing_results = Path(working_dir) / f"_perfs_{engine}.json"

    if existing_results.exists():
        metrics = json.loads(existing_results.read_text())
    else:
        metrics = {}

    existing_findings = Path(working_dir) / f"_findings_{engine}.json"
    if existing_findings.exists():
        findings = json.loads(existing_findings.read_text())
    else:
        raise ValueError(f"{working_dir} does contain _findings_{engine}.json")
    template = (Path(__file__).parent / "report.html").read_text()
    report_file = Path(working_dir) / f"{name}_{engine}.html"
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
        parser.add_argument("-e", "--engine", choices=["dnn", "retinaface"], default="retinaface")
        parser.add_argument("--reset", action="store_true")
        parser.add_argument("--no-encoding", "-ne", action="store_true")
        parser.add_argument("--no-dedupe", "-nd", action="store_true")
        parser.add_argument("--processes", "-p", type=int, default=4)
        parser.add_argument("--threshold", "-t", type=float, default=0.5)

        parser.add_argument("--report", "-r", default="_report")
        parser.add_argument("--limit", "-l", type=int, default=100)

    def handle(self, image_path, reset, processes, no_encoding, no_dedupe, threshold, report, limit, engine,
               **options):
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
        if reset:
            (Path(image_path) / f"_encoding_{engine}.json").unlink(missing_ok=True)
            (Path(image_path) / f"_findings_{engine}.json").unlink(missing_ok=True)
            (Path(image_path) / f"_perfs_{engine}.json").unlink(missing_ok=True)

        self.stdout.write(f"Engine {engine}")

        if not (no_dedupe and no_encoding):
            process_path(image_path,
                         threshold=threshold,
                         engine=engine,
                         num_processes=processes,
                         skip_encoding=no_encoding)
        generate_report(image_path, engine=engine, name=report, limit=limit)
