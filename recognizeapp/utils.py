import json
import logging
from collections import defaultdict
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

NO_FACE_DETECTED = "NO FACE DETECTED"


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def preprocess_image(image_path):
    """Load and preprocess an image for face detection."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at path: {image_path}")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def get_face_detections(image_path, model="retinaface"):
    """Detect faces in an image using the specified model."""
    if model == "dnn":
        return get_face_detections_dnn(image_path)
    elif model == "retinaface":
        return get_face_detections_retinaface(image_path)
    else:
        raise ValueError(f"Unsupported model: {model}")


def get_face_detections_dnn(image_path, prototxt=settings.PROTOTXT, caffemodel=settings.CAFFEMODEL):
    """Use OpenCV DNN model for face detection."""
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        image = preprocess_image(image_path)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        face_regions = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                face_regions.append(box.astype("int").tolist())
        return image_path, face_regions

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return image_path, []


def get_face_detections_retinaface(image_path):
    """Use RetinaFace model for face detection."""
    try:
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
            root=settings.ML_MODELS,
        )
        app.prepare(ctx_id=-1)
        image = preprocess_image(image_path)
        faces = app.get(image)

        face_regions = [face.bbox.astype("int").tolist() for face in faces]
        return image_path, face_regions

    except Exception as e:
        logger.error(f"Error processing image {image_path} with RetinaFace: {e}")
        return image_path, []


def encode_images(files, existing_encoding, model="retinaface"):
    """Process and encode face data for a batch of images."""
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
        root=settings.ML_MODELS,
    )
    app.prepare(ctx_id=-1)  # Use CPU; modify ctx_id for GPU

    face_data = {**existing_encoding}

    for image_path in files:
        if image_path in face_data:
            continue

        image = cv2.imread(image_path)
        if image is None:
            face_data[image_path] = NO_FACE_DETECTED
            continue

        faces = app.get(image)
        encodings = [face.embedding for face in faces if face.embedding is not None]

        if encodings:
            face_data[image_path] = encodings
        else:
            face_data[image_path] = NO_FACE_DETECTED

    return face_data


def dedupe_images(file_pair, encodings, threshold, metric="cosine"):
    """Find duplicates between two sets of encodings."""
    path1, path2 = file_pair
    findings = defaultdict(list)

    encodings1 = encodings.get(path1, NO_FACE_DETECTED)
    encodings2 = encodings.get(path2, NO_FACE_DETECTED)

    if encodings1 == NO_FACE_DETECTED or encodings2 == NO_FACE_DETECTED:
        return {
            path1: [(NO_FACE_DETECTED, 99)] if encodings1 == NO_FACE_DETECTED else [],
            path2: [(NO_FACE_DETECTED, 99)] if encodings2 == NO_FACE_DETECTED else [],
        }

    for encoding1 in encodings1:
        for encoding2 in encodings2:
            if metric == "cosine":
                similarity = cosine_similarity(encoding1, encoding2)
                if similarity >= 1 - threshold:
                    findings[path1].append((path2, similarity))
                    findings[path2].append((path1, similarity))
                    break
            else:
                raise ValueError(f"Unsupported metric: {metric}")
    return findings


def process_files(
    files: list[str],
    working_dir: str,
    metric="cosine",
    threshold=0.4,
    num_processes=4,
    skip_encoding=False,
    existing_findings=None,
):
    """
    Process a list of image files to encode faces and find duplicates.

    :param files: List of file paths to process.
    :param working_dir: Directory to save intermediate results.
    :param metric: Similarity metric to use ('cosine').
    :param threshold: Tolerance threshold for duplicate detection.
    :param num_processes: Number of parallel processes.
    :param skip_encoding: Skip encoding step if True.
    :param existing_findings: Dictionary of existing findings (optional).
    """
    encoding_file = Path(working_dir) / "_encoding.json"
    findings_file = Path(working_dir) / f"_findings_{metric}.json"
    results_findings = Path(working_dir) / f"_results_{metric}.json"
    start_time = datetime.now()

    if encoding_file.exists():
        with encoding_file.open("r", encoding="utf-8") as f:
            face_data = json.load(f)
    else:
        face_data = {}

    if isinstance(existing_findings, Path) and existing_findings.exists():
        with existing_findings.open("r", encoding="utf-8") as f:
            existing_findings = json.load(f)
    elif not isinstance(existing_findings, dict):
        existing_findings = {}

    chunk_size = max(1, len(files) // num_processes)
    args = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]

    if not skip_encoding:
        with Pool(processes=num_processes) as pool:
            try:
                partial_sums = pool.map(partial(encode_images, existing_encoding=face_data), args)
            except KeyboardInterrupt:
                pool.terminate()
                pool.close()
        face_data = {}
        for d in partial_sums:
            face_data.update(d)
        encoding_file.write_text(json.dumps(face_data, cls=NumpyEncoder))

    elapsed_time = datetime.now() - start_time

    if findings_file.exists():
        with findings_file.open("r", encoding="utf-8") as f:
            findings = json.load(f)
    else:
        findings = {}

    file_pairs = [(files[i], files[j]) for i in range(len(files)) for j in range(i + 1, len(files))]

    with Pool(processes=num_processes) as pool:
        try:
            partial_func = partial(dedupe_images, encodings=face_data, threshold=threshold, metric=metric)
            partial_results = pool.map(partial_func, file_pairs)
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()

    for result in partial_results:
        for key, value in result.items():
            findings.setdefault(key, []).extend(value)

    findings_file.write_text(json.dumps(findings, cls=NumpyEncoder))
    results_findings.write_text(
        json.dumps(
            {
                "Total Time": str(datetime.now() - start_time),
                "Encoding Time": str(elapsed_time),
                "Total Files": len(files),
                "Duplicates": len(findings),
                "Threshold": threshold,
                "Processes": num_processes,
                "Metric": metric,
            }
        )
    )


def generate_report(working_dir, model="dnn"):
    """
    Generate a Django-based report for duplicates.
    """
    findings_file = Path(working_dir) / f"_findings_{model}.json"
    if not findings_file.exists():
        raise ValueError(f"Findings file not found: {findings_file}")

    findings = json.loads(findings_file.read_text())
    results = []
    for img, duplicates in findings.items():
        for dup in duplicates:
            results.append([img, dup[0], dup[1]])

    results = sorted(results, key=lambda x: x[2], reverse=True)
    report_file = Path(working_dir) / f"_report_{model}.html"
    template_path = settings.STATIC_ROOT / "report.html"

    if not template_path.exists():
        raise ValueError(f"Template not found at {template_path}")

    from django.template import Context, Template

    template = Template(template_path.read_text())
    report_file.write_text(template.render(Context({"findings": results})))

    print(f"Report generated: {report_file}")
