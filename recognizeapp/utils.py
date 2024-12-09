import logging
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from django.conf import settings
from numpy.linalg import norm

logger = logging.getLogger(__name__)

from constance import config
from insightface.app import FaceAnalysis
from pathlib import Path


def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))


def euclidean_distance(embedding1, embedding2):
    return norm(embedding1 - embedding2)


def preprocess_image(image_path):
    """Load and convert image to RGB if necessary."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at path: {image_path}")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def get_face_detections(image_path):
    model_choice = config.FACE_MODEL.lower()

    if model_choice == "dnn":
        return get_face_detections_dnn(image_path)
    elif model_choice == "retinaface":
        return get_face_detections_retinaface(image_path)
    else:
        raise ValueError(f"Unsupported face model: {model_choice}")


def preprocess_image(image_path):
    """Load and convert image to RGB if necessary."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at path: {image_path}")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def get_face_detections_dnn(image_path, prototxt=settings.PROTOTXT, caffemodel=settings.CAFFEMODEL):
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


def encode_faces(image_path, face_regions):
    """
    Generate face encodings for detected face regions using the configured model.

    :param image_path: Path to the image file.
    :param face_regions: List of bounding boxes [(x1, y1, x2, y2)].
    :return: Tuple (image_path, encodings).
    """
    model_choice = config.FACE_MODEL.lower()
    encodings = []

    if model_choice in ["cnn", "small", "dnn"]:
        image = face_recognition.load_image_file(image_path)
        face_locations = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in face_regions]
        face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations, model=model_choice)
        encodings = [np.array(encoding) for encoding in face_encodings if len(encoding) == 128]

    elif model_choice == "retinaface":
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
            root=settings.ML_MODELS,
        )
        app.prepare(ctx_id=-1)  # Use CPU; set ctx_id=0 for GPU
        image = cv2.imread(image_path)
        faces = app.get(image)
        for face in faces:
            if face.embedding is not None:
                encodings.append(face.embedding)
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")

    return image_path, encodings


def find_duplicates(face_encodings, threshold=0.2, metric="cosine"):
    """
    Find duplicate faces based on a similarity metric.

    :param face_encodings: Dictionary of {path: [embeddings]}.
    :param threshold: Threshold for duplicate detection.
    :param metric: Similarity metric ("cosine" or "euclidean").
    :return: List of duplicate pairs.
    """
    duplicates = []
    encoding_list = list(face_encodings.items())

    for i in range(len(encoding_list)):
        path1, encodings1 = encoding_list[i]
        for j in range(i + 1, len(encoding_list)):
            path2, encodings2 = encoding_list[j]
            for encoding1 in encodings1:
                for encoding2 in encodings2:
                    if metric == "cosine":
                        similarity = cosine_similarity(encoding1, encoding2)
                        if similarity >= 1 - threshold:  # Adjust threshold for cosine
                            logger.info(
                                f"Duplicate found between {Path(path1).name} and {Path(path2).name} with similarity: {similarity}"
                            )
                            duplicates.append((path1, path2))
                            break
                    elif metric == "euclidean":
                        distance = euclidean_distance(encoding1, encoding2)
                        print(f"Cosine distance: {distance}")
                        if distance <= threshold:
                            logger.info(
                                f"Duplicate found between {Path(path1).name} and {Path(path2).name} with distance: {distance}"
                            )
                            duplicates.append((path1, path2))
                            break
                    else:
                        raise ValueError(f"Unsupported metric: {metric}")
    return duplicates


def process_folder_parallel(folder_path, prototxt, caffemodel):
    start_time = time.time()
    image_paths = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))

    face_data = {}
    images_without_faces_count = 0

    with Pool(cpu_count()) as pool:
        face_regions_results = pool.starmap()
        for image_path, regions in face_regions_results:
            if regions:
                _, encodings = encode_faces(image_path, regions)
                face_data[image_path] = encodings
            else:
                images_without_faces_count += 1

    duplicates = find_duplicates(face_data, config.TOLERANCE)

    end_time = time.time()
    return len(duplicates), images_without_faces_count, end_time - start_time
