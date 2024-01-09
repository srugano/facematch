# Import necessary libraries
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np
from django.conf import settings
import cv2
import face_recognition

import logging

# Define the path for storing face encodings
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
logger = logging.getLogger(__name__)


def get_face_detections_dnn(image_path, prototxt=settings.PROTOTXT, caffemodel=settings.CAFFEMODEL):
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image at path: {image_path}")

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        face_regions = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                face_regions.append(box.astype("int").tolist())
        return image_path, face_regions

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return image_path, []


def encode_faces(image_path, face_regions):
    image = face_recognition.load_image_file(image_path)
    encodings = []
    # logger.warning(f"regions {face_regions}")
    # from celery.contrib import rdb; rdb.set_trace()
    for x1, y1, x2, y2 in face_regions:
        face_image = image[y1:y2, x1:x2]
        face_encodings = face_recognition.face_encodings(face_image)
        if face_encodings:
            encodings.append(face_encodings[0])
    return image_path, encodings


def find_duplicates(face_encodings, threshold=0.4):
    duplicates = []
    for path1, encodings1 in face_encodings.items():
        for path2, encodings2 in face_encodings.items():
            if path1 != path2:
                for encoding1 in encodings1:
                    for encoding2 in encodings2:
                        distance = face_recognition.face_distance([encoding1], encoding2)
                        if distance < threshold:
                            duplicates.append((path1, path2))
    return duplicates


def process_folder_parallel(folder_path, prototxt, caffemodel):
    start_time = time.time()
    image_paths = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))

    face_data = {}  # Store face encodings for each image
    images_without_faces_count = 0  # Counter for images without faces

    with Pool(cpu_count()) as pool:
        # Get face regions
        face_regions_results = pool.starmap(
            get_face_detections_dnn, [(str(image_path), prototxt, caffemodel) for image_path in image_paths]
        )

        # Get face encodings and count images without faces
        for image_path, regions in face_regions_results:
            if regions:  # Proceed only if faces are detected
                _, encodings = encode_faces(image_path, regions)
                face_data[image_path] = encodings
            else:
                images_without_faces_count += 1  # Increment counter if no faces are detected

    # Find duplicates
    duplicates = find_duplicates(face_data, threshold=0.3)

    end_time = time.time()
    return len(duplicates), images_without_faces_count, end_time - start_time
