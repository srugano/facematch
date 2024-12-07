import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np
from django.conf import settings
import cv2
import face_recognition
import logging

logger = logging.getLogger(__name__)


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



def encode_faces(image_path, face_regions):
    image = face_recognition.load_image_file(image_path)
    encodings = []
    face_locations = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in face_regions]
    face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations, model="cnn")
    for encoding in face_encodings:
        if len(encoding) == 128:
            encodings.append(np.array(encoding))
    return image_path, encodings


def find_duplicates(face_encodings, threshold=0.2):
    duplicates = []
    encoding_list = list(face_encodings.items())
    for i in range(len(encoding_list)):
        path1, encodings1 = encoding_list[i]
        for j in range(i + 1, len(encoding_list)):
            path2, encodings2 = encoding_list[j]
            for encoding1 in encodings1:
                distances = face_recognition.face_distance(encodings2, encoding1)
                valid_distances = distances[distances <= threshold]
                if valid_distances.size > 0:
                    logger.info(f"Duplicate found between {path1} and {path2} with distance(s): {valid_distances}")
                    duplicates.append((path1, path2))
                    break
    return duplicates


def process_folder_parallel(folder_path, prototxt, caffemodel):
    start_time = time.time()
    image_paths = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))

    face_data = {}
    images_without_faces_count = 0

    with Pool(cpu_count()) as pool:
        face_regions_results = pool.starmap(
            get_face_detections_dnn, [(str(image_path), prototxt, caffemodel) for image_path in image_paths]
        )
        for image_path, regions in face_regions_results:
            if regions:
                _, encodings = encode_faces(image_path, regions)
                face_data[image_path] = encodings
            else:
                images_without_faces_count += 1

    duplicates = find_duplicates(face_data, threshold=0.3)

    end_time = time.time()
    return len(duplicates), images_without_faces_count, end_time - start_time
