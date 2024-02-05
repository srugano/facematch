# Import necessary libraries
import logging
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import face_recognition
import numpy as np

# Define the path for storing face encodings
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
logger = logging.getLogger(__name__)


def get_face_detections_dnn(image_path, net):
    try:
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

    for region in face_regions:
        if isinstance(region, (list, tuple)) and len(region) == 4:
            top, right, bottom, left = region
            adjusted_region = (top, right, bottom, left)
            face_encodings = face_recognition.face_encodings(image, known_face_locations=[adjusted_region], model="cnn")
            encodings.extend(face_encodings)
        else:
            logger.warning(f"Invalid face region format for image {image_path}")

    return image_path, encodings


def find_duplicates(face_encodings, threshold=0.4):
    duplicate_images = set()

    for path1, encodings1 in face_encodings.items():
        for path2, encodings2 in face_encodings.items():
            if path1 != path2 and path2 not in duplicate_images:
                for encoding1 in encodings1:
                    for encoding2 in encodings2:
                        distance = face_recognition.face_distance([encoding1], encoding2)
                        if distance < threshold:
                            duplicate_images.update([path1, path2])
                            break  # Break the innermost loop if a duplicate is found

    return len(duplicate_images)


def process_folder_parallel(folder_path, prototxt, caffemodel):
    start_time = time.time()
    image_paths = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))

    face_data = {}  # Store face encodings for each image
    images_without_faces_count = 0  # Counter for images without faces

    with Pool(cpu_count()) as pool:
        # Get face regions
        face_regions_results = pool.starmap(
            get_face_detections_dnn,
            [(str(image_path), prototxt, caffemodel) for image_path in image_paths],
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
