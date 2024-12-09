import logging
import pickle
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from threading import Lock

import cv2
import face_recognition
import numpy as np
import psutil
from celery import shared_task
from constance import config
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from face_recognition import compare_faces, face_encodings

from .models import Individual
from .utils import encode_faces, find_duplicates, get_face_detections

logger = logging.getLogger(__name__)


def preprocess_image_for_encoding(image_path):
    """
    Preprocess image to improve face detection in face_recognition.
    Applies histogram equalization to the Y channel of YUV color space.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at path: {image_path}")
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    return equalized_image


def load_encodings():
    """Load existing face encodings from the file."""
    if default_storage.exists("encodings.pkl"):
        with default_storage.open("encodings.pkl", "rb") as file:
            encodings = pickle.load(file)
    else:
        encodings = {}
    return encodings


def save_encodings(encodings):
    """Save the updated encodings to the file."""
    encodings_data = pickle.dumps(encodings)
    file_name = "encodings.pkl"

    if default_storage.exists(file_name):
        default_storage.delete(file_name)
    file_content = ContentFile(encodings_data)
    default_storage.save(file_name, file_content)


encodings_lock = Lock()


@shared_task
def generate_face_encoding(individual_id, tolerance=config.TOLERANCE):
    start_time = time.time()
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024**2)
    logger.info("Starting face recognition for individual")

    try:
        individual = Individual.objects.get(id=individual_id)
        if not individual.photo or not default_storage.exists(individual.photo.path):
            logger.error(f"Photo for individual ID {individual_id} is missing or invalid.")
            return
        image_path = individual.photo.path
        image_path, regions = get_face_detections(image_path)

        if not regions:
            logger.error(f"No face detected in the image for individual ID {individual_id}.")
            return
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image, known_face_locations=regions)

        if not encodings:
            logger.error(f"No encodings generated for the image of individual ID {individual_id}.")
            return

        current_encoding = encodings[0]
        with encodings_lock:
            all_encodings = load_encodings()
            all_encodings[individual.id] = current_encoding
            save_encodings(all_encodings)
        matches = []
        all_ids, all_encodings = zip(*all_encodings.items())
        all_encodings_array = np.array(all_encodings)
        results = face_recognition.compare_faces(all_encodings_array, current_encoding, tolerance)
        matches = [str(all_ids[idx]) for idx, match in enumerate(results) if match and all_ids[idx] != individual.id]
        if matches:
            individual.duplicate = ", ".join(matches)
            individual.save()

    except Individual.DoesNotExist:
        logger.error(f"Individual with ID {individual_id} does not exist.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    ram_after = process.memory_info().rss / (1024**2)
    elapsed_time = time.time() - start_time
    ram_used = ram_after - ram_before

    logger.info(f"generate_face_encoding task completed in {elapsed_time:.2f} seconds, using {ram_used:.2f} MB of RAM")


@shared_task
def nightly_face_encoding_task(folder_path, threshold=config.TOLERANCE):
    start_time = time.time()
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024**2)
    logger.warning("Starting nightly face encoding task")
    face_data = {}
    images_without_faces_count = 0

    all_image_paths = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))
    for image_path in all_image_paths:
        image_path_str = str(image_path)
        image_path, regions = get_face_detections(image_path_str)

        if regions:
            _, encodings = encode_faces(image_path_str, regions)
            if encodings:
                face_data[image_path] = encodings
        else:
            images_without_faces_count += 1
    duplicates = find_duplicates(face_data, threshold)
    save_encodings(face_data)

    end_time = time.time()
    ram_after = process.memory_info().rss / (1024**2)
    elapsed_time = end_time - start_time
    ram_used = ram_after - ram_before
    logger.info(
        f"Nightly face encoding task completed in {elapsed_time:.2f} seconds, using approximately {ram_used} MB of RAM "
        f"found {len(duplicates)} duplicates, {images_without_faces_count} images without faces"
    )
