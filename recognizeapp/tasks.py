import json
import logging
import time
from multiprocessing import cpu_count
from pathlib import Path
from threading import Lock

import face_recognition
import numpy as np
import psutil
from celery import shared_task
from constance import config
from django.core.files.storage import default_storage

from .models import Individual
from .utils import NumpyEncoder, encode_images, generate_report, process_files

logger = logging.getLogger(__name__)

NO_FACE_DETECTED = "NO FACE DETECTED"

encodings_lock = Lock()


def load_encodings(folder_path):
    """
    Load existing face encodings from a JSON file in the specified folder.

    :param folder_path: Path to the folder containing the encoding JSON file.
    :return: Dictionary of encodings.
    """
    encoding_file = Path(folder_path) / "_encodings.json"
    if encoding_file.exists():
        with encoding_file.open("r", encoding="utf-8") as file:
            encodings = json.load(file)
    else:
        encodings = {}
    return encodings


def save_encodings(folder_path, encodings):
    """
    Save updated face encodings to a JSON file in the specified folder.

    :param folder_path: Path to the folder to save the encoding JSON file.
    :param encodings: Dictionary of encodings to save.
    """
    encoding_file = Path(folder_path) / "_encodings.json"
    with encoding_file.open("w", encoding="utf-8") as file:
        json.dump(encodings, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


@shared_task
def generate_face_encoding(individual_id, tolerance=config.TOLERANCE):
    """
    Generate face encoding for a single individual.
    """
    start_time = time.time()
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024**2)

    try:
        individual = Individual.objects.get(id=individual_id)
        if not individual.photo or not default_storage.exists(individual.photo.path):
            logger.warning(f"Image not found for individual ID {individual_id}")
            return

        image_path = individual.photo.path

        face_data = encode_images([image_path], existing_encoding={})
        encodings = face_data.get(image_path, None)

        if encodings == NO_FACE_DETECTED or not encodings:
            logger.warning(f"No faces detected for individual ID {individual_id}")
            return

        current_encoding = encodings[0]
        with encodings_lock:
            all_encodings = load_encodings()
            all_encodings[individual.id] = current_encoding
            save_encodings(all_encodings)

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
    """
    Process all images in a folder to encode faces and find duplicates.
    """
    start_time = time.time()
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024**2)

    logger.info("Starting nightly face encoding task.")

    # Prepare paths and initialize variables
    all_image_paths = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))
    total_images = len(all_image_paths)

    if total_images == 0:
        logger.warning(f"No images found in folder: {folder_path}")
        return

    # Process files to encode faces and find duplicates
    findings_file = Path(folder_path) / f"_findings_retinaface.json"
    try:
        process_files(
            files=[str(img) for img in all_image_paths],
            working_dir=folder_path,
            metric="cosine",
            threshold=threshold,
            num_processes=cpu_count(),
            skip_encoding=False,
            existing_findings=findings_file,
        )

        # Generate report after processing
        generate_report(folder_path, model="retinaface")

    except Exception as e:
        logger.error(f"Error during nightly face encoding task: {e}")
        return

    ram_after = process.memory_info().rss / (1024**2)
    elapsed_time = time.time() - start_time
    ram_used = ram_after - ram_before

    logger.info(
        f"Nightly face encoding task completed in {elapsed_time:.2f} seconds, using approximately {ram_used:.2f} MB of RAM. "
        f"Report generated successfully in folder: {folder_path}."
    )
