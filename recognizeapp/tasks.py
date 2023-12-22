import time
import psutil
import numpy as np
from celery import shared_task
from .models import Individual
import face_recognition
from django.core.files.storage import default_storage
import pickle
from django.core.files.base import ContentFile
import os
import logging

logger = logging.getLogger(__name__)


def load_encodings():
    with default_storage.open("encodings.pkl", "rb") as file:
        encodings = pickle.load(file)
    return encodings


@shared_task
def generate_face_encoding(individual_id):
    start_time = time.time()
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 ** 2)
    logger.warning("Running face recognition for this individual")
    try:
        individual = Individual.objects.get(id=individual_id)
        if individual.photo:
            image = face_recognition.load_image_file(individual.photo.path)
            current_encoding = face_recognition.face_encodings(image)[0]
            all_encodings = load_encodings()
            matches = []

            for other_id, other_encoding in all_encodings.items():
                if other_id != individual.id:
                    results = face_recognition.compare_faces([np.array(other_encoding)], current_encoding)
                    if any(results):
                        matches.append(str(other_id))

            if matches:
                individual.duplicate = ", ".join(matches)
                individual.save()
    except Individual.DoesNotExist:
        pass
    except IndexError:
        pass

    ram_after = process.memory_info().rss / (1024 ** 2)
    elapsed_time = time.time() - start_time
    ram_used = ram_after - ram_before

    logger.warning(f"generate_face_encoding task completed in {elapsed_time} seconds, using approximately {ram_used} MB of RAM")


@shared_task
def nightly_face_encoding_task(model="hog"):
    start_time = time.time()
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 ** 2)

    logger.warning("Starting nightly face encoding task")
    encodings = {}
    for individual in Individual.objects.all():
        if individual.photo:
            image = face_recognition.load_image_file(individual.photo.path)
            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            if face_encodings:
                encodings[individual.id] = face_encodings[0]

    encodings_data = pickle.dumps(encodings)
    file_name = "encodings.pkl"
    file_content = ContentFile(encodings_data)
    default_storage.save(file_name, file_content)

    file_path = default_storage.path(file_name)
    file_size = os.path.getsize(file_path) / (1024 ** 2)

    ram_after = process.memory_info().rss / (1024 ** 2)
    elapsed_time = time.time() - start_time
    ram_used = ram_after - ram_before

    logger.warning(f"Recognition task completed in {elapsed_time} seconds, using approximately {ram_used} MB of RAM, generated file size: {file_size} MB")