import time
import psutil
import numpy as np
from celery import shared_task
from .models import Individual
import face_recognition
from django.core.files.storage import default_storage
import pickle
from django.core.files.base import ContentFile
from pathlib import Path
import logging
from multiprocessing import Pool, cpu_count
from .utils import get_face_detections_dnn, find_duplicates, encode_faces

logger = logging.getLogger(__name__)


def load_encodings():
    with default_storage.open("encodings.pkl", "rb") as file:
        encodings = pickle.load(file)
    return encodings


@shared_task
def generate_face_encoding(individual_id, tolerance=0.6):
    start_time = time.time()
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024**2)
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
                    results = face_recognition.compare_faces([np.array(other_encoding)], current_encoding, tolerance)
                    if any(results):
                        matches.append(str(other_id))

            if matches:
                individual.duplicate = ", ".join(matches)
                individual.save()
    except Individual.DoesNotExist:
        pass
    except IndexError:
        pass

    ram_after = process.memory_info().rss / (1024**2)
    elapsed_time = time.time() - start_time
    ram_used = ram_after - ram_before

    logger.warning(
        f"generate_face_encoding task completed in {elapsed_time} seconds, using approximately {ram_used} MB of RAM"
    )


prototxt = "static/deploy.prototxt"
caffemodel = "static/res10_300x300_ssd_iter_140000.caffemodel"


@shared_task
def nightly_face_encoding_task(folder_path, prototxt=prototxt, caffemodel=caffemodel, threshold=0.3):
    start_time = time.time()
    logger.warning("Starting nightly face encoding task")

    face_data = {}
    images_without_faces_count = 0

    all_image_paths = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))
    for image_path in all_image_paths:
        image_path_str = str(image_path)
        image_path, regions = get_face_detections_dnn(image_path_str, prototxt, caffemodel)

        if regions:
            _, encodings = encode_faces(image_path_str, regions)
            face_data[image_path] = encodings
        else:
            images_without_faces_count += 1

    # Find duplicates
    duplicates = find_duplicates(face_data, threshold)

    end_time = time.time()

    # Save face encodings to a pickle file
    encodings_data = pickle.dumps(face_data)
    file_name = "encodings.pkl"
    file_content = ContentFile(encodings_data)
    default_storage.save(file_name, file_content)

    logger.warning(
        f"Nightly face encoding task completed in {end_time - start_time:.2f} seconds, "
        f"found {len(duplicates)} duplicates, {images_without_faces_count} images without faces"
    )
