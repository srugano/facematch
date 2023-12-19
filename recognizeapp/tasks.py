import json
import logging
import pickle

import face_recognition
import numpy as np
from celery import shared_task
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from .models import Individual

logger = logging.getLogger(__name__)


def load_encodings():
    with default_storage.open("encodings.pkl", "rb") as file:
        encodings = pickle.load(file)
    return encodings


@shared_task
def generate_face_encoding(individual_id):
    try:
        individual = Individual.objects.get(id=individual_id)
        if individual.photo:
            # Load the image and get encoding
            image = face_recognition.load_image_file(individual.photo.path)
            current_encoding = face_recognition.face_encodings(image)[0]

            # Load existing encodings
            all_encodings = load_encodings()

            # Initialize a list to hold IDs of matches
            matches = []

            # Compare with existing encodings
            for other_id, other_encoding in all_encodings.items():
                if other_id != individual.id:
                    results = face_recognition.compare_faces([np.array(other_encoding)], current_encoding)
                    if any(results):
                        matches.append(str(other_id))

            # Update the duplicate field if matches are found
            if matches:
                individual.duplicate = ", ".join(matches)
                individual.save()

    except Individual.DoesNotExist:
        # Handle the case where the Individual does not exist
        pass
    except IndexError:
        # Handle the case where no face encodings are found
        pass


@shared_task
def nightly_face_encoding_task(model="hog"):
    encodings = {}
    for individual in Individual.objects.all():
        if individual.photo:
            image = face_recognition.load_image_file(individual.photo.path)
            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if face_encodings:
                encodings[individual.id] = face_encodings[0]

    # Serialize and save the encodings
    encodings_data = pickle.dumps(encodings)
    file_name = "encodings.pkl"
    file_content = ContentFile(encodings_data)
    default_storage.save(file_name, file_content)
