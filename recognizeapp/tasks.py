import json
import logging

import face_recognition
import numpy as np
from celery import shared_task

from .models import Individual

logger = logging.getLogger(__name__)


@shared_task
def generate_face_encoding(individual_id):
    try:
        individual = Individual.objects.get(id=individual_id)
        if individual.photo:
            image = face_recognition.load_image_file(individual.photo.path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                encoding = encodings[0]
                individual.set_face_encoding(encoding)
                individual.save()

                matches = []

                # Compare with existing individuals
                for other_individual in Individual.objects.exclude(id=individual.id):
                    if other_individual.face_encoding:
                        other_encoding = np.array(json.loads(other_individual.face_encoding))
                        results = face_recognition.compare_faces([other_encoding], encoding)
                        if any(results):
                            matches.append(str(other_individual.id))

                # Update the duplicate field if matches are found
                if matches:
                    individual.duplicate = ", ".join(matches)
                    individual.save()
                    logger.info(f"Probable matches for {individual}: {', '.join(matches)}")
            else:
                # If no faces are found, do nothing
                pass

    except Individual.DoesNotExist:
        # Handle the case where the Individual does not exist
        logger.error(f"Individual with id {individual_id} does not exist.")
