import face_recognition
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Individual
import logging
import numpy as np
import json

logger = logging.getLogger(__name__)


@receiver(post_save, sender=Individual)
def generate_face_encoding(sender, instance, created, **kwargs):
    if created and instance.photo:  # Check if it's a new instance and photo exists
        # Load the image using face_recognition
        image = face_recognition.load_image_file(instance.photo.path)
        # Assuming the first face found is the correct one
        encoding = face_recognition.face_encodings(image)[0]
        # Save the encoding
        instance.set_face_encoding(encoding)
        instance.save()

        # Initialize a list to hold names of matches
        matches = []

        # Compare with existing individuals
        for other_individual in Individual.objects.exclude(id=instance.id):
            if other_individual.face_encoding:
                other_encoding = np.array(json.loads(other_individual.face_encoding))
                results = face_recognition.compare_faces([other_encoding], encoding)
                if any(results):
                    matches.append(other_individual.__str__())

        # Update the duplicate field if matches are found
        if matches:
            instance.duplicate = ", ".join(matches)
            instance.save()
            logger.info(f"Probable matches for {instance}: {', '.join(matches)}")
