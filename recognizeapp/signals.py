import face_recognition
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Individual


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
