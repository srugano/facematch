# from django.db import transaction
# from django.db.models.signals import post_save
# from django.dispatch import receiver
#
# from .models import Individual
# from .tasks import generate_face_encoding
#
#
# @receiver(post_save, sender=Individual)
# def trigger_face_encoding(sender, instance, created, **kwargs):
#     if created and instance.photo:
#         transaction.on_commit(lambda: generate_face_encoding.delay(instance.id))
