from celery import Celery

app = Celery(
    "recognizeapp",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

app.config_from_object("recognizeapp.c", namespace="CELERY_")

CELERY_DEBUG = True

from . import tasks
