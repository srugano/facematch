import logging
import sys

# Configure global logging
logging.basicConfig(
    level=logging.INFO,  # Log level (INFO, DEBUG, etc.)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to stdout
        logging.FileHandler("celery_tasks.log", mode="a"),  # Optional log file
    ],
)

# Set Celery-specific logging level if necessary
celery_logger = logging.getLogger("celery")
celery_logger.setLevel(logging.INFO)

# Celery application setup
from celery import Celery

app = Celery(
    "recognizeapp",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

app.config_from_object("recognizeapp.c", namespace="CELERY_")
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
)

CELERY_DEBUG = True

from . import tasks
