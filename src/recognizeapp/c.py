from celery import Celery

app = Celery(
    "recognizeapp",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

app.config_from_object("recognizeapp.c", namespace="CELERY_")
app.conf.update(
    task_serializer="json",  # Use JSON for task arguments
    accept_content=["json"],  # Accept only JSON
    # result_serializer="json",            # Use JSON for results
    timezone="UTC",  # Ensure consistent timezone
    enable_utc=True,  # Use UTC timestamps
    task_track_started=True,  # Track task start
    task_time_limit=3600,  # Set a timeout for tasks (in seconds)
    task_acks_late=True,  # Ensure task acknowledgment only after completion
)

CELERY_DEBUG = True

from . import tasks
