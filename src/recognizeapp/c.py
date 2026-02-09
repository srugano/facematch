from celery import Celery, Task
from celery.utils.imports import qualname

app = Celery(
    "recognizeapp",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    accept_content=["json"],  # Accept only JSON
    broker_heartbeat=0,
    enable_utc=True,  # Use UTC timestamps
    health_check_interval=4,
    result_serializer="json",            # Use JSON for results
    socket_keepalive=True,
    task_acks_late=True,
    task_acks_on_failure_or_timeout=False,
    task_send_sent_event=True,
    task_serializer="json",  # Use JSON for task arguments
    task_time_limit=3600,  # Set a timeout for tasks (in seconds)
    task_track_started=True,
    timezone="UTC",  # Ensure consistent timezone
)

app.config_from_object("recognizeapp.c", namespace="CELERY_")


class DedupeTask(Task):
    pass


from . import tasks
