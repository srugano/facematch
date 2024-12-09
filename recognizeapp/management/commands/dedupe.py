import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from django.utils.translation import gettext as _

from recognizeapp.tasks import nightly_face_encoding_task
from recognizeapp.utils import get_face_detections_retinaface

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


import subprocess

from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections


class Command(BaseCommand):
    help = (
        "Runs the command-line client for specified database, or the "
        "default database if none is provided."
    )
    def add_arguments(self, parser):
        parser.add_argument(
            "image_path",
        )
        parameters = parser.add_argument_group("parameters", prefix_chars="--")
        parameters.add_argument("parameters", nargs="*")

    def handle(self, image_path, **options):
        nightly_face_encoding_task(image_path, threshold=0.4)
