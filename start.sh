#!/bin/bash

# Start Celery worker in the background
celery -A face_rec worker -l info &

# Start Django's development server
python3 manage.py runserver 0.0.0.0:8000
