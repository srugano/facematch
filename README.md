# Django Face Recognition System

This Django project implements a face recognition system using the `face_recognition` library. It includes functionality for encoding faces from a training dataset, recognizing faces in new images, and running these tasks asynchronously using Celery.

## Features

- **Face Encoding**: Encodes faces from a training dataset and stores the encodings in a database.
- **Face Recognition**: Compares new face encodings against stored encodings to recognize individuals.
- **Asynchronous Task Processing**: Utilizes Celery for handling time-consuming tasks like face encoding and recognition.
- **Scalable Architecture**: Designed to handle a growing number of face recognition requests efficiently.

## Getting Started

### Prerequisites

- Python 3.x
- Django
- Celery
- Redis (or another message broker supported by Celery)
- face_recognition library

### Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/yourusername/facematch.git
  ```
