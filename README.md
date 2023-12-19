# Django Face Recognition System

This Django project implements a face recognition system using the `face_recognition` library. It includes functionality for comparing a new face encoding against each individual in the database one by one in a straightforward approach. 

This strategy, while simple, does not scale well with an increasing number of people. Each comparison necessitates substantial computer effort, and as the dataset develops, the overall time and resources required for these computations climb linearly. This causes a performance bottleneck, especially in bigger systems with dozens or millions of users. The additional computational load can strain the system, resulting in slower reaction times and perhaps higher operational expenses as more powerful hardware is required to meet the demand.


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
2. Navigate to the cloned directory:
   ```bash
   cd facematch
   ```
3. Install packages
   ```bash
   pip install requirements.txt
   ```

### Configuration

1. **Set up a message broker** (e.g., Redis) for Celery.
2. **Configure the broker URL** in your project's `settings.py`:
    ```python
    CELERY_BROKER_URL = 'redis://localhost:6379/0'  # Modify as per your broker configuration
    ```
3. Run migrations to set up your database:
  ```python 
    python manage.py migrate
  ```

### Running the Project

1. Start the Django development server:
  ```python 
  python manage.py runserver
  ```
2. In a separate terminal, start the Celery worker:
  ```css
    celery -A facematch worker -l info
  ```