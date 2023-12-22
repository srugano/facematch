# Django Face Recognition System

This Django project implements a face recognition system using the `face_recognition` library with two different models for face detection: HOG (Histogram of Oriented Gradients) and CNN (Convolutional Neural Network). These models are chosen based on the processing capabilities available (CPU=hog vs. GPU=cnn).

- The first `simple strategy` for implementation in this Django project, consist of comparing a new individual's face against each individual in the database one by one in a straightforward approach. This does not scale well with an increasing number of people. Each comparison necessitates substantial computer effort, and as the dataset develops, the overall time and resources required for these computations climb linearly. This causes a performance bottleneck, especially in bigger systems with dozens or millions of users. 
- The second implementation is a nightly Celery task that runs to process all individuals in the database. This task generates face encodings for each individual and compiles them into a `single data structure` (like a list or dictionary). This compiled data is then serialized (e.g., as a pickle file) and stored. When a new individual's image is uploaded for recognition, the system loads this pre-compiled encodings file. The new image is encoded, and this encoding is compared against the pre-compiled data. This approach significantly reduces the processing time as it eliminates the need to encode the entire dataset for each recognition request. But the produced file can become very big, unless we break it down into entities based on geographic proximity for example (clustering).
- We can further improve the system by `clustering` by grouping similar face encodings, the system compares a new face only with cluster representatives instead of all entries. This reduces comparisons, speeding up recognition. Clustering also organizes data better, aiding in its management and analysis.
- Also, we can `batch process` the individuals in order to enhance the system scalability and efficiency. We can process multiple entries or recognition requests collectively, not individually. This approach optimizes resource use, especially during nightly updates or handling multiple recognition tasks, ensuring smoother and faster operations.
- At last we can use `a machine learning model` for the very large datasets, consider training a model that can classify or match faces more efficiently than one-by-one comparisons. (`Azure method`)


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
  ```bash
  celery -A facematch worker -l info
  ```
3. Running Celery Beat
  When you start your Celery worker, also start Celery Beat to ensure the scheduled tasks are executed:
  ```bash
  celery -A facematch beat -l info
  ```
4. From the admin site, run the task `nightly_face_encoding_task` from celery beat. It will produce the dataset necessair for the comparaison. 
5. Add individuals from the admin websit. As you add individuals, the system will find duplicates gradually in the background and update the new individual with the IDs of the duplicates.