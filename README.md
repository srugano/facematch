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
   pip install -r requirements.txt
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
4. You will have to create a superuser:
  ```python
     python manage.py createsuperuser
  ```
### Running the Project

1. Start the Django development server:
  ```python 
  python manage.py runserver
  ```

2. Log into the admin server http://llovalhost:8000/admin and Update the **Config** variable for the Tolerance to **0.4**. Also, the [DNN caffemodel](https://github.com/sr6033/face-detection-with-OpenCV-and-DNN/blob/master/res10_300x300_ssd_iter_140000.caffemodel) is the default but there are more extensively trained ArcNet/FaceNet models that have been trained on diverse datasets, generally better at detecting faces across different demographics, handling occlusions (e.g., masks, sunglasses, head coverings). So you should have [RetinaFace model](https://github.com/deepinsight/insightface/blob/master/python-package/README.md) ready to be instaled with the requirements. Update the DNN model to use the **retinaface**


3. In a separate terminal, start the Celery worker:
  ```bash
  celery -A face_rec worker -l info
  ```
4. Running Celery Beat
  When you start your Celery worker, also start Celery Beat to ensure the scheduled tasks are executed:
  ```bash
  celery -A face_rec beat -l info
  ```
5. From the admin site, run the task `nightly_face_encoding_task` from celery beat. It will produce the dataset necessair for the comparaison.
```python
$folder_path="/home/stock/dEV/face_rec/individual_photos/"
$nightly_face_encoding_task.delay(folder_path=folder_path)
``` 
You should get logs resembling this:

```bash
[2024-12-08 20:35:03,258: INFO/MainProcess] Task recognizeapp.tasks.nightly_face_encoding_task[cedf15c7-72a1-46b2-85da-bbafa8ac2c16] received
[2024-12-08 20:35:03,260: WARNING/ForkPoolWorker-1] Starting nightly face encoding task
[2024-12-08 20:46:59,458: INFO/ForkPoolWorker-1] Duplicate found between 1672584790683.jpg and 1672585996282.jpg with similarity: 0.5490464568138123
[2024-12-08 20:46:59,471: INFO/ForkPoolWorker-1] Duplicate found between 1262424872919.jpg and 1262304901305.jpg with similarity: 0.5264420509338379
[2024-12-08 20:46:59,472: INFO/ForkPoolWorker-1] Duplicate found between 1262355278215.jpg and 1262338824104.jpg with similarity: 0.5062891244888306
[2024-12-08 20:46:59,482: INFO/ForkPoolWorker-1] Duplicate found between 1672587204763.jpg and 1672587019995.jpg with similarity: 0.9347215294837952
[2024-12-08 20:46:59,488: INFO/ForkPoolWorker-1] Duplicate found between 1672591316297.jpg and 1262308937778.jpg with similarity: 0.5152726173400879
[2024-12-08 20:46:59,504: INFO/ForkPoolWorker-1] Duplicate found between 169900729996T.jpg and 1699007299965.jpg with similarity: 1.0
[2024-12-08 20:46:59,513: INFO/ForkPoolWorker-1] Duplicate found between 1262307336878.jpg and 1262430908975.jpg with similarity: 0.508508026599884
[2024-12-08 20:46:59,526: INFO/ForkPoolWorker-1] Duplicate found between 1262308475322.jpg and 1262312212683.jpg with similarity: 0.5270479917526245
[2024-12-08 20:46:59,546: INFO/ForkPoolWorker-1] Duplicate found between 1262304901305.jpg and 1262310228896.jpg with similarity: 0.5257623791694641
[2024-12-08 20:46:59,556: INFO/ForkPoolWorker-1] Duplicate found between 1672585996282.jpg and 1672585279204.jpg with similarity: 0.5041026473045349
[2024-12-08 20:46:59,562: INFO/ForkPoolWorker-1] Duplicate found between 1262307542265.jpg and 1262304151314.jpg with similarity: 0.5015009045600891
[2024-12-08 20:46:59,567: INFO/ForkPoolWorker-1] Duplicate found between 1262307729662.jpg and 1262355519680.jpg with similarity: 0.5033266544342041
[2024-12-08 20:46:59,570: INFO/ForkPoolWorker-1] Duplicate found between 1514769828219.jpg and 1514769963968.jpg with similarity: 0.8707915544509888
[2024-12-08 20:46:59,571: INFO/ForkPoolWorker-1] Duplicate found between 1262308937778.jpg and 1262305535045.jpg with similarity: 0.5417503118515015
[2024-12-08 20:46:59,571: INFO/ForkPoolWorker-1] Duplicate found between 1262308937778.jpg and 1514767788895.jpg with similarity: 0.5110607743263245
[2024-12-08 20:46:59,572: INFO/ForkPoolWorker-1] Duplicate found between 1262308937778.jpg and 1262355519680.jpg with similarity: 0.5006821751594543
[2024-12-08 20:46:59,594: INFO/ForkPoolWorker-1] Duplicate found between 1672587876893.jpg and 1672589333736.jpg with similarity: 0.5233407616615295
[2024-12-08 20:46:59,608: INFO/ForkPoolWorker-1] Duplicate found between 1514767788895.jpg and 1514767969951.jpg with similarity: 0.9579339027404785
[2024-12-08 20:46:59,612: INFO/ForkPoolWorker-1] Nightly face encoding task completed in 716.35 seconds, using approximately 237.44921875 MB of RAM found 18 duplicates, 6 images without faces
[2024-12-08 20:46:59,615: INFO/ForkPoolWorker-1] Task recognizeapp.tasks.nightly_face_encoding_task[cedf15c7-72a1-46b2-85da-bbafa8ac2c16] succeeded in 716.3553295159945s: None
```
5. Add individuals from the admin website. As you add individuals, the system will find duplicates gradually in the background and update the new individual with the IDs of the duplicates.
