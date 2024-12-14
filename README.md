# Dedupe POC

This repository is a proof-of-concept (POC) for detecting duplicate images using facial recognition techniques. It leverages DeepFace for face detection and feature embedding, and provides options for both single-process and multi-process deduplication. Celery integration allows for distributed task processing, making it suitable for larger datasets.



## Features

1. **Face Recognition**:
   - Detect faces in images.
   - Generate feature embeddings using DeepFace.

2. **Duplicate Detection**:
   - Compare embeddings to identify duplicate or similar images.
   - Flexible backend and model configurations.

3. **Multi-Process Support**:
   - Local execution with single or multiple processes.
   - Distributed task processing using Celery.

4. **Performance Metrics**:
   - Tracks encoding and deduplication times.
   - Provides a comprehensive HTML report.

---

## Installation

### 1. Create and Activate a Virtual Environment
```bash
uv venv
uv sync
```

### 2. Set the `DEEPFACE_HOME` Environment Variable
```bash
export DEEPFACE_HOME=$PWD
```

This sets the working directory as the DeepFace home, ensuring all necessary models and configurations are correctly accessed.

---

## Usage

### **Run Locally (Single Process)**

Prepare a directory with images (e.g., `data/IMAGES`) and run:
```bash
dedupe data/IMAGES -p 1
```

### **Run Locally (Multiple Processes)**

For improved performance on larger datasets, specify the number of processes (e.g., 4):
```bash
dedupe data/IMAGES -p 4
```

### **Distributed Execution with Celery**

#### 1. Start a Celery Worker
In the first terminal, start a Celery worker:
```bash
watchmedo auto-restart --directory=./src/ --pattern *.py --recursive -- celery -A recognizeapp.c.app worker
```

#### 2. Run Deduplication with Queueing
In a second terminal, run the deduplication task with Celery:
```bash
dedupe data/IMAGES -p 4 --queue
```

---

## Monitoring with Flower

Flower provides a web interface to monitor Celery workers and tasks.

### 1. Start Flower
In the first terminal:
```bash
watchmedo auto-restart --directory=./src/ --pattern *.py --recursive -- celery -A recognizeapp.c.app flower
```

### 2. Access the Dashboard
Open your browser and navigate to:
```
http://localhost:5555
```

---

## Technical Details

### **Face Detection and Embedding**

The project uses **DeepFace** for face detection and embedding generation. Supported models and backends include:
- **Models**: `VGG-Face`, `Facenet`, `DeepFace`, `ArcFace`, and others.
- **Backends**: `opencv`, `mtcnn`, `retinaface`, and more.

The `DeepFace.represent()` function generates 128-dimensional feature vectors for each detected face, which are then compared to identify duplicates.

---

### **Deduplication Process**

1. **Encoding**:
   - Images are processed to extract face embeddings.
   - Images without detectable faces are flagged as `NO_FACE_DETECTED`.

2. **Comparison**:
   - Feature vectors are compared using cosine similarity or distance-based metrics.
   - Duplicate pairs are identified if the similarity exceeds a defined threshold.

3. **Report Generation**:
   - Results are saved in a JSON format and an HTML report is generated.
   - Metrics like total time, new images processed, and findings are included.

---

### **Multi-Process Execution**

- For local execution, Python's `multiprocessing` module is used to parallelize encoding and deduplication.
- Distributed task execution with Celery allows for scaling across multiple machines.

---

### **Key CLI Options**

- `-p, --processes`: Number of processes to use (default: CPU count).
- `--queue`: Use Celery for distributed task processing.
- `--reset`: Reset findings and encodings before processing.
- `--report`: Generate an HTML report after deduplication.
- `--model-name`: Specify the model name to use (e.g., `VGG-Face`, `ArcFace`, `Facenet`, ...). 
- `--detector-backend`: Specify the model name to use (e.g., `retinaface`, `mtcnn`, ...). 

---

## Troubleshooting

1. **`NO_FACE_DETECTED` for Valid Images**:
   - Ensure the correct model is specified (e.g., `VGG-Face`, `ArcFace`. Default to VGG-Face) or the correct backend.
   - Try enabling `enforce_detection=False` in `DeepFace`.

2. **Celery Worker Not Starting**:
   - Check if `watchmedo` is installed: `uv sync`.
   - Verify Celery configurations.

3. **Performance Issues**:
   - Use multiple processes for large datasets.
   - Chose light models (but loose on the accuracy)

---

## Future Enhancements

- Add support for storing encodings in databases (e.g., postgres, redis).
- Integrate GPU acceleration for faster embedding generation.
- Extend reporting capabilities with more detailed analytics.

