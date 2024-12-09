import logging
import os
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from django.conf import settings
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

from pathlib import Path

from constance import config
from insightface.app import FaceAnalysis


def generate_html_report(duplicates, output_file, elapsed_time, ram_used, images_without_faces_count):
    """
    Generate an HTML report of duplicate image comparisons.

    :param duplicates: List of tuples [(path1, path2, distance)].
    :param output_file: Path to the HTML file to save.
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Duplicate Images Report</title>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            th {{
                background-color: #f4f4f4;
            }}
            img {{
                max-width: 100px;
                max-height: 100px;
            }}
        </style>
    </head>
    <body>
        <h1>Duplicate Images Report</h1>
        <table>
            <thead>
                <tr>
                    <th>First Image Name</th>
                    <th>First Image</th>
                    <th>Second Image</th>
                    <th>Second Image Name</th>
                    <th>Distance</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        <h3>Task Metrics</h3>
        <ul>
            <li>Total time taken: {elapsed_time:.2f} seconds</li>
            <li>Total RAM used: {ram_used:.2f} MB</li>
            <li>Images without faces detected: {images_without_faces_count}</li>
        </ul>
    </body>
    </html>
    """

    rows = ""

    for path1, path2, distance in duplicates:
        image1 = f'<img src="{path1}" alt="Image 1">' if os.path.exists(path1) else "Image unavailable"
        name1 = Path(path1).name if os.path.exists(path1) else "N/A"
        image2 = f'<img src="{path2}" alt="Image 2">' if os.path.exists(path2) else "Image unavailable"
        name2 = Path(path2).name if os.path.exists(path2) else "N/A"
        rows += f"""
        <tr>
            <td>{image1}</td>
            <td>{name1}</td>
            <td>{image2}</td>
            <td>{name2}</td>
            <td>{distance:.8f}</td>
        </tr>
        """
    html_content = html_template.format(rows=rows,
        elapsed_time=elapsed_time,
        ram_used=ram_used,
        images_without_faces_count=images_without_faces_count)
    with open(output_file, "w") as file:
        file.write(html_content)
    print(f"Report generated: {output_file}")


def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))


def euclidean_distance(embedding1, embedding2):
    return norm(embedding1 - embedding2)


def preprocess_image(image_path):
    """Load and convert image to RGB if necessary."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at path: {image_path}")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def get_face_detections(image_path):
    model_choice = config.FACE_MODEL.lower()

    if model_choice == "dnn":
        return get_face_detections_dnn(image_path)
    elif model_choice == "retinaface":
        return get_face_detections_retinaface(image_path)
    else:
        raise ValueError(f"Unsupported face model: {model_choice}")


def get_face_detections_dnn(image_path, prototxt=settings.PROTOTXT, caffemodel=settings.CAFFEMODEL):
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        image = preprocess_image(image_path)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        face_regions = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                face_regions.append(box.astype("int").tolist())
        return image_path, face_regions

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return image_path, []


def get_face_detections_retinaface(image_path):
    try:
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
            root=settings.ML_MODELS,
        )
        app.prepare(ctx_id=-1)
        image = preprocess_image(image_path)
        faces = app.get(image)

        face_regions = [face.bbox.astype("int").tolist() for face in faces]
        return image_path, face_regions

    except Exception as e:
        logger.error(f"Error processing image {image_path} with RetinaFace: {e}")
        return image_path, []


def encode_faces(image_path, face_regions):
    """
    Generate face encodings for detected face regions using the configured model.

    :param image_path: Path to the image file.
    :param face_regions: List of bounding boxes [(x1, y1, x2, y2)].
    :return: Tuple (image_path, encodings).
    """
    model_choice = config.FACE_MODEL.lower()
    encodings = []

    if model_choice in ["cnn", "small", "dnn"]:
        image = face_recognition.load_image_file(image_path)
        face_locations = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in face_regions]
        face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations, model=model_choice)
        encodings = [np.array(encoding) for encoding in face_encodings if len(encoding) == 128]

    elif model_choice == "retinaface":
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
            root=settings.ML_MODELS,
        )
        app.prepare(ctx_id=-1)  # Use CPU; set ctx_id=0 for GPU
        image = cv2.imread(image_path)
        faces = app.get(image)
        for face in faces:
            if face.embedding is not None:
                encodings.append(face.embedding)
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")

    return image_path, encodings


def find_duplicates(face_encodings, threshold=0.2, metric="cosine", existing: dict[str, list] = None, out=sys.stdout):
    """
    Find duplicate faces based on a similarity metric.

    :param face_encodings: Dictionary of {path: [embeddings]}.
    :param threshold: Threshold for duplicate detection.
    :param metric: Similarity metric ("cosine" or "euclidean").
    :return: List of duplicate pairs.
    """
    duplicates = []
    findings = defaultdict(list)
    findings.update(existing or {})
    encoding_list = list(face_encodings.items())
    out.write(f"{len(findings)} known duplicates already found\n")
    for i in range(len(encoding_list)):
        path1, encodings1 = encoding_list[i]
        for j in range(i + 1, len(encoding_list)):
            path2, encodings2 = encoding_list[j]
            if path2 in findings.get(path1, []):
                continue
            for encoding1 in encodings1:
                for encoding2 in encodings2:
                    if metric == "cosine":
                        similarity = cosine_similarity(encoding1, encoding2)
                        if similarity >= 1 - threshold:
                            duplicates.append((path1, path2, similarity))
                            findings[path1].append((path2, similarity.tolist()))
                            findings[path2].append((path1, similarity.tolist()))
                            break
                    elif metric == "euclidean":
                        distances = face_recognition.face_distance(encodings2, encoding1)
                        valid_indices = np.where(distances <= threshold)[0]
                        for idx in valid_indices:
                            duplicates.append((path1, path2, distances[idx]))
                            findings[path1].append((path2, distances))
                            findings[path2].append((path1, distances))
                    else:
                        raise ValueError(f"Unsupported metric: {metric}")
    return findings


def process_folder_parallel(folder_path, prototxt, caffemodel):
    start_time = time.time()
    image_paths = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))

    face_data = {}
    images_without_faces_count = 0

    with Pool(cpu_count()) as pool:
        face_regions_results = pool.starmap()
        for image_path, regions in face_regions_results:
            if regions:
                _, encodings = encode_faces(image_path, regions)
                face_data[image_path] = encodings
            else:
                images_without_faces_count += 1

    duplicates = find_duplicates(face_data, config.TOLERANCE)

    end_time = time.time()
    return len(duplicates), images_without_faces_count, end_time - start_time
