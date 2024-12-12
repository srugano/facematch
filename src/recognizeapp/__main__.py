import multiprocessing
import sys
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Union, Any
import click
from jinja2 import Template
import tracemalloc

from recognizeapp.utils import Dataset, encode_faces

NO_FACE_DETECTED = "NO FACE DETECTED"



def process_files(files: list[str], threshold=0.4, num_processes=4, depface_options=None, pre_encodings=None,
                  pre_findings=None):
    start_time = datetime.now()
    tracemalloc.start()

    if num_processes > 1:
        chunk_size = len(files) // num_processes
        args = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    else:
        args = [sorted(files)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            partial_enc = pool.map(partial(encode_faces, options=depface_options, pre_encodings=pre_encodings), args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
    encodings = {}
    for d in partial_enc:
        encodings.update(d)

    # encodings = encode_faces(args, depface_options, pre_encodings=pre_encodings)
    encoding_time = datetime.now()

    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            partial_find = pool.map(partial(dedupe_images, threshold=threshold,
                                            options=depface_options, encodings=encodings,
                                            pre_findings=pre_findings), args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
    findings = {}
    for d in partial_find:
        findings.update(d)

    end_time = datetime.now()
    snapshot1 = tracemalloc.take_snapshot()
    top_stats = snapshot1.statistics('traceback')

    metrics = {"Encoding Time": str(encoding_time - start_time),
               "Deduplication Time": str(end_time - encoding_time),
               "Total Time": str(end_time - start_time),
               "RAM Mb": str(top_stats[0].size / 1024 / 1024),
               "Total Files": len(files),
               "Duplicates": len(findings),
               "Threshold": threshold,
               "Processes": num_processes,
               **depface_options
               }

    sys.stdout.write(f"Encoding Time:      {encoding_time - start_time} \n")
    sys.stdout.write(f"Deduplication Time: {end_time - encoding_time} \n")
    sys.stdout.write(f"Total Time:         {end_time - start_time} \n")
    return encodings, findings, metrics


@click.command()
@click.argument("path", type=click.Path())
@click.option("-t", "--threshold", type=float, default=0.0)
@click.option("--reset", is_flag=True)
@click.option("--queue", is_flag=True)
@click.option("-p", "--processes", type=int, default=multiprocessing.cpu_count(),
              help="The number of processes to use.")
@click.option("--model-name",
              type=click.Choice(["VGG-Face", 'Facenet', 'Facenet512', 'OpenFace',
                                 'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'SFace', 'GhostFaceNet']),
              default="VGG-Face")
@click.option("--detector-backend",
              type=click.Choice(
                  ['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface', 'skip']),
              default="retinaface")
def cli(path, processes, threshold, reset, queue, **depface_options):
    patterns = ("*.png", "*.jpg", "*.jpeg")
    files = [str(f.absolute()) for f in Path(path).iterdir() if any(f.match(p) for p in patterns)]

    processes = min(len(files), processes)
    ds = Dataset({"path": path, "options": depface_options})
    report_file = Path(path) / f"_report_{depface_options["model_name"]}_{depface_options["detector_backend"]}.html"
    click.echo(f"Processing {len(files)} files")

    if reset:
        ds.reset()
    pre_encodings = ds.get_encoding()
    pre_findings = ds.get_findings()
    # if encoding_file.exists():
    #     pre_encodings = json.loads(encoding_file.read_text())
    # else:
    #     pre_encodings = None
    # if findings_file.exists():
    #     pre_findings = json.loads(findings_file.read_text())
    # else:
    #     pre_findings = None

    if queue:
        from .tasks import process_dataset
        config = {"threshold": threshold,
                  "options": depface_options,
                  "path": path
                  }
        res = process_dataset.delay(config)
    else:
        # initialize to evalue perf
        # DeepFace.build_model(depface_options["model_name"], "facial_recognition")
        # DeepFace.build_model(depface_options["detector_backend"], "face_detector")
        encodings, findings, metrics = process_files(files,
                                                     threshold=threshold, num_processes=processes,
                                                     depface_options=depface_options,
                                                     pre_encodings=pre_encodings,
                                                     pre_findings=pre_findings)
        metrics["Processes"] = processes
        ds.update_findings(findings)
        ds.update_encodings(encodings)
        ds.save_run_info(metrics)

        generate_report(Path(path).absolute(), findings, metrics, report_file)
