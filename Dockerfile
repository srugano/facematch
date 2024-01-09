FROM python:3.10-slim

WORKDIR /usr/src/app

COPY . /usr/src/app
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx build-essential cmake libopencv-dev libopencv-highgui-dev libopencv-imgcodecs-dev libopencv-core-dev\
    && rm -rf /var/lib/apt/lists/*
RUN pip install dlib
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENV PYTHONUNBUFFERED 1

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
