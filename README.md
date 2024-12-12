dedupe POC
----------

## Install

    uv venv
    uv sync
    export DEEPFACE_HOME=$PWD


## Run local single process

Create one directory with some images es. data/IMAGES

    dedupe data/IMAGES -p 1


## Run local multiple (4) process

    dedupe data/IMAGES -p 4


## Use Celery

in first shell

    watchmedo auto-restart --directory=./src/ --pattern *.py --recursive -- celery -E -A recognizeapp.c.app worker

in second shell

    dedupe data/IMAGES -p 4 --queue


## Check everything with Flower

in first shell

    watchmedo auto-restart --directory=./src/ --pattern *.py --recursive -- celery -A recognizeapp.c.app flower
