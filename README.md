# MOM.ai

MOM.ai is an AI-enabled tool to tag multiple speakers from a pre-recorded meeting audio and generate the transcript along with the MoM summary

## Prerequisite
Ubuntu\
Python 3\
At least 8GB of RAM\
At least 2GB of free disk space

## Installation

Clone the git repo and navigate to the project root

Setup a virtual python environment if required

Install redis server if don't already have it. The server should run on localhost:6379
```bash
sudo apt-get update
sudo apt-get install redis-server
```

Use the python package manager [pip](https://pip.pypa.io/en/stable/) to install MOM.ai python requirements.
```bash
pip install -r requirements.txt
```

Run django migrations to update the database
```bash
python manage.py makemigrations
python manage.py migrate
```

Start the celery workers as background tasks by running the command:
```bash
nohup celery -A prototype worker -l info &
```

Finally run the django development server
```bash
python manage.py runserver
```

## Usage

By default the development server runs on 127.0.0.1:8000

In a browser navigate to http://127.0.0.1:8000/home to begin using the app

You may only upload .wav file for processing

The web app will take a long time to load initially as it has to download an AI model of 1.2GB size
