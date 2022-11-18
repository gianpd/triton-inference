FROM python:3.10.8-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./requirements.txt

# # install cv2 dependencies
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install python requirements
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY code/ ./
ENTRYPOINT ["/bin/bash", "bootstrap.sh"]

