FROM python:3.10.8-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./requirements.txt

# # install cv2 dependencies
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install python requirements with no-cache-dir
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

COPY code/ ./
ENTRYPOINT ["/bin/bash", "bootstrap.sh"]

