#!/bin/bash

echo "Running triton server container ..."
docker run --gpus=all -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models \
nvcr.io/nvidia/tritonserver:22.08-py3 bash
