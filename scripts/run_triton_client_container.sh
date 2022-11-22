#!/bin/bash
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:22.08-py3-sdk bash