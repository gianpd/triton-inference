#!/bin/bash
echo "pull NVDIA docker container image ..."
docker pull NVIDA:nvcr.io/nvidia/tensorflow:22.09-tf2-py3
echo "Run the NVIDIA container ..."
pip install 
docker run --gpus=all --rm -it --net=host -v $PWD:/njdd_od_quantization nvcr.io/nvidia/tensorflow:22.09-tf2-py3 bash