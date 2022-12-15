#!/bin/bash

echo "Building TritonClient docker image"
docker build --force-rm --network host --no-cache -t nj-triton-client-container .

echo "Running TritonClient container"
docker run --rm --network host -v /tmp/KeyDB/:/workspace/tmp/docker nj-triton-client-container