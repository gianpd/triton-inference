#!/bin/bash

KEY_CONTAINER_NAME="NJKeyDB-server"
if [[ -z $1 ]];
then
echo "key DB name container name not provided. Setting KEY_CONTAINER_NAME to $KEY_CONTAINER_NAME."
else
KEY_CONTAINER_NAME=$1
fi
echo Building fake grabber
docker pull fake-grabber --name fake-grabber
docker build --rm --no-cache -t fake-grabber .
echo Running fake grabber with args: "$@"
HOST=$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' $KEY_CONTAINER_NAME) # get the keydb host
echo HOST "$HOST"
if [[ -z $2 ]];
then
echo "Running fake grabber container without test"
docker run --rm --name fake-grabber-container -t fake-grabber $HOST
else
echo "Running fake grabber container with test"
docker run --rm --name fake-grabber-container -t fake-grabber $HOST $2
fi