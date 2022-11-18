#!/bin/bash
echo args: "$@"
HOST=$1
if [[ -z $2 ]];
then
echo "### Launching fake_grabber.py without test ..."
echo "ENVS: ${HOST}"
python fake_grabber.py --host $HOST
else
echo "### Launching fake_grabber.py with test mode ..."
echo "ENVS: ${HOST}"
python fake_grabber.py --host $HOST --test
fi
