#!/bin/bash

echo $@

if [[ -z $1 ]];
then
echo "Launching fake grabber without test mode ..."
python fake_grabber.py
else
echo "Launching fake grabber with test mode ..."
python fake_grabber.py --test
fi