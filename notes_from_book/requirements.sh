#!/bin/bash

pip3 install --upgrade pip
pip3 install --default-timeout=1000 scikit-learn tensorflow tensorboard keras

sudo apt install -y python3-pandas python3-scipy python3-numpy python3-seaborn python3-keras libcudart11.0
