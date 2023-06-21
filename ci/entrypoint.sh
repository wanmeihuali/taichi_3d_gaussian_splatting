#!/bin/bash
set -e
# assert TRAIN_CONFIG is set
if [ -z "$TRAIN_CONFIG" ]; then
    echo "TRAIN_CONFIG is not set"
    exit 1
fi
ls /opt/ml/input/data
ls /opt/ml/input/data/training
ls /opt/ml/input/data/training/image
ln -s /opt/ml/input/data/training/image image
python3 gaussian_point_train.py --train_config $TRAIN_CONFIG