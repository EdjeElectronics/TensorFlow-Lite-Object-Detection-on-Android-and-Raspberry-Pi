#!/bin/bash

source tflite1-env/bin/activate

python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model --threshold=0.6