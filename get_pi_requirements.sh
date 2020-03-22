#!/bin/bash

# Get packages required for OpenCV
echo "!!! Installing dependencies for OpenCV !!!"
sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install qt4-dev-tools libatlas-base-dev

# Need to get an older version of OpenCV because version 4 has errors
echo "!!! Installing OpenCV version 3.4.6.27 !!!"
pip3 install opencv-python==3.4.6.27

# Get packages required for TensorFlow
# Using the tflite_runtime packages available at https://www.tensorflow.org/lite/guide/python
# Will change to just 'pip3 install tensorflow' once newer versions of TF are added to piwheels

#pip3 install tensorflow

version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

echo "!!! Using Python version " $version " !!!"

if [ $version == "3.7" ]; then
echo "!!! Installing TFLite for Python 3.7 !!!"
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
elif [ $version == "3.5" ]; then
echo "!!! Installing TFLite for Python 3.5 !!!"
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_armv7l.whl
else
echo "!!! Suitable version of python not installed. Please install Python 3.5 or 3.7 !!!" 
fi

echo "!!! Installation complete !!!"
