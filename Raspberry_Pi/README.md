# How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi (with Optional Coral USB Accelerator)

*(Insert fancy .gif showing TFLite and EdgeTPU performance here!)*

## Introduction
This guide provides step-by-step instructions for how to set up TensorFlow Lite on the Raspberry Pi and use it to run object detection models. It also shows how to set up the Coral USB Accelerator on the Pi and run Edge TPU detectopm models. It works for the Raspberry Pi 3 and Raspberry Pi 4 running either Rasbpian Buster or Rasbpian Stretch.

This guide is the second part of my larger TensorFlow Lite tutorial series:

1. [How to Train, Convert, and Run Custom TensorFlow Lite Object Detection Models on Windows 10](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#part-1---how-to-train-convert-and-run-custom-tensorflow-lite-object-detection-models-on-windows-10)
2. How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi (with Optional Coral USB Accelerator) *<--- You are here!*
3. How to Run TensorFlow Lite Object Detection Models on Android Devices

TensorFlow Lite (TFLite) models run much faster than regular TensorFlow models on the Raspberry Pi. You can see a comparison of framerates obtained using regular TensorFlow, TensorFlow Lite, and Coral USB Accelerator models in my TensorFlow Lite Performance Comparison YouTube video. *(link to be added later)*

This portion of the guide is split in to two parts:

* [Part 1. Run TensorFlow Lite Object Detection Models on the Raspberry Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/tree/master/Raspberry_Pi#part-1---how-to-set-up-and-run-tensorflow-lite-object-detection-models-on-the-raspberry-pi)
* Part 2. Run Edge TPU Object Detection Models on the Raspberry Pi Using the Coral USB Accelerator

This repository also includes scripts for running the TFLite and Edge TPU models on images, videos, or webcam/Picamera feeds.

## Part 1 - How to Set Up and Run TensorFlow Lite Object Detection Models on the Raspberry Pi

Setting up TensorFlow Lite on the Raspberry Pi is much easier than regular TensorFlow! These are the steps needed to set up TensorFlow Lite:

1. Update the Raspberry Pi and download this repository
2. Install TensorFlow Lite dependencies and OpenCV
3. Install TensorFlow Lite runtime
4. Set up TensorFlow Lite detection model
5. Run TensorFlow Lite model!

### 1. Update the Raspberry Pi and download this repository
First, the Raspberry Pi needs to be fully updated. Open a terminal and issue:
```
sudo apt-get update
sudo apt-get dist-upgrade
```
Depending on how long it’s been since you’ve updated your Pi, the update could take anywhere between a minute and an hour. 

While we're at it, let's make sure the camera interface is enabled in the Raspberry Pi Configuration menu. Clickk the Pi icon in the top left corner of the screen, select Preferences -> Raspberry Pi Configuration, and go to the Interfaces tab and verify Camera is set to Enabled. If it isn't, enable it now, and reboot the Raspberry Pi.

*(Add picture here!)*

Make a new directory called tflite, cd into it, and download this GitHub repository. This repository contains the scripts we'll use to run the TensorFlow, as well as some shell scripts that will make installing everything easier. Issue:

```
mkdir tflite
cd tflite
git clone THIS REPOSITORY!!!
```

### 2. Install TensorFlow Lite dependencies and OpenCV
Next, we'll install OpenCV and the package dependencies for TensorFlow Lite. OpenCV is not needed to run TensorFlow Lite, but the object detection scripts in this repository use it.

*Skipping the rest of this step until I get the shell script written*

### 3. Install TensorFlow Lite runtime
The TensorFlow team provides an interpreter-only package for TensorFlow Lite that is drastically smaller than the full TensorFlow package. The reduced TensorFlow Lite runtime is a smaller download and takes less space on the hard drive. Better yet, it doesn't conflict with regular TensorFlow at all: if you've already [installed TensorFlow using my other guide](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi), you can still install and run the TensorFlow Lite runtime without any problems.

The [Python quickstart page of the official TensorFlow website](https://www.tensorflow.org/lite/guide/python) shows how to install the TensorFlow Lite runtime. If you are running Raspbian Buster (the latest release of Raspberry Pi's OS), download and install the Python 3.7 wheel file. If you are running Raspbian Stretch (the older release, which doesn't have Python 3.7 installed by default), download and install the Python 3.5 wheel file. You can see which OS you have by issuing `lsb_release -a` and checking if the Codename says "stretch" or "buster".

<p align="center">
  <img src="https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/doc/TFL_download_links.png">
</p>

Once you've installed TensorFlow Lite, you can delete the downloaded .whl file.

### 4. Set up TensorFlow Lite detection model
