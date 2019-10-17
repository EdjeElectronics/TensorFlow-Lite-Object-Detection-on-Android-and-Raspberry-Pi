# How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi (with Optional Coral USB Accelerator)

*(Insert fancy .gif showing TFLite and EdgeTPU performance here!)*

This guide provides step-by-step instructions for how to set up TensorFlow Lite on the Raspberry Pi and use it to run object detection models. It also shows how to set up the Coral USB Accelerator on the Pi and run Edge TPU detectopm models. It works for the Raspberry Pi 3 and Raspberry Pi 4 running either Rasbpian Buster or Rasbpian Stretch.

This guide is the second part of my larger TensorFlow Lite tutorial series:

1. [How to Train, Convert, and Run Custom TensorFlow Lite Object Detection Models on Windows 10](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#part-1---how-to-train-convert-and-run-custom-tensorflow-lite-object-detection-models-on-windows-10)
2. How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi (with Optional Coral USB Accelerator) *<--- You are here!*
3. How to Run TensorFlow Lite Object Detection Models on Android Devices

TensorFlow Lite (TFLite) models run much faster than regular TensorFlow models on the Raspberry Pi. On a Pi 3B+, average framerate improves from 1.37 FPS to 2.19 FPS when using TFLite. On a Pi 4 4GB, average framerate improves from 2.17 FPS to 4.68 FPS. When the Coral USB Accelerator is used, it increase the Pi 3B+ framerate to 12.9 and the Pi 4 4GB framerate to a whopping 34.4 FPS! A full comparison of framerates can be seen in my performance comparison YouTube video. *(link to be added later)*

This portion of the guide is split in to two parts:

* Part 1. Run TensorFlow Lite Object Detection Models on the Raspberry Pi
* Part 2. Run Edge TPU Object Detection Models on the Raspberry Pi Using the Coral USB Accelerator

This repository also includes scripts for running the TFLite and Edge TPU models on images, videos, or webcam/Picamera feeds.

## Part 1 - How to Set Up and Run TensorFlow Lite Object Detection Models on the Raspberry Pi

Setting up TensorFlow Lite on the Raspberry Pi is much easier than regular TensorFlow! These are the steps needed to set up TensorFlow Lite:

1. Update the Raspberry Pi
2. Install TensorFlow Lite dependencies and OpenCV
3. Install TensorFlow Lite runtime
4. Set up TensorFlow Lite detection model
5. Run TensorFlow Lite model!

### 1. Update the Raspberry Pi
First, the Raspberry Pi needs to be fully updated. Open a terminal and issue:
```
sudo apt-get update
sudo apt-get dist-upgrade
```
Depending on how long it’s been since you’ve updated your Pi, the upgrade could take anywhere between a minute and an hour.

*(Add picture here!)*

### 2. Install TensorFlow Lite dependencies and OpenCV
Next, we'll install OpenCV and the package dependencies for TensorFlow Lite. OpenCV is not needed to run TensorFlow Lite, but the object detection scripts in this repository use it.




