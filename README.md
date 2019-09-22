# TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
A tutorial showing how to train, convert, and run TensorFlow Lite object detection models on Android devices, the Raspberry Pi, and more!

<Fancy GIF or image showing an example of TF Lite object detector in action - to be inserted here later!>


## Currently Under Construction

This guide is currently under construction! Here are expected dates for when major portions will be completed:

| Part | Description | Expected Completion Date |
|------|-------------|--------------------------|
|Part 1| How to Train, Convert, and Run Custom TensorFlow Lite Object Detection Models|September 22, 2019 |
|Part 2| How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi with [Coral USB Accelerator](https://coral.withgoogle.com/products/accelerator/)|September 29, 2019 |
|Part 3| How to Run TensorFlow Lite Object Detection Models on Android Devices|October 6, 2019 |

I will also be creating a series of YouTube videos that walk through each step of the guide.


## Introduction
TensorFlow Lite is an optimized framework for deploying lightweight deep learning models on resource-constrained edge devices. TensorFlow Lite models have faster inference time and require less processing power, so they can be used to obtain faster performance in realtime applications. This guide provides step-by-step instructions for how train a custom TensorFlow Object Detection model, convert it into an optimized format that can be used by TensorFlow Lite, and run it on Android phones or the Raspberry Pi.

The guide is broken into three major portions. Each portion will have its own dedicated README file in this repository.
1. How to Train, Convert, and Run Custom TensorFlow Lite Object Detection Models
2. How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi (with optional Coral USB Accelerator)
3. How to Run TensorFlow Lite Object Detection Models on Android Devices

This repository also contains Python code for running the newly converted TensorFlow Lite model to perform detection on images, videos, or webcam feeds.

### A Note on Versions
I used TensorFlow v1.13 while creating this guide, because TF v1.13 is a stable version that has great support from Anaconda. I will also periodically update the guide to make sure it works with newer versions of TensorFlow. 

The TensorFlow team is always hard at work releasing updated versions of TensorFlow. I recommend picking one version and sticking with it for all your TensorFlow projects. Every part of this guide should work with newer or older versions, but you may need to use different versions of the tools needed to run or build TensorFlow (CUDA, cuDNN, bazel, etc). Google has provided a list of build configurations for [Linux](https://www.tensorflow.org/install/source#linux), [macOS](https://www.tensorflow.org/install/source#macos), and [Windows](https://www.tensorflow.org/install/source_windows#tested_build_configurations) that show which tool versions were used to build and run each version of TensorFlow.

## Part 1 - How to Train, Convert, and Run Custom TensorFlow Lite Object Detection Models
Part 1 of this guide gives instructions for training and deploying your own custom TensorFlow Lite object detection model on a Windows PC. There are three primary steps to this process:
1. Train a quantized SSD-MobileNet model using TensorFlow, and export frozen graph for TensorFlow Lite
2. Build TensorFlow from source on your PC
3. Use TensorFlow Lite Optimizing Converter (TOCO) to create optimzed TensorFlow Lite model

This portion is a continuation of my previous guide: [How To Train an Object Detection Model Using TensorFlow on Windows 10](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10). I'll assume you have already set up TensorFlow to train a custom object detection model as described in my previous guide, including:
* Setting up an Anaconda virtual environment for training
* Gathering and labeling training images
* Preparing training data (generating TFRecords and label map)

 This tutorial uses the same Anaconda virtual environment, files, and directory structure that was set up in the previous one.
 
Parts 2 and 3 of this guide will go on to show how to deploy this newly trained TensorFlow Lite model on the Raspberry Pi or an Android device. If you're not feeling up to training and converting your own TensorFlow Lite model, you can skip Part 1 and use my custom-trained TFLite bird detection model (link to be added later) or use the [Quantized SSD MobileNet model from the Detection Model Zoo](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz) for Part 2 or Part 3.
 
### Step 1: Train Quantized SSD-MobileNet Model and Export Frozen TensorFlow Lite Graph
First, we’ll use transfer learning to train a “quantized” SSD-MobileNet model. Quantized models use 8-bit integer values instead of 32-bit floating values within the neural network, allowing them to run much more efficiently on GPUs or specialized TPUs (TensorFlow Processing Units).

 **You can also use a standard SSD-MobileNet model (V1 or V2), but it will not run quite as fast as the quantized model. Also, you will not be able to run it on the Google Coral TPU Accelerator. If you’re using an SSD-MobileNet model that has already been trained, you can skip to step 1d (need to add link) of this guide.**

#### Step 1a. Download and extract quantized SSD-MobileNet model
