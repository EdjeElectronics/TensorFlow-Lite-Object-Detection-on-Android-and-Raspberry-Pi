# TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
A tutorial showing how to train, convert, and run TensorFlow Lite object detection models on Android devices, the Raspberry Pi, and more!

<Fancy GIF or image showing an example of TF Lite object detector in action - to be inserted here later!>


## Currently Under Construction

This guide is currently under construction! Here are expected dates for when major portions will be completed:

| Part | Description | Expected Completion Date |
|------|-------------|--------------------------|
|Part 1| How to Train, Convert, and Run Custom TensorFlow Lite Object Detection Models on Windows 10|September 22, 2019 |
|Part 2| How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi with [Coral USB Accelerator](https://coral.withgoogle.com/products/accelerator/)|September 29, 2019 |
|Part 3| How to Run TensorFlow Lite Object Detection Models on Android Devices|October 6, 2019 |

I will also be creating a series of YouTube videos that walk through each step of the guide.


## Introduction
TensorFlow Lite is an optimized framework for deploying lightweight deep learning models on resource-constrained edge devices. TensorFlow Lite models have faster inference time and require less processing power, so they can be used to obtain faster performance in realtime applications. This guide provides step-by-step instructions for how train a custom TensorFlow Object Detection model, convert it into an optimized format that can be used by TensorFlow Lite, and run it on Android phones or the Raspberry Pi.

The guide is broken into three major portions. Each portion will have its own dedicated README file in this repository.
1. How to Train, Convert, and Run Custom TensorFlow Lite Object Detection Models on Windows 10
2. How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi (with optional Coral USB Accelerator)
3. How to Run TensorFlow Lite Object Detection Models on Android Devices

This repository also contains Python code for running the newly converted TensorFlow Lite model to perform detection on images, videos, or webcam feeds.

### A Note on Versions
I used TensorFlow v1.13 while creating this guide, because TF v1.13 is a stable version that has great support from Anaconda. I will also periodically update the guide to make sure it works with newer versions of TensorFlow. 

The TensorFlow team is always hard at work releasing updated versions of TensorFlow. I recommend picking one version and sticking with it for all your TensorFlow projects. Every part of this guide should work with newer or older versions, but you may need to use different versions of the tools needed to run or build TensorFlow (CUDA, cuDNN, bazel, etc). Google has provided a list of build configurations for [Linux](https://www.tensorflow.org/install/source#linux), [macOS](https://www.tensorflow.org/install/source#macos), and [Windows](https://www.tensorflow.org/install/source_windows#tested_build_configurations) that show which tool versions were used to build and run each version of TensorFlow.

## Part 1 - How to Train, Convert, and Run Custom TensorFlow Lite Object Detection Models on Windows 10
Part 1 of this guide gives instructions for training and deploying your own custom TensorFlow Lite object detection model on a Windows 10 PC. There are three primary steps to this process:
1. Train a quantized SSD-MobileNet model using TensorFlow, and export frozen graph for TensorFlow Lite
2. Build TensorFlow from source on your PC
3. Use TensorFlow Lite Optimizing Converter (TOCO) to create optimzed TensorFlow Lite model

This portion is a continuation of my previous guide: [How To Train an Object Detection Model Using TensorFlow on Windows 10](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10). I'll assume you have already set up TensorFlow to train a custom object detection model as described in my previous guide, including:
* Setting up an Anaconda virtual environment for training
* Setting up TensorFlow directory structure
* Gathering and labeling training images
* Preparing training data (generating TFRecords and label map)

 This tutorial uses the same Anaconda virtual environment, files, and directory structure that was set up in the previous one.
 
Parts 2 and 3 of this guide will go on to show how to deploy this newly trained TensorFlow Lite model on the Raspberry Pi or an Android device. If you're not feeling up to training and converting your own TensorFlow Lite model, you can skip Part 1 and use my custom-trained TFLite bird detection model (link to be added later) or use the [TF Lite starter detection model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip) (taken from https://www.tensorflow.org/lite/models/object_detection/overview) for Part 2 or Part 3.
 
### Step 1: Train Quantized SSD-MobileNet Model and Export Frozen TensorFlow Lite Graph
First, we’ll use transfer learning to train a “quantized” SSD-MobileNet model. Quantized models use 8-bit integer values instead of 32-bit floating values within the neural network, allowing them to run much more efficiently on GPUs or specialized TPUs (TensorFlow Processing Units).

You can also use a standard SSD-MobileNet model (V1 or V2), but it will not run quite as fast as the quantized model. Also, you will not be able to run it on the Google Coral TPU Accelerator. If you’re using an SSD-MobileNet model that has already been trained, you can skip to Step 1d *(still need to add link)* of this guide.

#### Step 1a. Download and extract quantized SSD-MobileNet model
As I mentioned prevoiusly, this guide assumes you have already followed my [previous TensorFlow tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) and set up the Anaconda virtual environment and full directory structure needed for using the TensorFlow Object Detection API. If you've done so, you should have a folder at C:\tensorflow1\models\research\object_detection that has everything needed for training. (If you used a different base folder name than "tensorflow1", that's fine - just make sure you continue to use that name throughout this guide.)

If you don't have this folder, please go to my [previous tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) and work through at least Steps 1 and 2. If you'd like to train your own model to detect custom objects, you'll also need to work through Steps 3 and 4. If you don't want to train your own model but want to practice the process for converting a model to TensorFlow Lite, you can download the quantized MobileNet-SSD (see next paragraph) and then skip to Step 1d *(still need to add a link)*.

Google provides several quantized object detection models in their [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). This tutorial will use the SSD-MobileNet-V2-Quantized-COCO model. Download the model [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz). **Note: TensorFlow Lite does NOT support RCNN models such as Faster-RCNN! It only supports SSD models.** 

Move the downloaded .tar.gz file to the C:\tensorflow1\models\research\object_detection folder. (Henceforth, this folder will be referred to as the “\object_detection” folder.)  Unzip the .tar.gz file using a file archiver like WinZip or 7-Zip. After the file has been fully unzipped, you should have a folder called "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03" within the \object_detection folder.
