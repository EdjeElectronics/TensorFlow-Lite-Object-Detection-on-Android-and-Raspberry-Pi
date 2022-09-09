# TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
Train your own TensorFlow Lite object detection models and run them on the Raspberry Pi, Android phones, and other edge devices!

<p align="center">
   <img src="doc/BSR_demo.gif">
</p>

**Update (9/2/22):** I wrote a Google Colab notebook that can be used to train custom TensorFlow Lite models. It allows you to train, convert, and test a TFLite model on a Google Colab server, and then download and deploy it to your own device. It's much easier than trying to install and train everything on your local computer! 

**HELP WANTED:** Please try out this notebook and give me feedback on it! See [Issue #135](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/issues/135).

Check it out here: [Train_TFLite2_Object_Detction_Model.ipynb](./Train_TFLite2_Object_Detction_Model.ipynb)

<a href="https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Introduction
TensorFlow Lite is an optimized framework for deploying lightweight deep learning models on resource-constrained edge devices. TensorFlow Lite models have faster inference time and require less processing power than regular TensorFlow models, so they can be used to obtain faster performance in realtime applications. This guide provides step-by-step instructions for how train a custom TensorFlow Object Detection model, convert it into an optimized format that can be used by TensorFlow Lite, and run it on edge devices like the Raspberry Pi.

This repository also contains Python code for running the newly converted TensorFlow Lite model to perform detection on images, videos, web streams, or webcam feeds.

## Step 1. Train TensorFlow Lite Models
### Using Google Colab (recommended)
The easiest way to train, convert, and export a TensorFlow Lite model is using Google Colab. Colab provides you with a free GPU-enabled virtual machine on Google's servers that comes pre-installed with the libraries and packages needed for training.

I wrote a [Google Colab notebook](./Train_TFLite2_Object_Detction_Model.ipynb) that can be used to train custom TensorFlow Lite models. It goes through the process of preparing data, configuring a model for training, training the model, running it on test images, and exporting it to a downloadable TFLite format so you can deploy it to your own device. It makes training a custom TFLite model as easy as uploading an image and clicking Play on a few blocks of code!

<a href="https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Open the Colab notebook in your browser by clicking the icon above. Work through the instructions in the notebook to start training your own model. Once it's trained and exported, visit the [Setup TFLite Runtime Environment](#setup-tflite-runtime-environment-on-your-device) section to learn how to deploy it on your PC, Raspberry Pi, Android phone, or other edge devices.

### Using a Local PC
The old version of this guide shows how to set up a TensorFlow training environment locally on your PC. Be warned: it's a lot of work, and the guide is outdated. [Here's a link to the local training guide.](doc/local_training_guide.md)

## Step 2. Setup TFLite Runtime Environment on Your Device
Once you have a trained `.tflite` model, the next step is to deploy it on a device like a computer, Raspberry Pi, or Android phone. To run the model, you'll need to install the TensorFlow or the TensorFlow Lite Runtime on your device and set up the Python environment and directory structure to run your application in. The [deploy_guides](deploy_guides) folder in this repository has step-by-step guides showing how to set up a TensorFlow environment on several different devices. Links to the guides are given below.

### Raspberry Pi
Follow the [Raspberry Pi setup guide](deploy_guides/Raspberry_Pi_Guide.md) to install TFLite Runtime on a Raspberry Pi 3 or 4 and run a TensorFlow Lite model. This guide also shows how to use the Google Coral USB Accelerator to greatly increase the speed of quantized models on the Raspberry Pi.

### Windows
Still to come!

### macOS
Still to come!

### Linux
Still to come!

### Android
Still to come!

### Embedded Devices
Still to come!

## Step 3. Run TensorFlow Lite Models!
There are four Python scripts to run the TensorFlow Lite object detection model on an image, video, web stream, or webcam feed: [TFLite_detection_image.py](TFLite_detection_image.py), [TFLite_detection_video.py](TFLite_detection_video.py), [TFLite_detection_stream.py](TFLite_detection_stream.py) and [TFLite_detection_wecam.py](TFLite_detection_webcam.py). The scripts are based off the label_image.py example given in the [TensorFlow Lite examples GitHub repository](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py).

The following instructions show how to run the webcam, video, and image scripts. These instructions assume your .tflite model file and labelmap.txt file are in the “TFLite_model” folder in your \object_detection directory as per the instructions given in the [Setup TFLite Runtime Environment](#step-2-setup-tflite-runtime-environment-on-your-device) guide.

If you’d like try using the sample TFLite object detection model provided by Google, simply download it [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip) and unzip it into the \object_detection folder. Then, use `--modeldir=coco_ssd_mobilenet_v1_1.0_quant_2018_06_29` rather than `--modeldir=TFLite_model` when running the script. 

For more information on options that can be used while running the scripts, use the `-h` option when calling the script. For example:

```
python TFLite_detection_image.py -h
```

##### Webcam
Make sure you have a USB webcam plugged into your computer. If you’re on a laptop with a built-in camera, you don’t need to plug in a USB webcam. 

From the \object_detection directory, issue: 

```
python TFLite_detection_webcam.py --modeldir=TFLite_model 
```

After a few moments of initializing, a window will appear showing the webcam feed. Detected objects will have bounding boxes and labels displayed on them in real time.

##### Video
To run the video detection script, issue:

```
python TFLite_detection_image.py --modeldir=TFLite_model
```

A window will appear showing consecutive frames from the video, with each object in the frame labeled. Press 'q' to close the window and end the script. By default, the video detection script will open a video named 'test.mp4'. To open a specific video file, use the `--video` option:

```
python TFLite_detection_image.py --modeldir=TFLite_model --video='birdy.mp4'
```

Note: Video detection will run at a slower FPS than realtime webcam detection. This is mainly because loading a frame from a video file requires more processor I/O than receiving a frame from a webcam.

##### Web stream
To run the script to detect images in a video stream (e.g. a remote security camera), issue: 

```
python TFLite_detection_stream.py --modeldir=TFLite_model --streamurl="http://ipaddress:port/stream/video.mjpeg" 
```

After a few moments of initializing, a window will appear showing the video stream. Detected objects will have bounding boxes and labels displayed on them in real time.

Make sure to update the URL parameter to the one that is being used by your security camera. It has to include authentication information in case the stream is secured.

If the bounding boxes are not matching the detected objects, probably the stream resolution wasn't detected. In this case you can set it explicitly by using the `--resolution` parameter:

```
python TFLite_detection_stream.py --modeldir=TFLite_model --streamurl="http://ipaddress:port/stream/video.mjpeg" --resolution=1920x1080
```

##### Image
To run the image detection script, issue:

```
python TFLite_detection_image.py --modeldir=TFLite_model
```

The image will appear with all objects labeled. Press 'q' to close the image and end the script. By default, the image detection script will open an image named 'test1.jpg'. To open a specific image file, use the `--image` option:

```
python TFLite_detection_image.py --modeldir=TFLite_model --image=squirrel.jpg
```

It can also open an entire folder full of images and perform detection on each image. There can only be images files in the folder, or errors will occur. To specify which folder has images to perform detection on, use the `--imagedir` option:

```
python TFLite_detection_image.py --modeldir=TFLite_model --imagedir=squirrels
```

Press any key (other than 'q') to advance to the next image. Do not use both the --image option and the --imagedir option when running the script, or it will throw an error.

<p align="center">
   <img src="doc/squirrels!!.png">
</p>

If you encounter errors while running these scripts, please check the [FAQ section](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#frequently-asked-questions-and-common-errors) of this guide. It has a list of common errors and their solutions. If you can successfully run the script, but your object isn’t detected, it is most likely because your model isn’t accurate enough. The FAQ has further discussion on how to resolve this.

## FAQs
