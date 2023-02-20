# TensorFlow Lite Object Detection on Android and Raspberry Pi
Train your own TensorFlow Lite object detection models and run them on the Raspberry Pi, Android phones, and other edge devices! 

<p align="center">
   <img src="doc/BSR_demo.gif">
</p>

Get started with training on Google Colab by clicking the icon below, or [click here to go straight to the YouTube video that provides step-by-step instructions](https://youtu.be/XZ7FYAMCc4M).

<a href="https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Introduction
TensorFlow Lite is an optimized framework for deploying lightweight deep learning models on resource-constrained edge devices. TensorFlow Lite models have faster inference time and require less processing power than regular TensorFlow models, so they can be used to obtain faster performance in realtime applications. 

This guide provides step-by-step instructions for how train a custom TensorFlow Object Detection model, convert it into an optimized format that can be used by TensorFlow Lite, and run it on edge devices like the Raspberry Pi. It also provides Python code for running TensorFlow Lite models to perform detection on images, videos, web streams, or webcam feeds.

## Step 1. Train TensorFlow Lite Models
### Using Google Colab (recommended)

The easiest way to train, convert, and export a TensorFlow Lite model is using Google Colab. Colab provides you with a free GPU-enabled virtual machine on Google's servers that comes pre-installed with the libraries and packages needed for training.

I wrote a [Google Colab notebook](./Train_TFLite2_Object_Detction_Model.ipynb) that can be used to train custom TensorFlow Lite models. It goes through the process of preparing data, configuring a model for training, training the model, running it on test images, and exporting it to a downloadable TFLite format so you can deploy it to your own device. It makes training a custom TFLite model as easy as uploading an image dataset and clicking Play on a few blocks of code!

<a href="https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Open the Colab notebook in your browser by clicking the icon above. Work through the instructions in the notebook to start training your own model. Once it's trained and exported, visit the [Setup TFLite Runtime Environment](#step-2-setup-tflite-runtime-environment-on-your-device) section to learn how to deploy it on your PC, Raspberry Pi, Android phone, or other edge devices.

### Using a Local PC
The old version of this guide shows how to set up a TensorFlow training environment locally on your PC. Be warned: it's a lot of work, and the guide is outdated. [Here's a link to the local training guide.](doc/local_training_guide.md)

## Step 2. Setup TFLite Runtime Environment on Your Device
Once you have a trained `.tflite` model, the next step is to deploy it on a device like a computer, Raspberry Pi, or Android phone. To run the model, you'll need to install the TensorFlow or the TensorFlow Lite Runtime on your device and set up the Python environment and directory structure to run your application in. The [deploy_guides](deploy_guides) folder in this repository has step-by-step guides showing how to set up a TensorFlow environment on several different devices. Links to the guides are given below.

### Raspberry Pi
Follow the [Raspberry Pi setup guide](deploy_guides/Raspberry_Pi_Guide.md) to install TFLite Runtime on a Raspberry Pi 3 or 4 and run a TensorFlow Lite model. This guide also shows how to use the Google Coral USB Accelerator to greatly increase the speed of quantized models on the Raspberry Pi.

### Windows
Follow the instructions in the [Windows TFLite guide](deploy_guides/Windows_TFLite_Guide.md) to set up TFLite Runtime on your Windows PC using Anaconda!

### macOS
Still to come!

### Linux
Still to come!

### Android
Still to come!

### Embedded Devices
Still to come!

## Step 3. Run TensorFlow Lite Models!
There are four Python scripts to run the TensorFlow Lite object detection model on an image, video, web stream, or webcam feed. The scripts are based off the label_image.py example given in the [TensorFlow Lite examples GitHub repository](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py).

* [TFLite_detection_image.py](TFLite_detection_image.py)
* [TFLite_detection_video.py](TFLite_detection_video.py)
* [TFLite_detection_stream.py](TFLite_detection_stream.py)
* [TFLite_detection_webcam.py](TFLite_detection_webcam.py)

The following instructions show how to run the scripts. These instructions assume your .tflite model file and labelmap.txt file are in the `TFLite_model` folder in your `tflite1` directory as per the instructions given in the [Setup TFLite Runtime Environment](#step-2-setup-tflite-runtime-environment-on-your-device) guide.

<p align="center">
   <img width="500" src="doc/squirrels!!.png">
</p>

If you’d like try using the sample TFLite object detection model provided by Google, simply download it [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip), unzip it to the `tflite1` folder, and rename it to `TFLite_model`. Then, use `--modeldir=coco_ssd_mobilenet_v1_1.0_quant_2018_06_29` rather than `--modeldir=TFLite_model` when running the script. 

<details>
   <summary>Webcam</summary>
Make sure you have a USB webcam plugged into your computer. If you’re on a laptop with a built-in camera, you don’t need to plug in a USB webcam. 

From the `tflite1` directory, issue: 

```
python TFLite_detection_webcam.py --modeldir=TFLite_model 
```

After a few moments of initializing, a window will appear showing the webcam feed. Detected objects will have bounding boxes and labels displayed on them in real time.
</details>

<details>
   <summary>Video</summary>
To run the video detection script, issue:

```
python TFLite_detection_video.py --modeldir=TFLite_model
```

A window will appear showing consecutive frames from the video, with each object in the frame labeled. Press 'q' to close the window and end the script. By default, the video detection script will open a video named 'test.mp4'. To open a specific video file, use the `--video` option:

```
python TFLite_detection_video.py --modeldir=TFLite_model --video='birdy.mp4'
```

Note: Video detection will run at a slower FPS than realtime webcam detection. This is mainly because loading a frame from a video file requires more processor I/O than receiving a frame from a webcam.
</details>

<details>
   <summary>Web stream</summary>
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
</details>

<details>
   <summary>Image</summary>
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

To save labeled images and a text file with detection results for each image, use the `--save_results` option. The results will be saved to a folder named `<imagedir>_results`. This works well if you want to check your model's performance on a folder of images and use the results to calculate mAP with the [calculate_map_catchuro.py](./util_scripts) script. For example:

```
python TFLite_detection_image.py --modeldir=TFLite_model --imagedir=squirrels --save_results
```

The `--noshow_results` option will stop the program from displaying images.
</details>

**See all command options**

For more information on options that can be used while running the scripts, use the `-h` option when calling them. For example:

```
python TFLite_detection_image.py -h
```

If you encounter errors, please check the [FAQ section](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#FAQs) of this guide. It has a list of common errors and their solutions. If you can successfully run the script, but your object isn’t detected, it is most likely because your model isn’t accurate enough. The FAQ has further discussion on how to resolve this.

## Examples
(Still to come!) Please see the [examples](examples) folder for examples of how to use your TFLite model in basic vision applications.

## FAQs
<details>
<summary>What's the difference between the TensorFlow Object Detection API and TFLite Model Maker?</summary>
<br>
Google provides a set of Colab notebooks for training TFLite models called [TFLite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker). While their object detection notebook is straightfoward and easy to follow, using the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) for creating models provides several benefits:

* TFLite Model Maker only supports EfficientDet models, which aren't as fast as SSD-MobileNet models.
* Training models with the Object Detection API generally results in better model accuracy.
* The Object Detection API provides significantly more flexibility in model and training configuration (training steps, learning rate, model depth and resolution, etc).
* Google still [recommends using the Object Detection API](https://www.tensorflow.org/lite/examples/object_detection/overview#fine-tuning_models_on_custom_data) as the formal method for training models with large datasets.
</details>

<details>
<summary>What's the difference between training, transfer learning, and fine-tuning?</summary>
<br>
Using correct terminology is important in a complicated field like machine learning. In this notebook, I use the word "training" to describe the process of teaching a model to recognize custom objects, but what we're actually doing is "fine-tuning". The Keras documentation gives a [good example notebook](https://keras.io/guides/transfer_learning/) explaining the difference between each term.

Here's my attempt at defining the terms:

* **Training**: The process of taking a full neural network with randomly initialized weights, passing in image data, calculating the resulting loss from its predictions on those images, and using backpropagation to adjust the weights in every node of the network and reduce its loss. In this process, the network learns how to extract features of interest from images and correlate those features to classes. Training a model from scratch typically takes millions of training steps and a large dataset of 100,000+ images (such as ImageNet or COCO). Let's leave actual training to companies like Google and Microsoft!
* **Transfer learning**: Taking a model that has already been trained, unfreezing the last layer of the model (i.e. making it so only the last layer's weights can be modified), and retraining the last layer with a new dataset so it can learn to identify new classes. Transfer learning takes advantage of the feature extraction capabilities that have already been learned in the deep layers of the trained model. It takes the extracted features and recategorizes them to predict new classes.
* **Fine-tuning**: Fine-tuning is similar to transfer learning, except more layers are unfrozen and retrained. Instead of just unfreezing the last layer, a significant amount of layers (such as the last 20% to 50% of layers) are unfrozen. This allows the model to modify some of its feature extraction layers so it can extract features that are more relevant to the classes its trying to identify. This notebook (and the TensorFlow Object Detection API) uses fine-tuning.

In general, I like to use the word "training" instead of "fine-tuning", because it's more intuitive and understandable to new users.
</details>

<details>
<summary>Should I get a Google Colab Pro subscription?</summary>
<br>
If you plan to use Colab frequently for training models, I recommend getting a Colab Pro subscription. It provides several benefits:

* Idle Colab sessions remain connected for longer before timing out and disconnecting
* Allows for running multiple Colab sessions at once
* Priority access to TPU and GPU-enabled virtual machines
* Virtual machines have more RAM

Colab keeps track of how much GPU time you use, and cuts you off from using GPU-enabled instances once you reach a certain use time. If you get the message telling you you're cut off from GPU instances, then that's a good indicator that you use Colab enough to justify paying for a Pro subscription.
</details>
