# Part 2 - How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi (with Optional Coral USB Accelerator)

**Section 2 of this guide, which shows how to use the Coral USB Accelerator, is still under construction! Expected completion date: 10/22/19.** In the meantime, you can use the guide [here](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi).

*(Insert fancy .gif showing TFLite and EdgeTPU performance here!)*

## Introduction
This guide provides step-by-step instructions for how to set up TensorFlow Lite on the Raspberry Pi and use it to run object detection models. It also shows how to set up the Coral USB Accelerator on the Pi and run Edge TPU detectopm models. It works for the Raspberry Pi 3 and Raspberry Pi 4 running either Rasbpian Buster or Rasbpian Stretch.

This guide is the second part of my larger TensorFlow Lite tutorial series:

1. [How to Train, Convert, and Run Custom TensorFlow Lite Object Detection Models on Windows 10](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#part-1---how-to-train-convert-and-run-custom-tensorflow-lite-object-detection-models-on-windows-10)
2. How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi (with Optional Coral USB Accelerator) *<--- You are here!*
3. How to Run TensorFlow Lite Object Detection Models on Android Devices

TensorFlow Lite (TFLite) models run much faster than regular TensorFlow models on the Raspberry Pi. You can see a comparison of framerates obtained using regular TensorFlow, TensorFlow Lite, and Coral USB Accelerator models in my [TensorFlow Lite Performance Comparison YouTube video](https://www.youtube.com/watch?v=TiOKvOrYNII).

This portion of the guide is split in to two sections:

* [Section 1. Run TensorFlow Lite Object Detection Models on the Raspberry Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md#part-1---how-to-set-up-and-run-tensorflow-lite-object-detection-models-on-the-raspberry-pi)
* Section 2. Run Edge TPU Object Detection Models on the Raspberry Pi Using the Coral USB Accelerator

This repository also includes scripts for running the TFLite and Edge TPU models on images, videos, or webcam/Picamera feeds. I

## Section 1 - How to Set Up and Run TensorFlow Lite Object Detection Models on the Raspberry Pi

Setting up TensorFlow Lite on the Raspberry Pi is much easier than regular TensorFlow! These are the steps needed to set up TensorFlow Lite:

- 1a. Update the Raspberry Pi
- 1b. Download this repository and create virtual environment
- 1c. Install TensorFlow Lite dependencies and OpenCV
- 1d. Install TensorFlow Lite runtime
- 1e Set up TensorFlow Lite detection model
- 1f. Run TensorFlow Lite model!

### Step 1a. Update the Raspberry Pi
First, the Raspberry Pi needs to be fully updated. Open a terminal and issue:
```
sudo apt-get update
sudo apt-get dist-upgrade
```
Depending on how long it’s been since you’ve updated your Pi, the update could take anywhere between a minute and an hour. 

While we're at it, let's make sure the camera interface is enabled in the Raspberry Pi Configuration menu. Click the Pi icon in the top left corner of the screen, select Preferences -> Raspberry Pi Configuration, and go to the Interfaces tab and verify Camera is set to Enabled. If it isn't, enable it now, and reboot the Raspberry Pi.

<p align="center">
  <img src="/doc/camera_enabled.png">
</p>

### Step 1b. Download this repository and create virtual environment

Next, clone this GitHub repository by issuing the following command. The repository contains the scripts we'll use to run TensorFlow Lite, as well as a shell script that will make installing everything easier. Issue:

```
git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git
```

This downloads everything into a folder called TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi. That's a little long to work with, so rename the folder to "tflite1" and then cd into it:

```
mv TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi tflite1
cd tflite1
```

We'll work in this /home/pi/tflite1 directory for the rest of the guide. Next up is to create a virtual environment called "tflite1-env".

I'm using a virtual environment for this guide because it prevents any conflicts between versions of package libraries that may already be installed on your Pi. As of now (11/3/19), the tflite_runtime library does NOT work if it's installed on the same system as the regular TensorFlow library. Keeping it installed in its own environment allows us to avoid this problem. If you've already installed TensorFlow on the Pi using my [other guide](https://www.youtube.com/watch?v=npZ-8Nj1YwY), you don't have to worry about any conflicts when following this guide.

Install virtualenv by issuing:

```
sudo pip3 install virtualenv
```

Then, create the "tflite1-env" virtual environment by issuing:

```
python3 -m venv tflite1-env
```

This will create a folder called tflite1-env inside the tflite1 directory. The tflite1-env folder will hold all the package libraries for this environment. Next, activate the environment by issuing:

```
source tflite1-env/bin/activate
```

**You'll need to issue the `source tflite1-env/bin/activate` command from inside the /home/pi/tflite1 directory to reactivate the environment every time you open a new terminal window. You can tell when the environment is active by checking if (tflite1-env) appears before the path in your command prompt, as shown in the screenshot below.**

At this point, here's what your tflite1 directory should look like if you issue `ls`.

*(Picture to be added)*

If your directory looks good, it's time to move on to Step 1c!

### Step 1c. Install TensorFlow Lite dependencies and OpenCV
Next, we'll install OpenCV and the package dependencies for TensorFlow Lite. OpenCV is not needed to run TensorFlow Lite, but the object detection scripts in this repository use it to grab images and draw detection results on them.

To make things easier, I wrote a shell script that will automatically download and install all the packages and dependencies. Run it by issuing:

```
bash get_pi_requirements.sh
```

This downloads about 300MB worth of installation files, so it will take a while. Go grab a cup of coffee while it's working! If you'd like to see everything that gets installed, simply open get_pi_dependencies.sh to view the list of packages.

**NOTE: If you get an error while running the `bash get_pi_requirements.sh` command, it's likely because your internet connection timed out, or because the downloaded package data was corrupted. If you get an error, try re-running the command a few more times.**

That was easy! On to the next step.

### Step 1d. Install TensorFlow Lite runtime
Google provides an interpreter-only package for TensorFlow Lite that is drastically smaller than the full TensorFlow package. The reduced TensorFlow Lite runtime is a smaller download and takes less space on the hard drive.

Go to the [Python quickstart page of the official TensorFlow website](https://www.tensorflow.org/lite/guide/python) and scroll to the table with download links for the tflite_runtime wheel files.

If you are running Raspbian Buster (the latest release of Raspberry Pi's OS), download and the Python 3.7 wheel file. If you are running Raspbian Stretch (the older release, which doesn't have Python 3.7 installed by default), download the Python 3.5 wheel file. You can see which OS you have by issuing `lsb_release -a` and checking if the Codename says "stretch" or "buster".

<p align="center">
  <img src="/doc/TFL_download_links.png">
</p>

The file should download to /home/pi/Downloads. Install it using the `pip3 install` command, and use "Tab" to complete the path to the wheel file (so you don't have to type the whole filename in manually).

```
pip3 install /home/pi/Downloads/tflite_runtime (Press "Tab" to complete path to .whl file)
```

The TensorFlow Lite runtime is now installed in our tflite1-env virtual environment!

### Step 1e. Set up TensorFlow Lite detection model
Next, we'll set up the detection model that will be used with TensorFlow Lite. This guide shows how to either download a sample TFLite model provided by Google, or how to use a model that you've trained yourself by following [Part 1 of my TensorFlow Lite tutorial series](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#part-1---how-to-train-convert-and-run-custom-tensorflow-lite-object-detection-models-on-windows-10).

A detection model has two files associated with it: a detect.tflite file (which is the model itself) and a labelmap.txt file (which provides a labelmap for the model). My preferred way to organize the model files is to create a folder (such as "BirdSquirrelRaccoon_TFLite_model") and keep both the detect.tflite and labelmap.txt in that folder. This is also how Google's downloadable sample TFLite model is organized.

#### Option 1. Using Google's sample TFLite model
Google provides a sample quantized SSDLite-MobileNet-v2 object detection model which is trained off the MSCOCO dataset and converted to run on TensorFlow Lite. It can detect and identify 80 different common objects, such as people, cars, cups, etc.

Download the sample model (which can be found on [the Object Detection page of the official TensorFlow website](https://www.tensorflow.org/lite/models/object_detection/overview)) by issuing:

```
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

Unzip it to a folder called "Sample_TFLite_model" by issuing (this command automatically creates the folder):

```
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model
```

Okay, the sample model is all ready to go! 

#### Option 2: Using your own custom-trained model
You can also use a custom object detection model by moving the model folder into the /home/pi/tflite directory. If you followed [Part 1 of my TensorFlow Lite guide](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#part-1---how-to-train-convert-and-run-custom-tensorflow-lite-object-detection-models-on-windows-10) to train and convert a TFLite model on your PC, you should have a folder named "TFLite_model" with a detect.tflite and labelmap.txt file. (It will also have a tflite_graph.pb and tflite_graph.pbtxt file, which are not needed by TensorFlow Lite but can be left in the folder.) 

You can simply copy that folder to a USB drive, insert the USB drive in your Raspberry Pi, and move the folder into the /home/pi/tflite1 directory. (Or you can email it to yourself, or put it on Google Drive, or do whatever your preferred method of file transfer is.) Here's an example of what my "BirdSquirrelRaccoon_TFLite_model" folder looks like in my /home/pi/tflite1 directory: 

*(Add picture of BirdSquirrelRaccoon_TFLite_model in my /home/pi/tflite1 directory)*

Now your custom model is ready to go!

### Step 1f. Run the TensorFlow Lite model!
It's time to see the TFLite object detection model in action! First, free up memory and processing power by closing any applications you aren't using. Also, make sure you have your webcam or Picamera plugged in.

Run the real-time webcam detection script by issuing the following command from inside the /home/pi/tflite1 directory. (Before running the command, make sure the tflite1-env environment is active by checking that (tflite1-env) appears in front of the command prompt.)

```
python3 TFLite_detection_picamera.py --modeldir=Sample_TFLite_model
```

Note: The TFLite_detection_picamera.py will work with either a Picamera or a USB webcam. It is actually going to be removed from this repository, and I will instruct people to use TFLite_detection_webcam.py instead.

(If your model folder has a different name than "Sample_TFLite_model", use that name instead. For example, I would use `--modeldir=BirdSquirrelRaccoon_TFLite_model` to run my custom bird, squirrel, and raccoon detection model.)

After a few moments of initializing, a window will appear showing the webcam feed. Detected objects will have bounding boxes and labels displayed on them in real time.

*(Add gif of object detector in action here?)*

Part 3 of my TensorFlow Lite training guide gives [instructions](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#video) for using the TFLite_detection_image.py and TFLite_detection_video.py scripts. Make sure to use `python3` rather than `python` when running the scripts.

## Section 2 - Run Edge TPU Object Detection Models on the Raspberry Pi Using the Coral USB Accelerator
The [Coral USB Accelerator](https://coral.withgoogle.com/products/accelerator/) is a USB hardware accessory for speeding up TensorFlow models. You can buy one [here (Amazon Associate link)](https://amzn.to/2BuG1Tv). 

*(Add picture of USB Accelerator and the Edge TPU chip)*

The USB Accelerator uses the Edge TPU (tensor processing unit), which is an ASIC (application-specific integrated circuit) chip specially designed with highly parallelized ALUs (arithmetic logic units). While GPUs (graphics processing units) also have many parallelized ALUs, the TPU has one key difference: the ALUs are directly connected to eachother. The output of one ALU can be directly passed to the input of the next ALU without having to be stored and retrieved from a memory buffer. The extreme paralellization and removal of the memory bottleneck means the TPU can perform up to 4 trillion arithmetic operations per second! This is perfect for running deep neural networks, which require millions of multiply-accumulate operations to generate outputs from a single batch of input data. 

My Master's degree was in ASIC design, so the Edge TPU is very interesting to me! If you're a computer architecture nerd like me and want to learn more about the Edge TPU, [here is a great article that explains how it works](https://cloud.google.com/blog/products/ai-machine-learning/what-makes-tpus-fine-tuned-for-deep-learning).

It makes object detection models run WAY faster, and it's easy to set up. These are the steps we'll go through to set up the Coral USB Accelerator:

2a. Install libedgetpu library
2b. Set up Edge TPU detection model
2c. Run super-speed detection!

### Step 2a. Install libedgetpu library
First, we'll download and install the Edge TPU runtime, which is the library needed to interface with the USB Acccelerator. These instructions follow the [USB Accelerator setup guide](https://coral.withgoogle.com/docs/accelerator/get-started/) from official Coral website.

Add the Coral package repository to your apt-get distribution list by issuing the following commands:

```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
```

Install the libedgetpu library by issuing:

```
sudo apt-get install libedgetpu1-std
```

Note: You can also install the libedgetpu1-max library, which runs the USB Accelerator at an overclocked frequency, allowing it to achieve even faster framerates. However, it also causes the USB Accelerator to get very hot. I haven't tried this option yet, but once I do, I will add some information here saying how much higher the framerate is and how hot the Accelerator gets. If you want to use the libedgetpu-max library, install it by using `sudo apt-get install libedgetpu1-max`. (You can't have both the -std and the -max libraries installed. If you install the -max library, the -std library will automatically be uninstalled.)

Alright! Now that the libedgetpu runtime is installed, it's time to set up an Edge TPU detection model to use it with.

### Step 2b. Set up Edge TPU detection model

