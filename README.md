# TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
A tutorial showing how to train, convert, and run TensorFlow Lite object detection models on Android devices, the Raspberry Pi, and more!

<Fancy GIF or image showing an example of TF Lite object detector in action - to be inserted here later!>


## Currently Under Construction

This guide is currently under construction! Here are expected dates for when major portions will be completed:

| Part | Description | Expected Completion Date |
|------|-------------|--------------------------|
|Part 1| How to Train, Convert, and Run Custom TensorFlow Lite Object Detection Models on Windows 10|September 23, 2019 |
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

This tutorial uses the same Anaconda virtual environment, files, and directory structure that was set up in the previous one. I'll continue to use my playing card detector as an example. I'll show the steps needed to train, convert, and run a quantized TensorFlow Lite version of my playing card detector model. *(I might use a different example if I can think of a better one!)*
 
Parts 2 and 3 of this guide will go on to show how to deploy this newly trained TensorFlow Lite model on the Raspberry Pi or an Android device. If you're not feeling up to training and converting your own TensorFlow Lite model, you can skip Part 1 and use my custom-trained TFLite bird detection model (link to be added later) or use the [TF Lite starter detection model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip) (taken from https://www.tensorflow.org/lite/models/object_detection/overview) for Part 2 or Part 3.
 
### Step 1: Train Quantized SSD-MobileNet Model and Export Frozen TensorFlow Lite Graph
First, we’ll use transfer learning to train a “quantized” SSD-MobileNet model. Quantized models use 8-bit integer values instead of 32-bit floating values within the neural network, allowing them to run much more efficiently on GPUs or specialized TPUs (TensorFlow Processing Units).

You can also use a standard SSD-MobileNet model (V1 or V2), but it will not run quite as fast as the quantized model. Also, you will not be able to run it on the Google Coral TPU Accelerator. If you’re using an SSD-MobileNet model that has already been trained, you can skip to Step 1d *(still need to add link)* of this guide.

**If you get any errors during this process, please look at the FAQ section at the bottom of this guide! It gives solutions to common errors that occur. (Link to be added)**

#### Step 1a. Download and extract quantized SSD-MobileNet model
As I mentioned prevoiusly, this guide assumes you have already followed my [previous TensorFlow tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) and set up the Anaconda virtual environment and full directory structure needed for using the TensorFlow Object Detection API. If you've done so, you should have a folder at C:\tensorflow1\models\research\object_detection that has everything needed for training. (If you used a different base folder name than "tensorflow1", that's fine - just make sure you continue to use that name throughout this guide.)

<Add picture of what the \object_detection folder should look like>

If you don't have this folder, please go to my [previous tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) and work through at least Steps 1 and 2. If you'd like to train your own model to detect custom objects, you'll also need to work through Steps 3, 4, and 5. If you don't want to train your own model but want to practice the process for converting a model to TensorFlow Lite, you can download the quantized MobileNet-SSD (see next paragraph) and then skip to Step 1d *(still need to add a link)*.

Google provides several quantized object detection models in their [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). This tutorial will use the SSD-MobileNet-V2-Quantized-COCO model. Download the model [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz). **Note: TensorFlow Lite does NOT support RCNN models such as Faster-RCNN! It only supports SSD models.** 

Move the downloaded .tar.gz file to the C:\tensorflow1\models\research\object_detection folder. (Henceforth, this folder will be referred to as the “\object_detection” folder.)  Unzip the .tar.gz file using a file archiver like WinZip or 7-Zip. After the file has been fully unzipped, you should have a folder called "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03" within the \object_detection folder.

#### Step 1b. Configure training
If you're training your own TensorFlow Lite model, make sure the following items from my [previous guide](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) have been completed:
* Train and test images and their XML label files are placed in the \object_detection\images\train and \object_detection\images\test folders
* train_labels.csv and test_labels.csv have been generated and are located in the \object_detection\images folder
* train.record and test.record have been generated and are located in the \object_detection folder
* labelmap.pbtxt file has been created and is located in the \object_detection\training folder
* proto files in \object_detection\protos have been generated

If you have any questions about these files or don’t know how to generate them, [Steps 2, 3, 4, and 5 of my previous tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) show how they are all created.

Copy the ssd_mobilenet_v2_quantized_300x300_coco.config file from the \object_detection\samples\configs folder to the \object_detection\training folder. Then, open the file using a text editor. *(Actually, I should probably instruct people to get the config file from the Model Zoo download rather than \samples\configs)*

Make the following changes to the ssd_mobilenet_v2_quantized_300x300_coco.config file. Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model! Also, the paths must be in double quotation marks ( " ), not single quotation marks ( ' ).

* Line 9. Change num_classes to the number of different objects you want the classifier to detect. For my card detector example, there are six classes, so I set num_classes: 6
* Line 141. Change batch_size: 24 to batch_size: 6 . The smaller batch size will prevent OOM (Out of Memory) errors during training. 
* Line 156. Change fine_tune_checkpoint to:
  * fine_tune_checkpoint: "C:/tensorflow1/models/research/object_detection/ ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.ckpt"
* Line 175. Change input_path to:
  * input_path: "C:/tensorflow1/models/research/object_detection/train.record"
* Line 177. Change label_map_path to:
  * label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
* Line 181. Change num_examples to the number of images you have in the \images\test directory. For my card detector example, there are 67 images, so I set num_examples: 67.
* Line 189. Change input_path to:
  * input_path: "C:/tensorflow1/models/research/object_detection/test.record"
* Line 191. Change label_map_path to:
  * label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
  
  Save and exit the training file after the changes have been made.
  
#### Step 1c. Run training in Anaconda virtual environment
All that's left to do is train the model! First, move the “train.py” file from the \object_detection\legacy folder into the main \object_detection folder.
  
Then, open a new Anaconda Prompt window by searching for “Anaconda Prompt” in the Start menu and clicking on it. Activate the “tensorflow1” virtual environment (which was set up in my [previous tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)) by issuing: 

```
activate tensorflow1
```

Then, set the PYTHONPATH environment variable by issuing:

```
set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```

Next, change directories to the \object_detection folder:

```
cd C:\tensorflow1\models\research\object_detection
```

Finally, train the model by issuing:

```
python train.py --logtostderr –train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config
```

If everything was set up correctly, the model will begin training after a couple minutes of initialization.

<Picture of training in progress to be added!>

Allow the model to train until the loss consistently drops below XXXXX. For my bird model, this took about XXXX steps, or XX hours of training (depending on how powerful your CPU and GPU are). (Please see [Step 6 my previous tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/README.md#6-run-the-training) for more information on training and an explanation of how to view the progress of the training job using TensorBoard.) 

Once training is complete (i.e. the loss has consistently dropped below XXXX), press Ctrl+C to stop training. The latest checkpoint will be saved in the \object_detection\training folder, and we will use that checkpoint to export the frozen TensorFlow Lite graph. Take note of the checkpoint number of the model.ckpt file in the training folder (i.e. model.ckpt-XXXXX), as it will be used later.

Note: train.py is deprecated, but the model_main.py script that replaced it doesn't log training progress by default, and it requires pycocotools to be installed. Using model_main.py requires a few extra setup steps, and I want to keep this guide as simple as possible. Since there are no major differences between train.py and model_main.py that will affect training ([see TensorFlow Issue #6100](https://github.com/tensorflow/models/issues/6100), I use train.py for this guide.

#### Step 1d. Export frozen inference graph for TensorFlow Lite
Now that training has finished, the model can be exported for conversion to TensorFlow Lite using the export_tflite_ssd_graph.py script. First, create a folder in \object_detection called “TFLite_model”. Next, let’s set up some environment variables so the commands are easier to type out. Issue the following commands in Anaconda Prompt. (Note, the XXXX in the second command should be replaced with the highest-numbered model.ckpt file in the \object_detection\training folder.) 

```
set CONFIG_FILE=C:\\tensorflow1\models\research\object_detection\training\ ssd_mobilenet_v2_quantized_300x300_coco.config
set CHECKPOINT_PATH= C:\\tensorflow1\models\research\object_detection\training\ model.ckpt-XXXX
set OUTPUT_DIR=C:\\tensorflow1\models\research\object_detection\TFLite_model
```

Now that those are set up, issue this command to export the model for TensorFlow Lite:

```
python export_tflite_ssd_graph.py --pipeline_config_path=%CONFIG_FILE% --trained_checkpoint_prefix=%CHECKPOINT_PATH% --output_directory=%OUTPUT_DIR% --add_postprocessing_op=true
```

After the command has executed, there should be two new files in the \object_detection\TFLite_model folder: tflite_graph.pb and tflite_graph.pbtxt. 

That’s it! The new inference graph has been trained and exported. This inference graph's architecture and operations are compatible with TensorFlow Lite's framework. However, the graph still needs to be converted to an actual TensorFlow Lite model. We'll do that in Step 2!

### Step 2. Build TensorFlow From Source
To convert the frozen graph we just exported into a model that can be used by TensorFlow Lite, we have to run it through the TensorFlow Lite Optimizing Converter (TOCO). Unfortunately, to use TOCO, we have to build TensorFlow from source on our computer. To do this, we’ll create a separate Anaconda virtual environment for building TensorFlow. 

This part of the tutorial breaks down step-by-step how to build TensorFlow from source on your Windows PC. It follows the [Build TensorFlow From Source on Windows](https://www.tensorflow.org/install/source_windows) instructions given on the official TensorFlow website, with some slight modifications. 

This guide will show how to build either the CPU-only version of TensorFlow or the GPU-enabled version of TensorFlow. **If you are only building TensorFlow to convert a TensorFlow Lite object detection model, I recommend building the CPU-only version.** It takes very little computational effort to export the model, so your CPU can do it just fine without help from your GPU. The guide shows how to build TensorFlow v1.13. If you would like to build a newer or older version, check the [build configuration list](https://www.tensorflow.org/install/source_windows#tested_build_configurations) and make sure you use the correct package versions.

If you’d like to build the GPU-enabled version for some other reason, then you need to have the appropriate version of CUDA and cuDNN installed. [The TensorFlow installation guide](https://www.tensorflow.org/install/gpu#windows_setup) explains how to install CUDA and cuDNN. Check the [build configuration list](https://www.tensorflow.org/install/source_windows#tested_build_configurations) to see which versions of CUDA and cuDNN are compatible with which versions of TensorFlow.

**If you get any errors during this process, please look at the FAQ section at the bottom of this guide! It gives solutions to common errors that occur. (Link to be added)**

#### Step 2a. Install MSYS2
MSYS2 has binary tools needed for building TensorFlow. It also automatically converts Windows-style directory paths to Linux-style paths when using Bazel. The Bazel build won’t work without MSYS2 installed! 

First, install MSYS2 by following the instructions on the [MSYS2 website](https://www.msys2.org/). Download the msys2-x86_64 executable file and run it. Use the default options for installation. After installing, open MSYS2 and issue:

```
pacman -Syu
```

<Picture of MSYS2 shell to be added here>

After it's completed, close the window, re-open it, and then issue the following two commands:

```
pacman -Su
pacman -S patch unzip
```

This updates MSYS2’s package manager and downloads the patch and unzip packages. The official TensorFlow build guide also installs MSYS2’s git package, but I have had encountered errors using MSYS2's version of git. Instead, we’ll use the git package provided by Anaconda. Close the MSYS2 window. We'll add the MSYS2 binary to the PATH environment variable in Step 2c.

#### Step 2b. Install Visual C++ Build Tools 2015
Install Microsoft Build Tools 2015 and Microsoft Visual C++ 2015 Redistributable by visiting the [Visual Studio older downloads](https://visualstudio.microsoft.com/vs/older-downloads/) page. Click the “Redistributables and Build Tools” dropdown at the bottom of the list.  Download and install the following two packages:

* **Microsoft Build Tools 2015 Update 3** - Use the default installation options in the install wizard. Once you begin installing, it goes through a fairly large download, so it will take a while if you have a slow internet connection. It may give you some warnings saying build tools or redistributables have already been installed. If so, that's fine; just click through them.
* **Microsoft Visual C++ 2015 Redistributable Update 3** – This may give you an error saying the redistributable has already been installed. If so, that’s fine.

Restart your PC after installation has finished.

#### Step 2c. Update Anaconda and create tensorflow-build environment
Now that the Visual Studio tools are installed and your PC is freshly restarted, open a new Anaconda Prompt window. First, update Anaconda to make sure its package list is up to date. In the Anaconda Prompt window, issue these two commands: 

```
conda update -n base -c defaults conda
conda update --all
```

The update process may take up to an hour, depending on how it's been since you installed or updated Anaconda. Next, create a new Anaconda virtual environment called “tensorflow-build”. We’ll work in this environment for the rest of the build process. Create and activate the environment by issuing:

```
conda create -n tensorflow-build pip python=3.6
conda activate tensorflow-build
```

After the environment is activated, you should see (tensorflow-build) before the active path in the command window. 

<Maybe I should add a picture of the Anaconda window here?>

Update pip by issuing:

```
python -m pip install --upgrade pip
```

We'll use Anaconda's git package to download the TensorFlow repository, so install git using:

```
conda install -c anaconda git
```

Next, add the MSYS2 binaries to this environment's PATH variable by issuing:

```
set PATH=%PATH%;C:\msys64\usr\bin
```

(If MSYS2 is installed in a different location than C:\msys64, use that location instead.) You’ll have to re-issue this command if you ever close and re-open the Anaconda Prompt window. 

