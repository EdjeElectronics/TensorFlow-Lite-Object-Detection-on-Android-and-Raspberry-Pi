# How to Run TensorFlow Lite Models on Windows
This guide shows how to set up a TensorFlow Lite Runtime environment on a Windows PC. We'll use [Anaconda](https://www.anaconda.com/) to create a Python environment to install the TFLite Runtime in. It's easy!

**Important Note: TensorFlow Lite models run slow on PCs, because the TFLite Runtime is optimized lower-power ARM processors. Running your model on Windows is a good way to test its accuracy, but it will run faster on edge devices like Android phones or the Raspberry Pi.**

## Step 1. Download and Install Anaconda
First, install [Anaconda](https://www.anaconda.com/), which is a Python environment manager that greatly simplifies Python package management and deployment. Anaconda allows you to create Python virtual environments on your PC without interfering with existing installations of Python. Go to the [Anaconda Downloads page](https://www.anaconda.com/products/distribution) and click the Download button.

When the download finishes, open the downloaded .exe file and step through the installation wizard. Use the default install options.

## Step 2. Set Up Virtual Environment and Directory Structure
Go to the Start Menu, search for "Anaconda Command Prompt", and click it to open up a command terminal. We'll create a folder called `tflite1` directly in the C: drive. (You can use any other folder location you like, just make sure to modify the commands below to use the correct file paths.) Create the folder and move into it by issuing the following commands in the terminal:

```
mkdir C:\tflite1
cd C:\tflite1
```

Next, create a Python 3.9 virtual environment by issuing:

```
conda create --name tflite1-env python=3.9
```

Enter "y" when it asks if you want to proceed. Activate the environment and install TensorFlow v2.8.0 by issuing:

```
conda activate tflite1-env
pip install tensorflow==2.8.0
```

Download the detection scripts from this repository by issuing:

```
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_image.py --output TFLite_detection_image.py
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_video.py --output TFLite_detection_video.py
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_webcam.py --output TFLite_detection_webcam.py
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_stream.py --output TFLite_detection_stream.py
```


