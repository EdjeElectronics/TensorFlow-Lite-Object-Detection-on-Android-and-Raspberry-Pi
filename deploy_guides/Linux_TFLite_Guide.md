# How to Run TensorFlow Lite Models on Linux
This guide shows how to set up a TensorFlow Lite Runtime environment on a Linux system. We'll use [Anaconda](https://www.anaconda.com/) to create a Python environment to install the TFLite Runtime in. It's easy!

## Step 1. Download and Install Anaconda
First, install [Anaconda](https://www.anaconda.com/), which is a Python environment manager that greatly simplifies Python package management and deployment. Anaconda allows you to create Python virtual environments without interfering with existing system-wide installations of Python. Go to the [Anaconda Downloads page](https://www.anaconda.com/products/distribution) and download the latest version for Linux.

Once downloaded, open a terminal and navigate to the directory where the installer is located, then run:

```
bash Anaconda3-*.sh
```

Follow the on-screen instructions to complete the installation. Restart your terminal or run:

```
source ~/.bashrc
```

## Step 2. Set Up Virtual Environment and Directory
Open a terminal and create a new directory for TensorFlow Lite:

```
mkdir ~/tflite1
cd ~/tflite1
```

Next, create a Python 3.9 virtual environment by issuing:

```
conda create --name tflite1-env python=3.9
```

Enter "y" when prompted to proceed. Activate the environment and install the required packages by running the commands below. We'll install TensorFlow, OpenCV, and a downgraded version of protobuf. TensorFlow is a pretty big download (about 450MB), so it will take a while.

```
conda activate tflite1-env
pip install tensorflow opencv-python protobuf==3.20.*
```

Download the detection scripts from the repository:

```
curl -O https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_image.py
curl -O https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_video.py
curl -O https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_webcam.py
curl -O https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_stream.py
```

## Step 3. Move TFLite Model into Directory
Next, download or move the custom TFLite model that was trained and downloaded from the Colab notebook into the `~/tflite1` directory. If you downloaded it from Colab, it should be in a file called `custom_model_lite.zip`. If you haven't trained a model yet and just want to test one, download a sample model by clicking this [Dropbox link](https://www.dropbox.com/scl/fi/4fk8ls8s03c94g6sb3ngo/custom_model_lite.zip?rlkey=zqda21sowk0hrw6i3f2dgbsyy&dl=0). Move the file to `~/tflite1` and unzip it:

```
tar -xf custom_model_lite.zip
```

At this point, you should have a folder at `~/tflite1/custom_model_lite` which contains at least a `detect.tflite` and `labelmap.txt` file.

## Step 4. Run TensorFlow Lite Model!
Now that everything is set up, running the TFLite model is easy. Just call one of the detection scripts and point it at your model folder with the `--modeldir` option. For example, to run your `custom_model_lite` model on a webcam, issue:

```
python TFLite_detection_webcam.py --modeldir=custom_model_lite
```

A window will appear showing detection results drawn on the live webcam feed, make sure to accept the use of webcam.. If you're running this on a headless system, ensure you have an appropriate display setup (like X forwarding or a VNC connection). You can read more about this here: [Stack Exchange Forum](https://unix.stackexchange.com/questions/12755/how-to-forward-x-over-ssh-to-run-graphics-applications-remotely), [archlinux](https://wiki.archlinux.org/title/X11vnc).

For more details on how to use the detection scripts for images, videos, or streams, please see [Step 3 in the main README page](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#step-3-run-tensorflow-lite-models).

Have fun using TensorFlow Lite! Stay tuned for more examples on how to build cool applications around your model.