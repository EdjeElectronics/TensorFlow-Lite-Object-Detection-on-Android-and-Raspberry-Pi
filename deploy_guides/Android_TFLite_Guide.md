# How to Run TensorFlow Lite Models on Android
This guide shows how to load a TensorFlow lite model on an Android device to detect objects using the rear camera.

<p align="center">
   <img src="../doc/coin_detection.jpg">
</p>

## Requirements
An Android device with an API level of 24+. This covers 97% of all active Android devices.
A phone is highly recommended. Tablets don't have the same performance or camera quality.

## Step 1. Train your TensorFlow Lite model
[Google Colab Notebook](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb)

Make sure to follow Step 8.2.4. for deploying on Android. It's required to initialize your model on the device.

## Step 2. Download and Install Android Studio
[Download Android Studio](https://developer.android.com/studio)

## Step 3. Setup your device for use with Android Studio
[Run apps on a hardware device](https://developer.android.com/studio/run/device)

<p align="center">
   <img src="../doc/Device_Setup_Step1.jpg" width="400">
   <img src="../doc/Device_Setup_Step2.jpg" width="400">
   <img src="../doc/Device_Setup_Step3.jpg" width="400">
   <img src="../doc/Device_Setup_Step4.jpg" width="400">
   <img src="../doc/Device_Setup_Step5.jpg" width="400">
</p>


## Step 4. Clone repository

