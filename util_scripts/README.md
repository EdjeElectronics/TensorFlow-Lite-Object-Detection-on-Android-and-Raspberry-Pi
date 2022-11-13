## Utility Scripts for TensorFlow Lite Object Detection

### Calculate model mAP - (calculate_map_cartucho.py)

### Create CSV annotation file - (create_csv.py)
This script creates a single CSV data file from a set of Pascal VOC annotation files. It is used in the [TFLite Training Colab](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb) as part of the data preparation process.

Original credit for the script goes to [datitran](https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py).

### Create TFRecord file - (create_tfrecord.py)
This script creates TFRecord files from a CSV annotation data file and a folder of images. TFRecords are the [data format required](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md) by the TensorFlow Object Detection API for training. The script is used in the [TFLite Training Colab](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb) as part of the data preparation process.

Original credit for the script goes to [datitran](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py).

### Split images into train, test, and validation sets - (train_val_test.py)
