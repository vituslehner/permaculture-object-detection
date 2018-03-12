# Object detection in the context of permaculture

Permaculture is a concept of cyclic, symbotic and permanent land use for growing vegetables and fruits.
In contrast to conventional agriculture, plants are kept permanently and symbiotic effects between neighboured plants
(e.g. fruit trees, berry shrubs, etc.)
are used to overcome problems of pest infestation and weeds. While this enables the efficient production
of organic food, it is hardly possible to manage such land using conventional land machines. Permaculture requires
a lot of manual labor.

To further support the labor - or in the long term replace it - we propose image classification and object detection
usage to visually detect fruits and plants using a conventional camera. This might be the base for future robotics
usage in permacultural context.

## Rationales of this repository

![Robot Perla][robot]

![Demo Setup][demo]

This repository should provide some guidance on how to get grip on the topic of deep learning 
and object detection using Tensorflow. It provides links to helpful online resources, sample codes and further ideas
on how to use Tensorflow Object Detection API. This is not going to be a comprehensive step-by-step tutorial
but more a collection of what we think is useful information.

The major objective of permaculture is to inhabit and use earth in a sustainable fashion. Reflecting on this,
we feel that we should publish the understanding we gained during the building of our prototype. This way
we can contribute to the interested community and help build a sustaining knowledge base.

This repository is the result of implementing a prototypical **Tomato detection system** that runs on a Raspberry Pi 3 and
AWS cloud. We attached the setup to a LEGO MINDSTORMS kit and lovingly named our robot Perla.
 
## About us

We are a small team of four students that created this project during a project seminar. The seminar context was
about technological innovation in the food producing sector from the perspective ot sustainable development. We are
an interdisciplinary team of engineers, computer scientists, environmental scientists and business administrators.

## Prequisites

We were using the following hardware and computation resources:

- Raspberry Pi 3
- Raspberry Pi Camera v2.1 (no IR)
- Amazon Web Services (AWS) EC2 Instance of type p2.xlarge, consisting of
    - one NVIDIA K80 GPU (12 GiB)
    - four vCPUs of Intel Xeon E5-2686
    - 61 GiB of memory
    - based in Frankfurt a.M., Germany
- some laptop (ideally running macOS or Linux)

Additionally, we were using a LEGO MINDSTORMS v3 kit for building a wheeled prototype. We attached the Raspy setup
to the MINDSTORMS robot, which made a very nice setup for demonstration purposes. As the assembly of MINDSTORMS
is very well documented by LEGO and tons of other internet resources, we will focus here on describing the object detection
backbone with the Raspy.

## Table of Contents

The content of this README can roughly be divided into two parts: one for setting up the cloud and training of a neural net for 
object detection, and one for wiring the Raspy to the cloud and back.

- [Object detection in the context of permaculture](#object-detection-in-the-context-of-permaculture)
  - [Rationales of this repository](#rationales-of-this-repository)
  - [About us](#about-us)
  - [Prequisites](#prequisites)
  - [Table of Contents](#table-of-contents)
- [Part I: Preparation of AWS cloud, selection of training data and training using Tensorflow](#part-i-preparation-of-aws-cloud-selection-of-training-data-and-training-using-tensorflow)
  - [Setup AWS EC2 computing instance](#setup-aws-ec2-computing-instance)
  - [Setting up the Tensorflow environment](#setting-up-the-tensorflow-environment)
  - [Training an object detection model](#training-an-object-detection-model)
    - [Prepare and preprocess training images](#prepare-and-preprocess-training-images)
    - [Convert images to Tensorflow-readable data format](#convert-images-to-tensorflow-readable-data-format)
    - [Train the model](#train-the-model)
      - [Notes on detection model](#notes-on-detection-model)
    - [Export the trained model](#export-the-trained-model)
- [Part II: Use the trained object detection model for inference with a Raspberry Pi setup](#part-ii-use-the-trained-object-detection-model-for-inference-with-a-raspberry-pi-setup)
  - [Script for using the inference model for object detection](#script-for-using-the-inference-model-for-object-detection)
  - [Script for publishing the inferred images](#script-for-publishing-the-inferred-images)

# Part I: Preparation of AWS cloud, selection of training data and training using Tensorflow

This part is about setting up the whole deep learning system in the cloud and train it using our own data.

## Setup AWS EC2 computing instance

We chose to use cloud computing resources for our use case as they may provide GPU resources which can be used
to accelerate the deep learning computation, especially during the training part. As we had no
GPU-equipped PC and did not want to invest in expensive hardware, that was our way to go.

We were experimenting with Google Cloud Platform and their ML engine which integrates tensorflow right away,
but found that this was not flexible enough for us and many online literature is based on Python scripting,
which seemed invonventional with ML Engine. That is why we decided for AWS instead, which provides
integration of several deep learning frameworks. You can register for AWS [here](https://aws.amazon.com/).

We selected one of the smaller GPU-equipped instance types (p2.xlarge, see Prequisites) which costed about
0,90 USD per hour, or even less using spot-instances. The instance seemed suited for our application.

As a base operating system we chose an AWS-provided virtual machine image (AMI) for machine learning purposes,
namely: Deep Learning AMI (Ubuntu) Version 4.0 (ami-3a7d1955). It is based on Ubuntu LTS 16.04 and ships
with a bunch of deep learning frameworks in (Conda-based) virtual environments, among them Tensorflow 1.5.
This took as the pain from installing anything like Python or Tensorflow ourselves.
Note: we were using AMI v4.0, but at the time of writing this, AMI v.5.0 is already released.

When logged in with an AWS account, you can select a region where you want to start your computing instance
in the top right corner of the window. Make sure the instance type you want to launch is available at that
place. Now you can navigation to the EC2 service and launch a new instance. Within the launch assistant, search for
the deep learning AMI and select the instance type you want. The rest of the settings can be left unchanged (we'll
edit the security group later on).

During launching the instance, you'll have to create a private key and download it. Using that key you can log in
to your instance using SSH.

```
# set correct key file permissions before first use:
chmod 0400 key-file.pem

ssh -i key-file.pem ubuntu@ec2-xx-xx-xx-xx.eu-central-1.compute.amazonaws.com
```

As you are likely going to have to transfer some files between your computer to the remote instance, you can
copy/paste and adapt the following SCP commands.

```
# upload from local computer to remote instance
scp -i key-file.pem path/to/file/or/directory ubuntu@ec2-xx-xx-xx-xx.eu-central-1.compute.amazonaws.com:~/path/to/remote/location

# download from remote instance to local computer
scp -i key-file.pem ubuntu@ec2-xx-xx-xx-xx.eu-central-1.compute.amazonaws.com:~/path/from/remote/location path/to/file/or/riectory 
```

## Setting up the Tensorflow environment

The used AWS deep learning AMI provides Tensorflow right away. After you log in to your machine using SSH,
you can activate the virtual Conda environment that contains Tensorflow 1.5 and Python 3.6 using the following command:

```
# Activation
source activate tensorflow_p36

# Exit/deactivation
source deactivate tensorflow_p36
```

While the Tensorflow framework is already installed and can be used within e.g. Python scripts, there are no
deep learning models on the machine yet. [Google provides a major Git repository](https://github.com/tensorflow/models.git)
with a stack of models, among them several object detection models. Clone the repository onto the instance. You'll
mostly work in the `models/research` subfolder.

```
git clone https://github.com/tensorflow/models.git
cd models/research
```

## Training an object detection model

There are already plenty of good guides on how to train Tensorflow object detection using own custom training data.
In our case we wanted to detect tomatos, which happens to be a more simple object type.

Special thanks should be expressed to the following articles and guides:
- Dat Tran: [Building a Real-Time Object Recognition App with Tensorflow and OpenCV](https://towardsdatascience.com/building-a-real-time-object-recognition-app-with-tensorflow-and-opencv-b7a2b4ebdc32)
- Dat Tran: [How to train your own Object Detector with TensorFlow’s Object Detector API](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)
- Sara Robinson [Build a Taylor Swift detector with the TensorFlow Object Detection API, ML Engine, and Swift](https://towardsdatascience.com/build-a-taylor-swift-detector-with-the-tensorflow-object-detection-api-ml-engine-and-swift-82707f5b4a56)
- Daniel Stang [Step by Step TensorFlow Object Detection API Tutorial — Part 1: Selecting a Model](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)
- [Tensoflow object detection model documentation](https://github.com/tensorflow/models/tree/master/research/object_detection)
    - [Jupyter notebook tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)

These resources were really helpful during implementation of our object detection. You will find most information
needed to train an own detection model there. In the following, we will just provide some specifics of where
our learning process differed from the ones in the tutorials.

To keep track of the whole process, these are the main steps of deep learning and object detection in special (inspired by Sara Robinson):
1. Prepare and preprocess training images
2. Convert images to Tensorflow-readable data format (TFRecord)
3. Train the model
4. Export the trained model

### 1. Prepare and preprocess training images

For our tomato detection use case we had to select a bunch of tomato pictures which we esseantially took
from Google images. We minded the image sizes as they should not be too small or too big (600-900px seems well).
While you usually try to find photos of the object in as many different surroundings and environments as possible
to prevent overfitting of the neural net, in our case we had quite a lot of photos with greeny background (as
that's where tomatos ... grow). This was okay for us as the main purpose of the detector is to work in such
environments anyway.

We are afraid we can not publish our dataset due to copyright issues. Our image collection is created from several
resources and we used the images for research purposes. We are not explicitely granted to publish any of that data.

We manually downloaded all the images and used a short script [inspired by Sara Robinson](https://github.com/sararob/tswift-detection/blob/master/resize.py) to resize the image files.
The script seemed a bit buggy so we fixed some things. We also added the capability to rename the files in a 
nice manner on the fly:

```python
# Copyright 2017 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', help='Directory of images to resize')
args = parser.parse_args()

image_dir = os.getcwd() + "/" + args.image_dir

i=1
for filename in os.listdir(image_dir):
    if(False == filename.startswith('.')):
        image = Image.open(image_dir + '/' + filename)
        width, height = image.size
        resize_amt = 600 / width
        new_height = int(round(height * resize_amt))
        image = image.resize((600, new_height))
        image.save(os.getcwd() + "/image_out/image_" + str(i).zfill(3) + ".jpg")
        i=i+1

```

For our first training run we only selected 40 pictures for training and 10 pictures for testing. That is very few,
but already showed not too bad results. Keep in mind that a tomato shrub holds more than one tomato, so we had
enough work with labeling hundreds of tomatos anyway. For labeling the images we used [LabelImg](https://github.com/tzutalin/labelImg).

![LabelImg Usage][labelimg]

### 2. Convert images to Tensorflow-readable data format

After exporting the labeled training data from LabelImg, we had to convert it to the TFRecord format.
We based our converter on the [conversion script of Sara Robinson](https://github.com/sararob/tswift-detection/blob/master/convert_to_tfrecord.py) (who based hers on the one of Dat Tran).
In contrast to Sara's Taylow Swift detector where an image can only contain *one* Taylor Swift,
our training images can for sure contain more than one tomato. You can find it below. For usage, see
Sara's article.

```python
# Copyright 2017 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import io
import xml.etree.ElementTree as ET
import tensorflow as tf

from object_detection.utils import dataset_util
from PIL import Image


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('images_dir', '', 'Path to directory of images')
flags.DEFINE_string('labels_dir', '', 'Path to directory of labels')
FLAGS = flags.FLAGS


def create_tf_example(example):

    image_path = os.getcwd() + '/' +  FLAGS.images_dir + example
    labels_path = os.getcwd() + '/' +  FLAGS.labels_dir + os.path.splitext(example)[0] + '.xml'

    # Read the image
    img = Image.open(image_path)
    width, height = img.size
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=img.format)

    height = height
    width = width
    encoded_image_data = img_bytes.getvalue()
    image_format = img.format.encode('utf-8')

    # Read the label XML
    tree = ET.parse(labels_path)
    root = tree.getroot()
    xmins = list()
    xmaxs = list()
    ymins = list()
    ymaxs = list()
    classes = list()
    classes_text = list()

    for object in root.findall('object'):
        coordinate = object.find('bndbox')
        xmins.append(float(coordinate.find('xmin').text) / width)
        xmaxs.append(float(coordinate.find('xmax').text) / width)
        ymins.append(float(coordinate.find('ymin').text) / height)
        ymaxs.append(float(coordinate.find('ymax').text) / height)
        classes_text.append('tomato'.encode('utf-8'))
        classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(encoded_image_data),
        'image/source_id': dataset_util.bytes_feature(encoded_image_data),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    for filename in os.listdir(FLAGS.images_dir):
        tf_example = create_tf_example(filename)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
```

That way we converted the training as well as the test data.

### 3. Train the model

For training the model we essentially followed the basic steps of most guides, like the ones of Dat Tran or Sara Robinson.
We are training in the cloud but control the whole process ourselves using shell. Google provides
some nice documenation on [how to train a model locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md). Our object detection pipeline config 
is based on Google's pets example and looks like that:

```
model { 
  faster_rcnn {
    num_classes: 1
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 800
      }
    }
    feature_extractor {
      type: 'faster_rcnn_resnet101'
      first_stage_features_stride: 16
    }
    first_stage_anchor_generator {
      grid_anchor_generator {
        scales: [0.25, 0.5, 1.0, 2.0]
        aspect_ratios: [0.5, 1.0, 2.0]
        height_stride: 16
        width_stride: 16
      }
    }
    first_stage_box_predictor_conv_hyperparams {
      op: CONV
      regularizer {
        l2_regularizer {
          weight: 0.0
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.01
        }
      }
    }
    first_stage_nms_score_threshold: 0.0
    first_stage_nms_iou_threshold: 0.7
    first_stage_max_proposals: 300
    first_stage_localization_loss_weight: 2.0
    first_stage_objectness_loss_weight: 1.0
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_box_predictor {
      mask_rcnn_box_predictor {
        use_dropout: false
        dropout_keep_probability: 1.0
        fc_hyperparams {
          op: FC
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
      }
    }
    second_stage_post_processing {
      batch_non_max_suppression {
        score_threshold: 0.0
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 300
      }
      score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
  }
}

train_config: {
  batch_size: 1
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        manual_step_learning_rate {
          initial_learning_rate: 0.0003
          schedule {
            step: 0
            learning_rate: .0003
          }
          schedule {
            step: 900000
            learning_rate: .00003
          }
          schedule {
            step: 1200000
            learning_rate: .000003
          }
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "/home/ubuntu/perla/local/data/resnet101/model.ckpt"
  from_detection_checkpoint: true
  # Note: The below line limits the training process to 200K steps, which we
  # empirically found to be sufficient enough to train the pets dataset. This
  # effectively bypasses the learning rate schedule (the learning rate will
  # never decay). Remove the below line to train indefinitely.
  num_steps: 200000
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
}

train_input_reader: {
  tf_record_input_reader {
    input_path: "/home/ubuntu/perla/local/data/train.record"
  }
  label_map_path: "/home/ubuntu/perla/local/data/perla_label_map.pbtxt"
}

eval_config: {
  num_examples: 2000
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "/home/ubuntu/perla/local/data/test.record"
  }
  label_map_path: "/home/ubuntu/perla/local/data/perla_label_map.pbtxt"
  shuffle: false
  num_readers: 1
}

```

Our `perla_label_map.pbtxt` file looks like this:
```
item {
  id: 1
  name: 'tomato'
}
```
Using the Tensorflow commands provided [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)
you can now start to train and evaluate the model. Have a look at Tensorboard as well.

#### Notes on detection model

When training new custom classes you are most likely going to train the last layer of an existing detection model.
The Single Shot MultiBox detection model using the MobileNet neural net ist popular in tutorials as its
comparibly lightweight (anc can be run on mobile devices) and fast but less accurate. Our first-try training data
was rather small, that's why we tried another more comprehensive but slower model first. The combinition of
Faster R-CNN with ResNet101 seemed to be a good compromise of accuracy and speed, and is also said to be good at
detection small objects. We went with Faster R-CNN ResNet101 trained on the COCO dataset with 90 pretrained object classes.

Note: As of our configuration, the model will only recognize tomatos after the training but not the other 90 classes.
That is because we did not include the COCO dataset for our further training and our pbtxt file does not contain the
other classes. The means the neural net forgets about the previous classes more or less. As of now, there seems to be
no easy way to extend a trained model with new classes.

In our case, the Loss rate shown in Tensorboard decreased quite fast. We guess this is due to the relatively simple
object and the overfitting dataset.

![TotalLoss for tomato training in Tensorboard][totalloss]

### 4. Export the trained model

We did nothing special here but used the default way to [export the trained model for inference](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md).

# Part II: Use the trained object detection model for inference with a Raspberry Pi setup

## Script for using the inference model for object detection

### Object detection server

As our trained model is quite huge and slow we decided to run inference in the cloud as well. That means that the Raspberry Pi
needs to upload its image data to the cloud instance for further detection. Therefor we adapted the script of
the [Jupyter noterbook tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb).
Instead of taking a fixed set of images as input, we made it listen for an image stream on a socket,
run the inference and provide the resulting images with labeled boxes (by saving it to disk first).

Our script looks like that:

```python
# coding: utf-8

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image 

import io
import socket
import struct
import base64
from io import BytesIO

from _thread import start_new_thread
import traceback

sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util


MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'


# Uncomment/comment to switch between own trained classes and COCO classes and inference graph
#PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
#NUM_CLASSES = 90

PATH_TO_CKPT = '/home/ubuntu/perla/models/research/exported_graphs/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'perla_label_map.pbtxt')
NUM_CLASSES = 1

# Optionally download model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


try:

  # Start a socket listening for connections on 0.0.0.0:8006 (0.0.0.0 means
  # all interfaces)
  server_socket = socket.socket()
  server_socket.bind(('0.0.0.0', 8006))
  server_socket.listen(0)
  
  # Accept a single connection and make a file-like object out of it
  connection = None
  curr = None
  
  def listen_to_images(app):
    global connection
    global curr
    run = True
    
    print('Start listening for images')
    connection = server_socket.accept()[0].makefile('rb')
    while run:
      try:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
          continue
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        print('==> Receiving image')
        #if(not (curr == None)):
        #  continue
        
        print('====> Preparing image')
        image_stream.seek(0)
        image = Image.open(image_stream)
        curr = image
      except OSError as err:
        print('TRUNCATED FILE (in preprocessing)...:', err)
        traceback.print_exc()
      except ValueError as err:
        print('STREAM CLOSED...:', err)
        print('...waiting for new connection')
        run = False
        #connection = server_socket.accept()[0].makefile('rb')
      except struct.error as err:
        print('STREAM CLOSED (struct)...:', err)
        print('...waiting for new connection')
        connection = server_socket.accept()[0].makefile('rb')
      except KeyboardInterrupt:
        print('Stopped by keyboard')
        raise
      except:
        err = sys.exc_info()[0]
        print('SHIT HAPPENED (in preprocessing)…:', err)
        traceback.print_exc()
        run = False
  
  def process_images(a):
    global curr
    
    print('Start processing job')
    i = 1
    while True:
      if(curr == None):
        continue
    
      print('> Processing image')
      image = curr
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      try:
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=3)
        
        # Save inferred image
        infimage = Image.fromarray(image_np)
        infimage.save('inferred_images/image.jpg')
      except OSError as err:
        print('TRUNCATED FILE...: ', err)
        traceback.print_exc()
      except:
        err = sys.exc_info()[0]
        print('SHIT HAPPENED…:', err)
        traceback.print_exc()
      curr = None
      print('> Image processed')
      i = i + 1 

  start_new_thread(listen_to_images, (99,))
  start_new_thread(process_images, (99,))
  
finally:
  connection.close()
  server_socket.close()
```

**Do not forget to open the incoming network port (here 8006) of the security group of your instance in AWS.**

The script essentially starts two seperate threads: one for listening for new images coming from the Raspberry Pi
and one for running the object detection on the latest image. the result of the inference gets saved to a file.
Put this script within `models/research/object_detection` and run it from there. Run it before you run the client.

### Image capturing client

The client-side (Raspy) also runs a Python script. It is based on the Raspberry Pi Python Camera documention and is basically
a Python script that captures images from the cam and sends them to the cloud instance with a very simplistic protocol.
You can find the base script [here](https://picamera.readthedocs.io/en/release-1.10/recipes1.html#capturing-to-a-network-stream).

Our script looks like that:

```python
import io
import socket
import struct
import time
import picamera
import sys
import traceback

target = ('ec2-xx-xx-xx-xx.eu-central-1.compute.amazonaws.com', 8006)

camera = picamera.PiCamera()
camera.resolution = (800, 600)
# Start a preview and let the camera warm up for 2 seconds
camera.start_preview()
time.sleep(2)


while True:
  
  try:
  
      # Connect a client socket to my_server:8000 (change my_server to the
      # hostname of your server)
      client_socket = socket.socket()
      client_socket.connect(target)
      
      # Make a file-like object out of the connection
      connection = client_socket.makefile('wb')
 
      # Note the start time and construct a stream to hold image data
      # temporarily (we could write it directly to connection but in this
      # case we want to find out the size of each capture first to keep
      # our protocol simple)
      start = time.time()
      stream = io.BytesIO()
      for foo in camera.capture_continuous(stream, 'jpeg'):
          try:
              # Write the length of the capture to the stream and flush to
              # ensure it actually gets sent
              connection.write(struct.pack('<L', stream.tell()))
              connection.flush()
              # Rewind the stream and send the image data over the wire
              stream.seek(0)
              connection.write(stream.read())
              
              # Reset the stream for the next capture
              stream.seek(0)
              stream.truncate()
          except:
              err = sys.exc_info()[0]
              print('what happened?', err)
              traceback.print_exc()
              raise
              
      # Write a length of zero to the stream to signal we're done
  except:
      try:
        connection.close()
        client_socket.close()
      except:
        print('ignore2')
      print('retry') 
      print('Waiting for 5 sec too')
      time.sleep(5)
      try:
        client_socket = socket.socket()
        client_socket.connect(target)
        connection = client_socket.makefile('wb')
      except:
        err = sys.exc_info()[0]
        print('reconnect failed', err) 
  finally:
      try:
        connection.close()
        client_socket.close()
      except:
        print('ignore')
```

It is different from the orginial in that it is more fault-tolerant. It trys to reconnect to the server in
case the connection gets lost or the camera captures a faulty image.

Run that script after you started the server.

## Web app for showing the inferred images

So far the inferred images get saved on the server, and by using SCP we can download and have a look at them. For our demonstration
we want to show the results as a live stream on our laptop. Therefor we developed a tiny Node application using JavaScript
that delivers a simplistic web page and shows the latest inferred image together with a time stamp. To always show the latest
inference result, we used WebSockets to push any new images to the web browser.

This mini application only consists of `package.json`, `index.html` and `server.js` which you can find below (in the given order).

```json
{
  "name": "project-perla-node-server",
  "version": "1.0.0",
  "description": "",
  "main": "server.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1",
    "start": "node server.js"
  },
  "author": "Project Perla",
  "license": "ISC",
  "dependencies": {
    "express": "^4.16.2",
    "socket.io": "^2.0.4"
  }
}
```

```html
<!DOCTYPE html>
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/2.0.4/socket.io.js"></script>
<script src="http://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
 
<title>Project Perla</title>

<script type="text/javascript">
$(function () {
    var socket = io();
    socket.on('image', function(obj){
        document.getElementById("image").src = 'data:image/jpeg;base64,' + obj.buffer;
        document.getElementById("date").innerHTML = obj.time;
    });
  });
</script>
<style>
html, body{width:100%;height:100%;margin:0;padding:0;}
p {
position:absolute;
}
img{
display: block;
margin: 0 auto;
max-width:100%;
height:100%;}
</style>
 
</head>
<body>
<p id="date"></p>
<img id="image"/>
</body>
</html>​
```

```js

var app = require('express')();
var http = require('http').Server(app);
var io = require('socket.io')(http);

var fs = require('fs'); // required for file serving

http.listen(8080, function(){
  console.log('listening on *:8080');
});

// location to index.html
app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

io.on('connection', function(socket){
  console.log('a user connected');
});

// listen to the inferred image file
var file = __dirname + '/../models/research/object_detection/inferred_images/image.jpg';
fs.watchFile(file,{interval:50}, (curr, prev) => {
  fs.readFile(file, function(err, buf){
    io.sockets.emit('image', { image: true, buffer: buf.toString('base64'), time: new Date().toUTCString() });
  });
});
```

Put these files into the same directory on your instance. Run `npm install` to install the required packages
and then run `node server.js` to start the web server.

**Do not forget to open the incoming network port (here 8080) of the security group of your instance in AWS.**

The node server listens for changed with the inferred image file and pushes its contents base64-encoded to the clients
every time the file changes. It checks for file changes every 50ms. This solution is far from perfect but worked
as expected.

We tried to write such a serverside script just within the Python detection script but we could not get the
threads of the detection and the threads and/or loops of the web server (AIOHTTP) and the WebSocket in sync.
We are not very confident with Python's deeper internals that is why we sticked with the simple node solution so far.
If you have any suggestions on how one could solve this, we would really like to hear them.

You can now go to http://ec2-xx-xx-xx-xx.eu-central-1.compute.amazonaws.com:8080 and be able to see the inferred images
in live (with a delay of some seconds of course).

#### Notes on startup

To successfully run the whole setup, you should do the following steps in the given order:

1. Start the object-detection Python script that also listens for images coming from the Raspberry Pi.
2. Start the node application.
3. Open you browser and enter the public DNS name of your AWS instane followed by the web server port (something like http://ec2-xx-xx-xx-xx.eu-central-1.compute.amazonaws.com:8080).
4. Run the capturing and sending Python script on the Raspberry Pi.

[robot]: https://github.com/vituslehner/permaculture-object-detection/raw/master/assets/robot-picture.png "Robot Perla"
[demo]: https://github.com/vituslehner/permaculture-object-detection/raw/master/assets/demo-setup.png "Robot Perla Demo Setup"
[labelimg]: https://github.com/vituslehner/permaculture-object-detection/raw/master/assets/labelimg-sample.png "Labeling tomatos using labelImg"
[totalloss]: https://github.com/vituslehner/permaculture-object-detection/raw/master/assets/tensorboard-totalloss.png "Total Loss graph in Tensorboard"
