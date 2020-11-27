import pathlib

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from dataclasses import dataclass
from tensorflow import keras
from tensorflow.keras import layers
import json
from PIL import Image

# from model.train.consts import img_height, img_width

# img_height = 100
# img_width = 100

img_height = 50
img_width = 50


@dataclass
class InceptionResult:
    class_name: str
    class_id: int
    score: float

class TrafficSign:
    def __init__(self, model, class_names, class_indexes):
        self.model = model
        self.class_names = class_names
        self.class_indexes = class_indexes

    def inception(self, frame: np.array) -> InceptionResult:
        # img = frame

        img = Image.fromarray(frame).resize(
            (img_width, img_height), Image.BILINEAR)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        # predictions = self.model(img_array, training=False)
        predictions = self.model.predict(img_array)


        prediction_index = np.argmax(predictions)
        prediction_class_index = int(self.class_indexes[prediction_index])
        prediction_class_name = self.class_names[prediction_class_index]

        score = np.max(predictions)
        print(
            "This image most likely belongs to {} prediction_class_index {} with a {:.2f} percent confidence."
                .format(prediction_class_name, prediction_class_index,  100 * score)
        )

        return InceptionResult(prediction_class_name, prediction_class_index, score)


def traffic_sign_factory():
    sign = pd.read_csv('signnames.csv')
    class_names = sign['SignName']

    model = tf.keras.models.load_model('saved_model/mobilenetv2-test2')

    with open('saved_model/mobilenetv2-test2/classes.json', 'r') as f:
        class_indexes = json.load(f)

    print(f'class_indexes = {class_indexes}')

    return TrafficSign(model, class_names, class_indexes)

# traffic_sign_model = traffic_sign_factory()

# img = keras.preprocessing.image.load_img('./data/traffic-sign/train/2/00002_00000_00002.png', target_size=(img_height, img_width), interpolation='bilinear')
# img1 = keras.preprocessing.image.load_img('./data/traffic-sign/test-clean/2/00067.png', target_size=(img_height, img_width), interpolation='bilinear')
# img2 = keras.preprocessing.image.load_img('./data/traffic-sign/test-clean/3/00036.png', target_size=(img_height, img_width), interpolation='bilinear')
# img3 = keras.preprocessing.image.load_img('./data/traffic-sign/test-clean/15/00210.png', target_size=(img_height, img_width), interpolation='bilinear')
# img4 = keras.preprocessing.image.load_img('./data/traffic-sign/test-clean/17/00035.png', target_size=(img_height, img_width), interpolation='bilinear')
# img5 = keras.preprocessing.image.load_img('./data/traffic-sign/test-clean/34/00386.png', target_size=(img_height, img_width), interpolation='bilinear')
#
#
# def process(img):
#     traffic_sign_model.inception(img)
#
# process(img1)
# process(img2)
# process(img3)
# process(img4)
# process(img5)
