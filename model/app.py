"""
Model instance to our specific application.
"""
import tensorflow as tf
from object_detection.utils import label_map_util

from model.object_detect import object_detection, object_detection_visualize
from model.util import load_model
from functools import partial

# Patch the location of gfile
tf.gfile = tf.io.gfile


def tf_object_detection_factory():
    model_name = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
    detection_model = load_model(model_name)

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    return detection_model, category_index

detection_model, category_index = tf_object_detection_factory()

def object_detection_factory():
    return partial(object_detection, detection_model)

def object_detection_visualize_factory():
    return partial(object_detection_visualize, category_index)

