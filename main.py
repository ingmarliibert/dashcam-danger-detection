import json
import pathlib
import os

import tensorflow as tf
import numpy as np

# import Object Detection module
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

from model.utils import load_model, show_inference, run_inference_for_image_path

utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
detection_model = load_model(model_name)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

for image_path in TEST_IMAGE_PATHS:
    filename = os.path.basename(image_path)

    result = run_inference_for_image_path(detection_model, image_path)
    with open(f'{filename}.json', 'w') as f:
        f.write(json.dumps(result, cls=NumpyEncoder))
