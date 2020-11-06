import os
import pathlib

import tensorflow as tf
from PIL import Image
import numpy as np
from object_detection.utils import ops as utils_ops

from model.app import object_detection_factory, object_detection_visualize_factory
from collisions import get_collisions

utils_ops.tf = tf.compat.v1

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("car-crash.jpg")))

object_detection = object_detection_factory()
object_detection_visualize = object_detection_visualize_factory()

for image_path in TEST_IMAGE_PATHS:
    filename = os.path.basename(image_path)
    image_np = np.array(Image.open(image_path))

    tf_results, results = object_detection(image_np)

    collisions = get_collisions(results)
    print(f'collisions = {collisions}')

    image_np_visualized = object_detection_visualize(tf_results, image_np)
    i = Image.fromarray(image_np_visualized)
    i.save('./result.jpg')
