from typing import Dict, List

import cv2

from model.object_detect import DetectedObject
from utils.rectangle import Rectangle


def get_distance_and_object(detections: Dict[str, List[DetectedObject]], lines) -> (float, DetectedObject):
    relevant_labels = ['car', 'motorcycle', 'bus', 'truck']

    minimum = [float('inf'), None]
    for label in detections:
        if label in relevant_labels:
            objects = detections[label]
            for object in objects:
                if not is_in_lane(object, lines):
                    continue
                distance = get_distance(object.detection_rectangle)

                if distance < minimum[0]:
                    minimum[0] = distance
                    minimum[1] = object

    return minimum


def is_in_lane(object: DetectedObject, lines):
    if lines is None:
        return True
    dx = 5
    for line in lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        min_x_in_bounds = x1 - dx < object.detection_rectangle.min_x < x2 + dx
        max_x_in_bounds = x1 - dx < object.detection_rectangle.max_x < x2 + dx

    return min_x_in_bounds or max_x_in_bounds


def get_distance(car_bounding_box: Rectangle):
    focal_length = 1
    avg_car_width = 2
    pixel_width = car_bounding_box.max_x - car_bounding_box.min_x
    distance = (avg_car_width * focal_length) / pixel_width
    return distance


if __name__ == '__main__':
    import os
    import pathlib

    import tensorflow as tf
    from PIL import Image
    import numpy as np
    from object_detection.utils import ops as utils_ops

    from model.app import object_detection_factory, object_detection_visualize_factory
    from model.line_detection import find_lines

    utils_ops.tf = tf.compat.v1

    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    # PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('../')
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("detect_distance_test.jpg")))

    object_detection = object_detection_factory()
    object_detection_visualize = object_detection_visualize_factory()

    for image_path in TEST_IMAGE_PATHS:
        filename = os.path.basename(image_path)
        image_np = np.array(Image.open(image_path))

        tf_results, results = object_detection(image_np)
        lines = find_lines(image_np)
        image_np_visualized = object_detection_visualize(tf_results, image_np)
        i = Image.fromarray(image_np_visualized)
        i.show()
        print(get_distance_and_object(results, lines))
