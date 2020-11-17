from typing import Dict, List

import cv2
import numpy as np

from model.object_detection import DetectedObject
from utils.rectangle import Rectangle


def get_distance_and_object(detections: Dict[str, List[DetectedObject]], lines) -> (float, DetectedObject):
    relevant_labels = ['car', 'motorcycle', 'bus', 'truck', ]

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


def calculate_position(bbox):
    transform_matrix = None
    warped_size = None
    pix_per_meter = None
    pos = np.array((bbox[0] / 2 + bbox[2] / 2, bbox[3])).reshape(1, 1, -1)
    dst = cv2.perspectiveTransform(pos, transform_matrix).reshape(-1, 1)
    return np.array((warped_size[1] - dst[1]) / pix_per_meter[1])


def is_in_lane(object: DetectedObject, lines):
    dx = 5
    x1, y1 = lines[0]
    x2, y2 = lines[1]
    min_x_in_bounds = x1 - dx < object.detection_rectangle.min_x < x2 + dx
    max_x_in_bounds = x1 - dx < object.detection_rectangle.max_x < x2 + dx

    return min_x_in_bounds or max_x_in_bounds


def get_distance(car_bounding_box: Rectangle):
    focal_length = 250
    avg_car_width = 2
    pixel_width = car_bounding_box.max_x - car_bounding_box.min_x
    width = (avg_car_width * focal_length) / pixel_width
    return width
