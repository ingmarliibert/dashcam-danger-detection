import warnings

import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops

from model.app import object_detection_factory, object_detection_visualize_factory
from model.distance_detection import get_distance_and_object

utils_ops.tf = tf.compat.v1

object_detection = object_detection_factory()
object_detection_visualize = object_detection_visualize_factory()

warnings.filterwarnings("ignore", category=np.RankWarning)

video = cv2.VideoCapture("detect_distance_demo_video.mp4")
count = 0
while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    tf_results, results = object_detection(frame)
    distance, closest_object = get_distance_and_object(results, None)

    if distance < 6:
        count = min(count + 1, 50)
    else:
        count = max(count - 1, 0)
    if count >= 20:
        cv2.putText(frame, "Too close! Distance too small.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)

    cv2.imshow("", frame)
    cv2.waitKey(1)
