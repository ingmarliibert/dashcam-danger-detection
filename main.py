import cv2
import tensorflow as tf
from object_detection.utils import ops as utils_ops

from collisions import get_collisions
from model.app import object_detection_factory, object_detection_visualize_factory
from model.distance_detection import get_distance_and_object
from model.line_detection import find_lines
from warn_user import alert_user

utils_ops.tf = tf.compat.v1

object_detection = object_detection_factory()
object_detection_visualize = object_detection_visualize_factory()


def display_text(text, frame, y_offset):
    cv2.putText(frame, text, (50, 50 + y_offset * 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


def display_warning_for_detection(results, frame, offset):
    danger_objects = ['person', 'bicycle', 'motorcycle', 'train', 'stop sign', 'cat', 'dog', 'sheep', 'horse', 'cow']
    dangers = []
    for object_type in results:
        if object_type in danger_objects:
            dangers.append((object_type, results[object_type]))
            danger_msg = f"Danger detected: {object_type.capitalize()}"
            display_text(danger_msg, frame, offset)
            print(danger_msg)
            alert_user(danger_msg)
            offset += 1
    return offset


def display_warning_for_lines(in_lines, frame, offset):
    if not in_lines:
        danger_msg = "Not in lines!"
        display_text(danger_msg, frame, offset)
        print(danger_msg)
        offset += 1
    return offset


def display_warning_for_distance(distance, frame, offset):
    if distance < 1:
        danger_msg = "Too close!"
        display_text(danger_msg, frame, 2)
        print(danger_msg)
        offset += 1
    return offset


def display_warning_for_collision(collisions, frame, offset):
    if len(collisions) > 0:
        danger_msg = "Collision detected!"
        display_text(danger_msg, frame, 3)
        alert_user(danger_msg)
        offset += 1
    return offset


def run():
    video = cv2.VideoCapture("1.mp4")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        tf_results, results = object_detection(frame)
        collisions = get_collisions(results)
        lines, in_lines = find_lines(frame)
        distance, closest_object = get_distance_and_object(results, lines)

        offset = display_warning_for_detection(results, frame, 0)
        offset = display_warning_for_collision(collisions, frame, offset)
        offset = display_warning_for_lines(in_lines, frame, offset)
        display_warning_for_distance(distance, frame, offset)

        cv2.imshow("", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    run()
