import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
from object_detection.utils import ops as utils_ops

import warn_user
from collisions import get_collisions
from model.app import object_detection_factory, object_detection_visualize_factory

# CONSTANTS
NB_COLLISION_FRAMES = 3 #frames
COLLISION_DELAY = 150 #frames


def analyse_video_for_collision(video):
    object_detection = object_detection_factory()
    object_detection_visualize = object_detection_visualize_factory()

    c_counter = 0
    delay = 0
    collisionDetected = False

    while (video.isOpened()):
        ret, frame = video.read()
        frame = cv2.resize(frame, (1280, 720))
        frame_np = np.array(frame)
        tf_results, results = object_detection(frame)
        collisions = get_collisions(results)
        image_np_visualized = object_detection_visualize(tf_results, frame_np)
        i = np.uint8(Image.fromarray(image_np_visualized))
        # print(f'collisions = {collisions}')

        # Noise reduction : if less than 3 consecutives collisions
        # do not warn
        # Also, if collision detected, set a delay until next collision
        delay += 1
        if (collisions):
            c_counter += 1
        else:
            c_counter = 0

        if (collisions and c_counter >= NB_COLLISION_FRAMES and delay > COLLISION_DELAY):
            graphical_alert_collision(i, "Collision detected !")
            warn_user.alert_user("Collision detected !")
            delay = 0
            collisionDetected = True
        elif (collisionDetected and delay < COLLISION_DELAY):
            graphical_alert_collision(i, "Collision detected !")
        elif(collisionDetected and delay >= COLLISION_DELAY):
            collisionDetected = False


        cv2.imshow("Collisions", i)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    pass


def graphical_alert_collision(frame, text):
    warn_user.write_alert(frame, text, 400, 100)


# main
vid = cv2.VideoCapture("2.mp4")
analyse_video_for_collision(vid)
