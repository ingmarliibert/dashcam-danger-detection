from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from object_detection.utils import label_map_util

from model.util import run_inference_for_single_image
from utils.rectangle import Rectangle
from object_detection.utils import visualization_utils as vis_util
import numpy as np


@dataclass
class TensorflowResults:
    classes: any
    scores: any
    boxes: any


@dataclass
class DetectedObject:
    class_name: str
    class_id: int
    tensorflow_class: dict
    score: int
    detection_boxes: list
    detection_rectangle: Rectangle


def organize_detections(result: TensorflowResults) -> Tuple[TensorflowResults, Dict[str, List[DetectedObject]]]:
    classes = result.classes
    scores = result.scores
    boxes = result.boxes

    filtered_classes = []
    filtered_scores = []
    filtered_boxes = []

    min_score_thresh = .6

    PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    detected_objects: Dict[str, List[DetectedObject]] = defaultdict(list)

    for class_id, score, box in zip(classes, scores, boxes):
        if score < min_score_thresh:
            continue

        filtered_classes.append(class_id)
        filtered_scores.append(score)
        filtered_boxes.append(box)

        class_name = category_index[class_id]['name']

        rectangle = Rectangle(box[0], box[1], box[2], box[3])
        detected = DetectedObject(class_name, class_id, category_index[class_id], score, box, rectangle)

        detected_objects[class_name].append(detected)

    return TensorflowResults(np.asarray(filtered_classes), np.asarray(filtered_scores), np.asarray(filtered_boxes)), detected_objects


def object_detection(detection_model, image: np.array):
    result = run_inference_for_single_image(detection_model, image)

    classes = result['detection_classes']
    scores = result['detection_scores']
    boxes = result['detection_boxes']

    tf_results = TensorflowResults(classes, scores, boxes)

    return organize_detections(tf_results)

def object_detection_visualize(category_index, objects: TensorflowResults, image: np.array):

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        objects.boxes,
        objects.classes,
        objects.scores,
        category_index,
        # instance_masks=output_dict.get('detection_masks_reframed', None),
        instance_masks=None,
        use_normalized_coordinates=True,
        line_thickness=8)

    return image
