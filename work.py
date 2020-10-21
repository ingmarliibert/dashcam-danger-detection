import json
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List
import numpy as np
from object_detection.utils import label_map_util

from utils.rectangle import Rectangle


@dataclass
class DetectedObject:
    class_name: str
    class_id: int
    score: int
    detection_boxes: list


def is_there_any_collision(objects: List[DetectedObject]):
    object_combinations = combinations(objects, 2)

    """
    Algorithm: loop over boxes
    -> you have one box, you have to test if it overlaps with any other box, so you loop over all boxes but not over the current one.
    -> you make rectangles from boxes and test if it overlaps.
    """
    for current_object, another_object in object_combinations:
        person_box = current_object.detection_boxes
        other_person_box = another_object.detection_boxes

        object_r = Rectangle(person_box[0], person_box[1], person_box[2], person_box[3])
        other_person_r = Rectangle(other_person_box[0], other_person_box[1], other_person_box[2], other_person_box[3])

        # person_r_area = rect_area(object_r)
        # col_threshold = person_r_area

        # area_between = area(object_r, other_person_r)
        # is_collision = (area_between and area_between > col_threshold)

        is_intersect = object_r.is_intersect(other_person_r)

        if is_intersect:
            intersection = object_r & other_person_r
            print(f'intersection = {intersection}')

            threshold = 0.06 * object_r.area

            # print(f'{threshold} {intersection.area}')
            if intersection.area > threshold:
                print('collision')


with open('car-crash.jpg.json', 'r') as f:
    result = json.loads(f.read())
    classes = result['detection_classes']
    scores = result['detection_scores']
    boxes = result['detection_boxes']

    min_score_thresh = .5

    PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    detected_objects: Dict[str, List[DetectedObject]] = defaultdict(list)

    for class_id, score, box in zip(classes, scores, boxes) :
        if score < min_score_thresh:
            continue

        class_name = category_index[class_id]['name']
        detected = DetectedObject(class_name, class_id, score, box)
        detected_objects[class_name].append(detected)

    persons: List[DetectedObject] = detected_objects.get('car')

    is_there_any_collision(persons)
