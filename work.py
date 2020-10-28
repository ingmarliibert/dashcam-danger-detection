import json
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from itertools import combinations, chain
from typing import Dict, List, Callable
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from utils.rectangle import Rectangle

@dataclass
class DetectedObject:
    class_name: str
    class_id: int
    score: int
    detection_boxes: list
    tensorflow_class: dict


@dataclass
class DetectedObjects:
    # label name -> List
    detected_objects: Dict[str, List[DetectedObject]]

    def compare_specific_objects(self, label_name: str, compare: Callable[[DetectedObject, DetectedObject], bool]):
        specific_objects = self.detected_objects.get(label_name)
        return self._compare_objects(specific_objects, compare)


    @staticmethod
    def _compare_objects(to_compare: List[DetectedObject], compare: Callable[[DetectedObject, DetectedObject], bool]):
        """
        Algorithm: loop over boxes
        -> you have one box, you have to test if it overlaps with any other box, so you loop over all boxes but not over the current one.
        -> you make rectangles from boxes and test if it overlaps.
        """
        object_combinations = combinations(to_compare, 2)

        results = []
        for current_object, to_compare in object_combinations:
            if compare(current_object, to_compare):
                results.append((current_object, to_compare))

        return results

    def compare_objects(self, compare: Callable[[DetectedObject, DetectedObject], bool]):
        all_objects = list(chain.from_iterable(self.detected_objects.values()))
        return self._compare_objects(all_objects, compare)


def is_there_any_collision(current_object: DetectedObject, another_object: DetectedObject) -> bool:
    person_box = current_object.detection_boxes
    other_person_box = another_object.detection_boxes

    object_r = Rectangle(person_box[0], person_box[1], person_box[2], person_box[3])
    other_person_r = Rectangle(other_person_box[0], other_person_box[1], other_person_box[2], other_person_box[3])

    is_intersect = object_r.is_intersect(other_person_r)

    if is_intersect:
        intersection = object_r & other_person_r
        print(f'intersection = {intersection}')

        threshold = 0.03 * object_r.area

        # print(f'{threshold} {intersection.area}')
        if intersection.area > threshold:
            print('collision')
            return True

    return False


with open('car-crash.jpg.json', 'r') as f:
    result = json.loads(f.read())
    classes = result['detection_classes']
    scores = result['detection_scores']
    boxes = result['detection_boxes']

    min_score_thresh = .5

    PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    detected_objects: Dict[str, List[DetectedObject]] = defaultdict(list)

    for class_id, score, box in zip(classes, scores, boxes):
        if score < min_score_thresh:
            continue

        class_name = category_index[class_id]['name']
        detected = DetectedObject(class_name, class_id, score, box, category_index[class_id])
        detected_objects[class_name].append(detected)

    objects = DetectedObjects(detected_objects)
    results = objects.compare_specific_objects('car', is_there_any_collision)
    print(results)

    result = results[0][0]

    image_np = np.array(Image.open('./car-crash-visualize.jpg'))

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.array([result.detection_boxes]),
        np.array([result.class_id]),
        np.array([result.score]),
        result.tensorflow_class,
        None,
        use_normalized_coordinates=True,
        line_thickness=8)

    result = Image.fromarray(image_np)
    result.save("something.jpg", "JPEG")

    # persons: List[DetectedObject] = detected_objects.get('car')

