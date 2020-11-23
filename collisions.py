from dataclasses import dataclass
from itertools import combinations, chain
from typing import Dict, List, Callable

from model.object_detect import DetectedObject
from utils.rectangle import Rectangle


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


def is_collision(current_object: DetectedObject, another_object: DetectedObject) -> bool:
    # person_box = current_object.detection_boxes
    # other_person_box = another_object.detection_boxes

    # object_r = Rectangle(person_box[0], person_box[1], person_box[2], person_box[3])
    # other_person_r = Rectangle(other_person_box[0], other_person_box[1], other_person_box[2], other_person_box[3])

    object_r = current_object.detection_rectangle
    other_person_r = another_object.detection_rectangle

    is_intersect = object_r.is_intersect(other_person_r)

    if is_intersect:
        intersection = object_r & other_person_r
        print(f'intersection = {intersection}')

        # increase threshold to get stronger, they will have to collide more to get detected as collision
        threshold = 0.04 * object_r.area

        # print(f'{threshold} {intersection.area}')
        if intersection.area > threshold:
            return True

    return False


def get_collisions(detected_objects):
    all_objects = list(chain.from_iterable(detected_objects.values()))

    """
    Algorithm: loop over boxes
    -> you have one box, you have to test if it overlaps with any other box, so you loop over all boxes but not over the current one.
    -> you make rectangles from boxes and test if it overlaps.
    """
    object_combinations = combinations(all_objects, 2)

    results = []
    for current_object, to_compare in object_combinations:
        if is_collision(current_object, to_compare):
            results.append([current_object, to_compare])

    return results

# TODO: Maxime, finish this function to work with multiple objects "results".
# def visualize_collision():
#     car_collisions = results[0]
#
#     first_car = car_collisions[0]
#     full_rectangle = first_car
#
#     for car_collision in car_collisions[1:]:
#         # merge Rectangles to create a huge one :)
#         min_x = min(full_rectangle.detection_rectangle.min_x, car_collision.detection_rectangle.min_x)
#         max_x = max(full_rectangle.detection_rectangle.max_x, car_collision.detection_rectangle.max_x)
#         min_y = min(full_rectangle.detection_rectangle.min_y, car_collision.detection_rectangle.min_y)
#         max_y = max(full_rectangle.detection_rectangle.max_y, car_collision.detection_rectangle.max_y)
#
#         full_rectangle = Rectangle(min_x, min_y, max_x, max_y)
#
#     print(f'full_rectangle={full_rectangle}')
#
#     image_np = np.array(Image.open('./car-crash-visualize.jpg'))
#
#     # # Visualization of the results of a detection.
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         np.array([full_rectangle.bounding_box]),
#         np.array([first_car.class_id]),
#         np.array([first_car.score]),
#         first_car.tensorflow_class,
#         None,
#         use_normalized_coordinates=True,
#         line_thickness=8)
#
#     result = Image.fromarray(image_np)
#     result.save("something.jpg", "JPEG")
