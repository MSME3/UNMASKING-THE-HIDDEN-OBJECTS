"""
Constants and target type compatibility definitions for AI Vision Search.
"""

# COCO Vehicle classes recognized by YOLOv8:
# bicycle(1), car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_YOLO_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

FOUR_WHEEL_TYPES = {"car", "bus", "truck"}
TWO_WHEEL_TYPES = {"motorcycle", "bicycle"}


def is_compatible_target_type(type1: str, type2: str) -> bool:
    """
    Evaluates whether two target types can be compared:
    - Same types (e.g. person == person, car == car)
    - 4-wheel cross-matching (e.g. car, SUV, truck, bus)
    - 2-wheel cross-matching (e.g. motorcycle, bicycle)
    """
    if type1 == type2:
        return True
    if type1 in FOUR_WHEEL_TYPES and type2 in FOUR_WHEEL_TYPES:
        return True
    if type1 in TWO_WHEEL_TYPES and type2 in TWO_WHEEL_TYPES:
        return True
    return False

