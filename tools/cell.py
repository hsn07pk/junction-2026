"""Measured geometry of the robot cell (config/cell_calibration.json): one place for the values that depend on how
this robot, mat and gripper are set up, each with how it was measured."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP  # noqa: E402

F = os.path.join(APP, "config", "cell_calibration.json")
C = json.load(open(F))


def get(key):
    return C[key]["value"]


def held_threshold() -> float:
    """Gripper reading above which the jaws hold something: midway between closing on nothing and on a brick."""
    return 0.5 * (get("gripper_empty_closed") + get("gripper_held_min"))
