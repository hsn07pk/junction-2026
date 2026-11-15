"""Fold the arm to its taught rest pose (rising first with the base joint unchanged, never a low sweep), gripper
open; used by stop_all.sh and park_release.sh. Poses from config/cell_calibration.json."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell  # noqa: E402
import grasp_auto as ga  # noqa: E402


def fold():
    q = ga.state()["present"]
    ga.move([q[0]] + list(cell.get("up_q_tail")) + [25.0], speed=15, comp=False)
    ga.move(list(cell.get("rest_q")) + [25.0], speed=15, comp=False)
