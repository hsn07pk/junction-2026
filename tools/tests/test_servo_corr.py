"""Servo Jacobian correction learned from steps: converges to a rotated camera, ignores tracking jumps."""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pick_any as pa  # noqa: E402


def _fresh(tmp):
    pa.SERVO_CORR_FILE = os.path.join(tmp, "servo_corr.json")
    pa.SERVO_CORR.update({"C": np.eye(2), "samples": [], "loaded": True})


def test_learns_a_rotated_camera():
    a = math.radians(12.0)
    C_true = 1.05 * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    rng = np.random.default_rng(1)
    saved = (pa.SERVO_CORR_FILE, dict(pa.SERVO_CORR))
    try:
        with tempfile.TemporaryDirectory() as t:
            _fresh(t)
            for _ in range(25):
                u = rng.uniform(-60, 60, 2)
                pa.learn_servo_corr(u, C_true @ u + rng.normal(0, 1.5, 2))
            assert np.abs(pa.servo_corr() - C_true).max() < 0.05, pa.servo_corr()
            assert os.path.exists(pa.SERVO_CORR_FILE)
    finally:
        pa.SERVO_CORR_FILE = saved[0]
        pa.SERVO_CORR.clear()
        pa.SERVO_CORR.update(saved[1])


def test_ignores_small_steps_and_jumps():
    saved = (pa.SERVO_CORR_FILE, dict(pa.SERVO_CORR))
    try:
        with tempfile.TemporaryDirectory() as t:
            _fresh(t)
            pa.learn_servo_corr(np.array([3.0, 2.0]), np.array([30.0, -40.0]))      # too small to teach anything
            pa.learn_servo_corr(np.array([40.0, 0.0]), np.array([-200.0, 150.0]))   # the tracker jumped to another blob
            assert np.allclose(pa.servo_corr(), np.eye(2)) and pa.SERVO_CORR["samples"] == []
    finally:
        pa.SERVO_CORR_FILE = saved[0]
        pa.SERVO_CORR.clear()
        pa.SERVO_CORR.update(saved[1])
