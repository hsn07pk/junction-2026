"""Wrist camera model (eye-in-hand), fitted offline from the sort loop's own data (I-61, 18:45Z). OFFLINE TOOL:
not used by the sort loop yet (see README.md for the validation numbers and the switch criteria).

The camera rides on the gripper. work/eyeinhand/calib.py fitted a pinhole camera with one radial term and its pose
in the TCP frame of kin.fk from 117 target-brick observations in still-arm wrist frames of 8 episodes
(any_1731, any_1740): reprojection median 5.2 px. Leave-one-episode-out, a brick position predicted from the scan
frame alone lands 0.6-7.4 mm (median 2.7) from where the servo later centred it, and the predicted TCP target is
1.1-6.0 mm (median 5.1) from the servo-converged hover. The raw scan Jacobian was 26 mm (median) / 48 mm (max) off.

Model file: app/tools/wrist_cam_model.json (copied from work/eyeinhand/wrist_cam_model.json). Missing file -> M is
None and callers fall back to the learned scan map.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

import kin

MODEL_F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrist_cam_model.json")
M = json.load(open(MODEL_F)) if os.path.exists(MODEL_F) else None
_RC = cv2.Rodrigues(np.asarray(M["rvec_tcp_cam"], float).reshape(3, 1))[0] if M else None
_DQ = np.asarray(M.get("joint_offsets_norm", [0, 0, 0, 0, 0]), float) if M else np.zeros(5)  # v3: joint offsets


def camera_pose(q):
    """Camera axes (columns, base frame) and camera centre (mm, base frame) for body joints q (normalized)."""
    p, Rm = kin.fk(list(np.asarray(list(q)[:5], float) + _DQ))
    Rm = np.asarray(Rm, float)
    return Rm @ _RC, np.asarray(p, float) + Rm @ np.asarray(M["t_tcp_cam_mm"], float)


def project(q, P) -> np.ndarray:
    """Pixel of base-frame point P (mm) seen from joints q."""
    Rbc, cb = camera_pose(q)
    X = Rbc.T @ (np.asarray(P, float) - cb)
    x, y = X[0] / X[2], X[1] / X[2]
    d = 1.0 + M["k1"] * (x * x + y * y)
    return np.array([M["f"] * x * d + M["cx"], M["f"] * y * d + M["cy"]])


def pixel_to_mat(q, u, z: float | None = None) -> np.ndarray:
    """Base-frame xy (mm) of the point at height z (default: brick top face) seen at pixel u from joints q."""
    z = M["z_top_mm"] if z is None else z
    f, cx, cy, k1 = M["f"], M["cx"], M["cy"], M["k1"]
    xd, yd = (u[0] - cx) / f, (u[1] - cy) / f
    x, y = xd, yd
    for _ in range(8):  # invert the radial term (fixed point)
        d = 1.0 + k1 * (x * x + y * y)
        x, y = xd / d, yd / d
    Rbc, cb = camera_pose(q)
    ray = Rbc @ np.array([x, y, 1.0])
    s = (z - cb[2]) / ray[2]
    return (cb + s * ray)[:2]


def target_offset(q, target_px) -> np.ndarray:
    """Mat point (brick-top height) that the servo target pixel sees, relative to the TCP xy (mm)."""
    p, _ = kin.fk(list(np.asarray(list(q)[:5], float) + _DQ))
    return pixel_to_mat(q, target_px) - np.asarray(p[:2], float)


def tcp_target(q_hover, brick_xy, target_px, iters: int = 3):
    """TCP xy (model frame, joint offsets applied) at the hover height that puts brick_xy under the servo target pixel.
    Accounts for the tool yaw change of the differential-IK move. Returns (xy, q_at_target or None)."""
    p, _ = kin.fk(list(np.asarray(list(q_hover)[:5], float) + _DQ))
    t = np.asarray(brick_xy, float) - target_offset(q_hover, target_px)
    qt = None
    for _ in range(iters):
        qt = kin.ik_rel(list(q_hover)[:5], [t[0] - p[0], t[1] - p[1], 0.0])
        if qt is None:
            break
        t = np.asarray(brick_xy, float) - target_offset(qt, target_px)
    return t, qt
