#!/usr/bin/env python3
"""Kinematics helpers on top of app/pick_v2/kinematics.py (URDF FK/IK, normalized units).

Validated 17:13Z against depth: FK TCP z differences match the measured jaw-tip heights within 2 mm over 50 mm
(hover_al -> desc1 -> grasp). Mat surface = TCP z MAT_Z (calibrated: grasp FK z -32.3 mm with the lowest jaw
tip 6 mm above the mat). All public lengths are millimetres; angles radians.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pick_v2.kinematics import Chain, FollowerCalibration, solve_top_down  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP, R  # noqa: E402  (roots from this file's location)
URDF = (f"{R}/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/robotics-ai/"
        "platforms/so101/urdf/so101.urdf")
CH = Chain(URDF)
CAL = FollowerCalibration.from_studio_json(f"{APP}/motion/project_robots.json", "SO101_Follower")
LO, HI = CAL.rad_bounds(CH, margin_units=3)
import cell  # noqa: E402
MAT_Z = cell.get("mat_z_mm")  # TCP z (mm) when the TCP touches the mat
HOVER_DZ = cell.get("hover_dz_mm")  # TCP height above the mat at the hover
CALF = f"{APP}/tools/wrist_calib.json"


def fk(n):
    T = CH.fk(CAL.rad_from_norm(list(n)[:5]))
    return T[:3, 3] * 1000.0, T[:3, :3]


def psi_of(n):
    _, Rm = fk(n)
    return math.atan2(Rm[1, 0], Rm[0, 0])


def ik(x, y, z, psi, q_ref, tol_mm=1.5, tol_rot=0.04):
    """Top-down IK. x, y, z in mm (base frame). Returns 5 normalized body joints or None."""
    ref = CAL.rad_from_norm(list(q_ref)[:5])
    r = solve_top_down(CH, CAL, x / 1000.0, y / 1000.0, z / 1000.0, psi, LO, HI, q_ref=ref, warm_start=True)
    if r.pos_err_m * 1000 > tol_mm or r.rot_err_rad > tol_rot:
        r = solve_top_down(CH, CAL, x / 1000.0, y / 1000.0, z / 1000.0, psi, LO, HI, q_ref=ref)
    if r.pos_err_m * 1000 > tol_mm or r.rot_err_rad > tol_rot:
        return None
    n = CAL.norm_from_rad(r.q)
    if np.any(np.abs(n) > 99):
        return None
    return [round(float(v), 2) for v in n]


def load_calib():
    if os.path.exists(CALF):
        return json.load(open(CALF))
    return None


def m_base(psi, cal=None):
    """Wrist-cam px per mm of TCP motion along base x, y at the hover height, for tool yaw psi."""
    cal = cal or load_calib()
    Mt = np.array(cal["M_tool"])
    c, s = math.cos(psi), math.sin(psi)
    R2 = np.array([[c, -s], [s, c]])
    return Mt @ R2.T


from pick_v2.kinematics import solve_ik  # noqa: E402

PAN_XY_MM = (38.8353, 0.0)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ik_rel(q, d_mm, dyaw=0.0, tol_mm=1.0, tol_rot=0.03):
    """Differential IK from the CURRENT configuration (warm start, small moves): translate the TCP by d_mm (base
    frame) keeping the in-plane tool tilt (orientation follows the change of azimuth about the pan axis), plus an
    optional yaw dyaw (rad) about the base z. Absolute FK offsets cancel to first order; returns 5 joints or None."""
    qr = CAL.rad_from_norm(list(q)[:5])
    T = CH.fk(qr)
    p = T[:3, 3] * 1000.0
    pn = p + np.asarray(d_mm, dtype=float)
    a0 = math.atan2(p[1] - PAN_XY_MM[1], p[0] - PAN_XY_MM[0])
    a1 = math.atan2(pn[1] - PAN_XY_MM[1], pn[0] - PAN_XY_MM[0])
    T2 = np.eye(4)
    T2[:3, :3] = _rz(dyaw) @ _rz(a1 - a0) @ T[:3, :3]
    T2[:3, 3] = pn / 1000.0
    r = solve_ik(CH, T2, qr, LO, HI, rot_w=0.5, max_iter=120)
    if r.pos_err_m * 1000 > tol_mm or r.rot_err_rad > tol_rot:
        return None
    n = CAL.norm_from_rad(r.q)
    if np.any(np.abs(n) > 99):
        return None
    return [round(float(v), 2) for v in n]


def ik_dir(p_mm, a_dir, q_ref, tol_mm=1.5, tol_dir=0.03, iters=150):
    """IK for the TCP at p_mm (base frame) with the tool's approach axis (tool z) along a_dir, wrist roll kept from
    q_ref (the roll turns about the approach axis and changes neither). On a 5-axis arm the approach axis can only
    lie in the vertical plane through the pan axis and the TCP: a_dir off that plane has no exact solution and the
    result is None. Damped least squares on the first four joints. Returns 5 normalised joints or None."""
    q = np.asarray(CAL.rad_from_norm(list(q_ref)[:5]), float)
    pt = np.asarray(p_mm, float) / 1000.0
    at = np.asarray(a_dir, float) / np.linalg.norm(a_dir)

    def err(qq):
        T = CH.fk(qq)
        return np.concatenate([T[:3, 3] - pt, 0.05 * (T[:3, 2] - at)])

    lam = 0.01
    e = err(q)
    for _ in range(iters):
        J = np.zeros((6, 4))
        for i in range(4):
            dq = np.zeros(5)
            dq[i] = 1e-6
            J[:, i] = (err(q + dq) - e) / 1e-6
        step = -J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
        m = float(np.max(np.abs(step)))
        if m > 0.15:
            step *= 0.15 / m
        qn = q.copy()
        qn[:4] = np.clip(q[:4] + step, LO[:4], HI[:4])
        en = err(qn)
        if float(en @ en) < float(e @ e):
            q, e, lam = qn, en, max(lam / 2, 1e-4)
        else:
            lam *= 4
        if np.linalg.norm(e[:3]) * 1000 < 0.05 and np.linalg.norm(e[3:]) / 0.05 < 1e-3:
            break
    T = CH.fk(q)
    if np.linalg.norm(T[:3, 3] - pt) * 1000 > tol_mm or np.linalg.norm(T[:3, 2] - at) > tol_dir:
        return None
    n = CAL.norm_from_rad(q)
    if np.any(np.abs(n) > 99):
        return None
    return [round(float(v), 2) for v in n]
