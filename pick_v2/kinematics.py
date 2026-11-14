"""SO-101 forward and inverse kinematics from the URDF, and the calibration-based normalized-unit conversion.

Conventions (sources in research.md section 4):
- Normalized body joint n in [-100, 100] maps linearly to the calibrated tick range, with n = 0 at the middle
  of the range (LeRobot PR #777 'zero position ... in middle of the range for each joint'; molmoact2 README
  lines 331-335: runtime = (ticks - mid) * 200 / (range_max - range_min)).
- Radians per normalized unit k = (range_max - range_min) / 200 * 2*pi / 4095 (molmoact2 so101.py).
- URDF joint angle q = sign * k * n + zero_offset. sign = +1 and zero_offset = 0 by default. These are NOT
  verified on hardware: the live check in runbook.md step 2 confirms them.
- The tool frame is 'gripper_frame_link' (98 mm past gripper_link). Its z axis points from the tip back to the
  wrist, so a top-down grasp has R[:, 2] = (0, 0, -1). The moving jaw closes along the tool x axis.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

BODY_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TICKS_PER_REV = 4095.0
NORM_SPAN = 200.0
R_TOP_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # tool z down, tool x forward


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF roll-pitch-yaw convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n == 0.0:
        return np.eye(3)
    x, y, z = a / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def rot_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis times angle) of a rotation matrix."""
    cos_t = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    theta = math.acos(cos_t)
    if theta < 1e-9:
        return np.zeros(3)
    if math.pi - theta < 1e-6:
        M = (R + np.eye(3)) / 2.0
        idx = int(np.argmax(np.diag(M)))
        axis = M[:, idx] / math.sqrt(max(M[idx, idx], 1e-12))
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(theta))
    return w * theta


def transform(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def _rotate_about_joint(T: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """T @ Rot(axis, angle) with a fast path for axis = z (every SO-101 joint in this URDF)."""
    if axis[0] == 0.0 and axis[1] == 0.0 and axis[2] == 1.0:
        c, s = math.cos(angle), math.sin(angle)
        out = T.copy()
        out[:, 0] = c * T[:, 0] + s * T[:, 1]
        out[:, 1] = -s * T[:, 0] + c * T[:, 1]
        return out
    return T @ transform(axis_angle_matrix(axis, angle), [0, 0, 0])


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class Chain:
    """Serial chain from base_link to the tool frame, parsed from a URDF file (read as text, never executed)."""

    def __init__(self, urdf_path: str | Path, base_link: str = "base_link", tip_link: str = "gripper_frame_link"):
        root = ET.parse(str(urdf_path)).getroot()
        by_child: dict[str, UrdfJoint] = {}
        for el in root.findall("joint"):
            origin_el = el.find("origin")
            xyz = [float(v) for v in origin_el.get("xyz", "0 0 0").split()] if origin_el is not None else [0, 0, 0]
            rpy = [float(v) for v in origin_el.get("rpy", "0 0 0").split()] if origin_el is not None else [0, 0, 0]
            axis_el = el.find("axis")
            axis = [float(v) for v in axis_el.get("xyz", "0 0 1").split()] if axis_el is not None else [0, 0, 1]
            limit_el = el.find("limit")
            lower = float(limit_el.get("lower", "-1e9")) if limit_el is not None else -math.inf
            upper = float(limit_el.get("upper", "1e9")) if limit_el is not None else math.inf
            joint = UrdfJoint(
                name=el.get("name", ""),
                kind=el.get("type", "fixed"),
                parent=el.find("parent").get("link"),
                child=el.find("child").get("link"),
                origin=transform(rpy_matrix(rpy), xyz),
                axis=np.asarray(axis, dtype=float),
                lower=lower,
                upper=upper,
            )
            by_child[joint.child] = joint
        path: list[UrdfJoint] = []
        link = tip_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(f"no joint leads to link {link!r} in {urdf_path}")
            path.append(by_child[link])
            link = by_child[link].parent
        path.reverse()
        self.joints: tuple[UrdfJoint, ...] = tuple(path)
        self.revolute: tuple[UrdfJoint, ...] = tuple(j for j in path if j.kind == "revolute")
        names = tuple(j.name for j in self.revolute)
        if names != BODY_JOINTS:
            raise ValueError(f"unexpected revolute chain {names}, expected {BODY_JOINTS}")
        self.urdf_lower = np.array([j.lower for j in self.revolute])
        self.urdf_upper = np.array([j.upper for j in self.revolute])
        self.tip_link = tip_link

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """4x4 pose of the tool frame for five body joint angles in radians (BODY_JOINTS order)."""
        T = np.eye(4)
        i = 0
        for j in self.joints:
            T = T @ j.origin
            if j.kind == "revolute":
                T = _rotate_about_joint(T, j.axis, float(q[i]))
                i += 1
        return T

    def frame_points(self, q: Sequence[float]) -> np.ndarray:
        """Origins of every joint frame plus the tool frame, shape (n, 3). Used for collision checks."""
        pts = []
        T = np.eye(4)
        i = 0
        for j in self.joints:
            T = T @ j.origin
            pts.append(T[:3, 3].copy())
            if j.kind == "revolute":
                T = _rotate_about_joint(T, j.axis, float(q[i]))
                i += 1
        pts.append(T[:3, 3].copy())
        return np.asarray(pts)


@dataclass(frozen=True)
class JointCalibration:
    name: str
    range_min: float
    range_max: float
    homing_offset: int

    @property
    def rad_per_unit(self) -> float:
        width = self.range_max - self.range_min
        if width <= 0:
            raise ValueError(f"invalid calibration range for {self.name}")
        return width / NORM_SPAN * 2.0 * math.pi / TICKS_PER_REV


class FollowerCalibration:
    """Calibration of one SO-101 arm and the normalized <-> radian conversion for its five body joints."""

    def __init__(
        self,
        joints: Mapping[str, JointCalibration],
        signs: Optional[Mapping[str, float]] = None,
        zero_offsets_rad: Optional[Mapping[str, float]] = None,
    ) -> None:
        missing = [j for j in BODY_JOINTS + ("gripper",) if j not in joints]
        if missing:
            raise ValueError(f"calibration is missing {missing}")
        self.joints = dict(joints)
        signs = dict(signs or {})
        zero_offsets_rad = dict(zero_offsets_rad or {})
        self.k = np.array([self.joints[j].rad_per_unit for j in BODY_JOINTS])
        self.sign = np.array([float(signs.get(j, 1.0)) for j in BODY_JOINTS])
        self.zero = np.array([float(zero_offsets_rad.get(j, 0.0)) for j in BODY_JOINTS])

    @classmethod
    def from_studio_json(cls, path: str | Path, robot_type: str = "SO101_Follower") -> "FollowerCalibration":
        """Read one robot entry of a Studio project export (shared/studio-api/project_robots.json format)."""
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
        entry = next(e for e in entries if e.get("type") == robot_type)
        if entry["payload"].get("unit", "normalized") != "normalized":
            raise ValueError("calibration unit is not normalized")
        joints = {
            name: JointCalibration(name, float(v["range_min"]), float(v["range_max"]), int(v["homing_offset"]))
            for name, v in entry["payload"]["calibration"].items()
        }
        return cls(joints)

    def rad_from_norm(self, n: Sequence[float]) -> np.ndarray:
        """Five body joints in normalized units to URDF radians."""
        n = np.asarray(n, dtype=float)[:5]
        return self.sign * self.k * n + self.zero

    def norm_from_rad(self, q: Sequence[float]) -> np.ndarray:
        """Five URDF radians to body joints in normalized units (inverse of rad_from_norm)."""
        q = np.asarray(q, dtype=float)[:5]
        return (q - self.zero) / (self.sign * self.k)

    def norm_bounds(self, margin_units: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        lo = np.full(5, -100.0 + margin_units)
        hi = np.full(5, 100.0 - margin_units)
        return lo, hi

    def rad_bounds(self, chain: Chain, margin_units: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Radian bounds: the calibrated normalized range intersected with the URDF joint limits."""
        lo_n, hi_n = self.norm_bounds(margin_units)
        a, b = self.rad_from_norm(lo_n), self.rad_from_norm(hi_n)
        lo = np.maximum(np.minimum(a, b), chain.urdf_lower)
        hi = np.minimum(np.maximum(a, b), chain.urdf_upper)
        return lo, hi


def _pose_error(T_target: np.ndarray, T: np.ndarray, rot_w: float) -> np.ndarray:
    p = T_target[:3, 3] - T[:3, 3]
    w = rot_log(T_target[:3, :3] @ T[:3, :3].T)
    return np.concatenate([p, rot_w * w])


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    pos_err_m: float
    rot_err_rad: float
    iterations: int

    def ok(self, pos_tol: float, rot_tol: float) -> bool:
        return self.pos_err_m <= pos_tol and self.rot_err_rad <= rot_tol


def solve_ik(
    chain: Chain,
    T_target: np.ndarray,
    q_seed: Sequence[float],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    rot_w: float = 0.3,
    max_iter: int = 120,
    max_step: float = 0.15,
) -> IKResult:
    """Damped least-squares (Levenberg-Marquardt) IK with numerical Jacobian and joint clamping."""
    q = np.clip(np.asarray(q_seed, dtype=float), lower, upper)
    e = _pose_error(T_target, chain.fk(q), rot_w)
    lam = 0.02
    it = 0
    for it in range(1, max_iter + 1):
        if np.linalg.norm(e) < 1e-9:
            break
        J = np.zeros((6, 5))
        h = 1e-6
        for i in range(5):
            dq = np.zeros(5)
            dq[i] = h
            J[:, i] = (_pose_error(T_target, chain.fk(q + dq), rot_w) - e) / h
        step = -J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
        m = float(np.max(np.abs(step)))
        if m > max_step:
            step *= max_step / m
        q_new = np.clip(q + step, lower, upper)
        e_new = _pose_error(T_target, chain.fk(q_new), rot_w)
        if np.linalg.norm(e_new) < np.linalg.norm(e):
            q, e = q_new, e_new
            lam = max(lam * 0.5, 1e-6)
        else:
            lam *= 4.0
            if lam > 10.0:
                break
    T = chain.fk(q)
    pos_err = float(np.linalg.norm(T_target[:3, 3] - T[:3, 3]))
    rot_err = float(np.linalg.norm(rot_log(T_target[:3, :3] @ T[:3, :3].T)))
    return IKResult(q=q, pos_err_m=pos_err, rot_err_rad=rot_err, iterations=it)


def solve_ik_multi(
    chain: Chain,
    T_target: np.ndarray,
    seeds: Sequence[Sequence[float]],
    lower: np.ndarray,
    upper: np.ndarray,
) -> IKResult:
    """Run IK from several seeds and keep the solution with the smallest residual, then the smallest move."""
    best: Optional[IKResult] = None
    best_key = None
    first = np.asarray(seeds[0], dtype=float)
    for s in seeds:
        r = solve_ik(chain, T_target, s, lower, upper)
        key = (round(r.pos_err_m + 0.05 * r.rot_err_rad, 4), float(np.linalg.norm(r.q - first)))
        if best_key is None or key < best_key:
            best, best_key = r, key
    assert best is not None
    return best


def top_down_pose(x: float, y: float, z: float, psi: float) -> np.ndarray:
    """Tool pose with the approach axis pointing down and the moving-jaw axis at angle psi (world frame)."""
    return transform(rot_z(psi) @ R_TOP_DOWN, [x, y, z])


def tool_jaw_direction(T: np.ndarray) -> np.ndarray:
    """World direction of the moving-jaw closing axis (tool x axis), projected on the table."""
    v = T[:3, 0]
    return v[:2] / max(np.linalg.norm(v[:2]), 1e-12)


def tcp_xyz(chain: Chain, cal: FollowerCalibration, n_body: Sequence[float]) -> np.ndarray:
    return chain.fk(cal.rad_from_norm(n_body))[:3, 3]


PAN_AXIS_XY = (0.0388353, 0.0)  # shoulder_pan origin in base_link (URDF joint 'shoulder_pan')


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# Roll that gives yaw 0 at pan 0 with R_TOP_DOWN (from the URDF geometry, FK checks in tests): roll ~ psi - ROLL_AT_PSI0.
ROLL_AT_PSI0 = -2.7438  # rad, equals the URDF lower limit of wrist_roll; the sign of the pan term is tested below


def solve_top_down(
    chain: Chain,
    cal: FollowerCalibration,
    x: float,
    y: float,
    z: float,
    psi: float,
    lower: np.ndarray,
    upper: np.ndarray,
    q_ref: Optional[Sequence[float]] = None,
    warm_start: bool = False,
) -> IKResult:
    """IK for a top-down tool at (x, y, z) with moving-jaw axis psi (world yaw, radians).

    warm_start=True uses only q_ref as the seed (about 8x faster; for a nearby target).

    Seeds: pan from the analytic azimuth (the URDF pan axis is flipped, so q_pan = -azimuth), a 2x2 grid of
    shoulder_lift / elbow seeds, and wrist_roll seeded from psi with both signs of the pan term.
    The best residual wins; ties go to the smallest move from q_ref.
    """
    az = math.atan2(y, x - PAN_AXIS_XY[0])
    pan0 = float(np.clip(-az, lower[0], upper[0]))
    roll_seeds = [float(np.clip(wrap_angle(psi - ROLL_AT_PSI0 + s * pan0), lower[4], upper[4])) for s in (1.0, -1.0)]
    seeds = []
    for lift_n in (-40.0, 20.0):
        for elbow_n in (-30.0, 40.0):
            for roll in roll_seeds:
                q = cal.rad_from_norm([0.0, lift_n, elbow_n, 62.0, 0.0])
                q[0] = pan0
                q[4] = roll
                seeds.append(q)
    ref = np.asarray(q_ref if q_ref is not None else seeds[0], dtype=float)
    if warm_start and q_ref is not None:
        seeds = [ref]
    best: Optional[IKResult] = None
    best_key = None
    T = top_down_pose(x, y, z, psi)
    for s in seeds:
        r = solve_ik(chain, T, s, lower, upper, rot_w=0.5, max_iter=80)
        key = (round(r.pos_err_m + 0.05 * r.rot_err_rad, 5), float(np.linalg.norm(r.q - ref)))
        if best_key is None or key < best_key:
            best, best_key = r, key
    assert best is not None
    return best
