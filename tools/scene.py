"""What is on the mat, measured from a wrist-camera image and the camera model: the mat itself, everything that is not
mat, whether a spot or a brick's surroundings are free, and where the two plates are.

Nothing here knows fixed colours or positions. The mat is the dominant colour of the image (mode of a coarse Lab
histogram); "not mat" is an Otsu split of the distance to it; plates are the not-mat regions far larger than a brick
(the size follows from the brick spec and the camera model), told apart by which of the two plate hues (blue and
green, the task's definition of the plates) is nearer; their centre and radius come from a circle fit on the mat plane.
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP  # noqa: E402

sys.path.insert(0, os.path.join(APP, "eyeinhand"))
import kin  # noqa: E402
import wristcam as wc  # noqa: E402

PLATE_HUE_DEG = {"blue": 240.0, "green": 120.0}  # the task defines the plates by these colours (hue circle, degrees)


def mat_colour(img: np.ndarray) -> np.ndarray:
    """Lab colour of the mat: the mode of a coarse Lab histogram of the image (the mat covers most of the view)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    q = (lab // 16).reshape(-1, 3).astype(np.int32)
    key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    mode = np.bincount(key).argmax()
    sel = key == mode
    return np.median(lab.reshape(-1, 3)[sel], axis=0).astype(np.float32)


def not_mat(img: np.ndarray, mat: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Pixels that are not mat (Otsu split of the Lab distance to the mat colour) and the split's threshold."""
    mat = mat_colour(img) if mat is None else mat
    d = np.linalg.norm(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) - mat, axis=2)
    top = float(d.max())
    if top <= 0:
        return np.zeros(img.shape[:2], bool), 0.0
    t, _ = cv2.threshold(np.clip(d / top * 255.0, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = t / 255.0 * top
    m = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return m, thr


def footprint_mask(q, shape, pts3d) -> np.ndarray:
    """Image mask of the convex hull of base-frame points seen from joints q."""
    uv = np.array([wc.project(q, p) for p in pts3d], np.float32)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(uv).astype(np.int32), 1)
    return m > 0


def cylinder_pts(xy, r_mm: float, z0: float, z1: float, n: int = 36) -> list:
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = [(xy[0] + r_mm * math.cos(t), xy[1] + r_mm * math.sin(t)) for t in a]
    return [np.array([x, y, z]) for x, y in ring for z in (z0, z1)]


def brick_area_px(q, xy, sp: dict) -> float:
    """Projected area (px) of one brick's top face at xy seen from q: the unit for 'object size' in this image."""
    h = sp["footprint_mm"] / 2.0
    z = kin.MAT_Z + sp["height_mm"]
    uv = np.array([wc.project(q, np.array([xy[0] + dx, xy[1] + dy, z])) for dx, dy in ((h, h), (h, -h), (-h, -h), (-h, h))],
                  np.float32)
    return float(cv2.contourArea(uv))


def objects_in(img: np.ndarray, q, region: np.ndarray, ignore: np.ndarray | None, min_px: float) -> list:
    """Not-mat components inside region (image mask), outside ignore, larger than min_px. Returns their areas."""
    nm, _ = not_mat(img)
    m = nm & region
    if ignore is not None:
        m &= ~ignore
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return [int(st[i, 4]) for i in range(1, n) if st[i, 4] >= min_px]


def spot_free(img: np.ndarray, q, xy, sp: dict, margin_mm: float, ignore: np.ndarray | None = None) -> dict:
    """Is a spot free for a brick (nothing but mat within half the footprint diagonal plus margin_mm, up to a
    brick's height)? Objects smaller than a tenth of a brick's top face are noise. Off-image spots are not free."""
    r = sp["footprint_mm"] * math.sqrt(0.5) + margin_mm
    pts = cylinder_pts(xy, r, kin.MAT_Z, kin.MAT_Z + sp["height_mm"])
    uv = np.array([wc.project(q, p) for p in pts])
    H, W = img.shape[:2]
    if not np.all((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)):
        return {"free": False, "why": "spot not fully in view"}
    region = footprint_mask(q, img.shape, pts)
    objs = objects_in(img, q, region, ignore, 0.1 * brick_area_px(q, xy, sp))
    return {"free": not objs, "objects_px": objs, "radius_mm": round(r, 1)}


def isolated(img: np.ndarray, q, xy, yaw: float, sp: dict, radius_mm: float, own_margin_mm: float,
             ignore: np.ndarray | None = None) -> dict:
    """Is the brick at xy (yaw) alone within radius_mm? Its own projected box, grown by own_margin_mm (position
    uncertainty), is ignored; any other object (brick, plate rim, cable, hand) inside the radius counts."""
    import survey as sv  # local: survey imports scene-free modules only
    fp = sp["footprint_mm"] + 2 * own_margin_mm
    own = sv.brick_faces(xy, yaw, (fp, fp), sp["height_mm"], kin.MAT_Z)
    own_mask = footprint_mask(q, img.shape, [p for f in own.values() for p in f["pts"]])
    pts = cylinder_pts(xy, radius_mm, kin.MAT_Z, kin.MAT_Z + sp["height_mm"])
    region = footprint_mask(q, img.shape, pts)
    if ignore is not None and ignore.shape == own_mask.shape:
        own_mask = own_mask | ignore
    objs = objects_in(img, q, region, own_mask, 0.1 * brick_area_px(q, xy, sp))
    return {"isolated": not objs, "objects_px": objs, "radius_mm": round(radius_mm, 1)}


def _hue_deg(bgr_mean) -> float:
    px = np.uint8([[[int(round(c)) for c in bgr_mean]]])
    return float(cv2.cvtColor(px, cv2.COLOR_BGR2HSV_FULL)[0, 0, 0]) * 360.0 / 256.0


def _chroma_stats(img: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Median chroma (Lab sqrt(a*^2 + b*^2)) of the masked pixels and its robust spread."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    c = np.hypot(lab[..., 1][mask] - 128.0, lab[..., 2][mask] - 128.0)
    med = float(np.median(c)) if c.size else 0.0
    return med, float(1.4826 * np.median(np.abs(c - med))) if c.size else 0.0


def _circle_fit(xy: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    ctr = c[:2]
    return ctr, float(math.sqrt(max(c[2] + ctr @ ctr, 0.0)))


def _hue_median(img: np.ndarray, mask: np.ndarray) -> float:
    """Circular median-like hue (degrees) of the masked pixels: the direction of the mean unit hue vector over the most
    saturated half (a few pixels of another colour at the rim do not drag it)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[..., 0][mask].astype(np.float32) * (2 * math.pi / 256.0)
    s = hsv[..., 1][mask].astype(np.float32)
    if h.size == 0:
        return 0.0
    top = s >= np.median(s)
    a = math.atan2(float(np.sin(h[top]).sum()), float(np.cos(h[top]).sum()))
    return (math.degrees(a) + 360.0) % 360.0


def _circle_fit_r(xy: np.ndarray, r: float, c0) -> tuple[np.ndarray, float, float]:
    """Centre of a circle of known radius r through the points (Gauss-Newton from c0, the free fit's centre moved to
    the side the arc bulges away from), its rms residual and the arc the points cover (degrees)."""
    c = np.asarray(c0, float)
    m = xy.mean(axis=0)
    if np.linalg.norm(c - m) > 3 * r or not np.isfinite(c).all():
        mid = xy[len(xy) // 2]
        n = mid - m
        c = mid - r * n / max(np.linalg.norm(n), 1e-9) if np.linalg.norm(n) > 0 else m
    for _ in range(30):
        d = xy - c
        dist = np.hypot(d[:, 0], d[:, 1])
        res = dist - r
        J = -d / np.maximum(dist, 1e-9)[:, None]
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        c = c + step
        if np.linalg.norm(step) < 1e-4:
            break
    d = xy - c
    rms = float(np.sqrt(np.mean((np.hypot(d[:, 0], d[:, 1]) - r) ** 2)))
    ang = np.sort(np.arctan2(d[:, 1], d[:, 0]))
    arc = math.degrees(2 * math.pi - float(np.diff(np.concatenate([ang, ang[:1] + 2 * math.pi])).max()))
    return c, rms, arc


def plates(img: np.ndarray, q, sp: dict, z_mm: float | None = None, ignore: np.ndarray | None = None) -> dict:
    """The blue and the green plate in this image: not-mat regions of plate colour (hue nearer one plate colour than
    the other, chroma well above the mat's), larger than a brick. Each gets its pixel mask, and, when its rim seen
    inside the image is a long enough arc (at least a quarter turn about the fitted centre, points within 5 % of the
    radius), the circle on the mat (centre, radius; the image edge is not the rim) at least twice the largest brick.
    ignore: pixels that are never scene (the jaws in the wrist view).
    Returns {colour: {"mask", "px", "hue_deg", "reliable", and when reliable "xy", "r_mm", "rms_mm", "arc_deg"}}."""
    z = kin.MAT_Z if z_mm is None else z_mm
    nm, _ = not_mat(img)
    if ignore is not None and ignore.shape == nm.shape:
        nm &= ~ignore
    mat_c, mat_s = _chroma_stats(img, ~nm)
    n, lab, st, _ = cv2.connectedComponentsWithStats(nm.astype(np.uint8))
    ctr_px = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0])
    unit = brick_area_px(q, wc.pixel_to_mat(q, ctr_px, kin.MAT_Z + sp["height_mm"]), sp)
    r_min = math.sqrt(4.0 / math.pi) * sp["footprint_mm"]  # area of four footprints
    r_known = kin.cell.get("plate_radius_mm")
    found = {}
    H, W = img.shape[:2]
    near_ignore = cv2.dilate(ignore.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0 \
        if ignore is not None and ignore.shape == nm.shape else None
    for i in range(1, n):
        if st[i, 4] < unit:  # smaller than one brick here: not a plate
            continue
        m = lab == i
        reg_c, _ = _chroma_stats(img, m)
        if reg_c <= mat_c + 6.0 * max(mat_s, 1e-3):  # hue means nothing without colour well above the mat's noise
            continue
        h = _hue_median(img, m)
        dist = {k: min(abs(h - v), 360.0 - abs(h - v)) for k, v in PLATE_HUE_DEG.items()}
        name = min(dist, key=dist.get)
        if dist[name] > abs(PLATE_HUE_DEG["blue"] - PLATE_HUE_DEG["green"]) / 2.0:  # nearer neither plate colour
            continue
        rec = {"mask": m, "px": int(st[i, 4]), "hue_deg": round(h, 1), "reliable": False}
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = max(cs, key=len).reshape(-1, 2)
        inner = pts[(pts[:, 0] > 1) & (pts[:, 0] < W - 2) & (pts[:, 1] > 1) & (pts[:, 1] < H - 2)]  # image edge is not the rim
        if near_ignore is not None and len(inner):
            inner = inner[~near_ignore[inner[:, 1], inner[:, 0]]]  # nor is the edge of what hides the plate (the jaws)
        if len(inner) >= 20:
            w = np.array([wc.pixel_to_mat(q, p, z) for p in inner[:: max(1, len(inner) // 200)]])
            c, r = _circle_fit(w)
            rms = float(np.sqrt(np.mean((np.hypot(w[:, 0] - c[0], w[:, 1] - c[1]) - r) ** 2)))
            ang = np.sort(np.arctan2(w[:, 1] - c[1], w[:, 0] - c[0]))
            gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * math.pi]))
            arc = math.degrees(2 * math.pi - float(gaps.max()))
            if r >= r_min and rms <= 0.05 * r and arc >= 90.0:
                rec.update(reliable=True, xy=[round(float(c[0]), 1), round(float(c[1]), 1)], r_mm=round(r, 1),
                           rms_mm=round(rms, 2), arc_deg=round(arc, 1))
            elif r_known:  # a short arc fixes the centre when the radius is known
                c, rms, arc = _circle_fit_r(w, r_known, c)
                if rms <= 0.05 * r_known and arc >= 30.0:
                    rec.update(reliable=True, xy=[round(float(c[0]), 1), round(float(c[1]), 1)], r_mm=r_known,
                               rms_mm=round(rms, 2), arc_deg=round(arc, 1), radius="known")
        if name not in found or rec["px"] > found[name]["px"]:
            found[name] = rec
    return found
