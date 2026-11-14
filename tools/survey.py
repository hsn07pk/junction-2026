"""Wrist-camera survey of a brick standing alone at the inspection spot: the top face and all four side faces, each
seen from a reachable pose chosen with the arm model (kin) and the eye-in-hand camera model (wristcam).

The camera rides on the gripper beside the roll axis, so rolling the wrist orbits the camera around the brick under
the TCP, and moving the TCP shifts the brick toward the image edge where its side faces show. plan_views() searches
TCP offsets (differential IK, top-down) and wrist rolls for the pose that shows each face largest and most frontal,
with the face inside the image and clear of the jaws. choose_spot() picks the inspection spot where every face gets a
good view. The planned poses are written to a plan file by plan_survey.py; nothing here is tuned by hand: the search
ranges are the reachable workspace and the roll range, the brick dimensions come from the brick spec file.
"""
from __future__ import annotations

import json
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

SPEC_F = os.path.join(APP, "config", "brick_spec.json")
PLAN_F = os.path.join(APP, "config", "survey_plan.json")
FACES = ("top", "side0", "side1", "side2", "side3")
IMG_W, IMG_H = 640, 480


def spec() -> dict:
    with open(SPEC_F) as f:
        return json.load(f)


def brick_faces(centre_xy, yaw: float, size_xy, height: float, z0: float) -> dict:
    """Face corners (base frame, mm) and outward normals of a box standing on the mat (bottom at z0)."""
    c, s = math.cos(yaw), math.sin(yaw)
    R2 = np.array([[c, -s], [s, c]])
    hx, hy = size_xy[0] / 2.0, size_xy[1] / 2.0
    xy = [np.asarray(centre_xy, float) + R2 @ np.array(v) for v in ((hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy))]
    bot = [np.array([p[0], p[1], z0]) for p in xy]
    top = [np.array([p[0], p[1], z0 + height]) for p in xy]
    out = {"top": {"pts": top, "n": np.array([0.0, 0.0, 1.0]), "area_mm2": size_xy[0] * size_xy[1]}}
    for k, (a, b, nrm, w) in enumerate(((0, 1, (1, 0), size_xy[1]), (1, 2, (0, 1), size_xy[0]),
                                         (2, 3, (-1, 0), size_xy[1]), (3, 0, (0, -1), size_xy[0]))):
        n2 = R2 @ np.array(nrm, float)
        out[f"side{k}"] = {"pts": [bot[a], bot[b], top[b], top[a]], "n": np.array([n2[0], n2[1], 0.0]),
                           "area_mm2": w * height}
    return out


def project_faces(q, faces: dict, jaw_mask=None, margin: int = 12) -> dict:
    """Per face: image polygon, cosine between the face normal and the direction to the camera, polygon area (px),
    whether the polygon is inside the image (with margin), and the fraction covered by the static jaw mask."""
    Rbc, cb = wc.camera_pose(q)
    res = {}
    for name, f in faces.items():
        pts = f["pts"]
        ctr = np.mean(pts, axis=0)
        to_cam = cb - ctr
        cosang = float(f["n"] @ to_cam / np.linalg.norm(to_cam))
        uv = np.array([wc.project(q, p) for p in pts])
        inside = bool(np.all((uv[:, 0] >= margin) & (uv[:, 0] <= IMG_W - margin) &
                             (uv[:, 1] >= margin) & (uv[:, 1] <= IMG_H - margin)))
        area = float(0.5 * abs(np.dot(uv[:, 0], np.roll(uv[:, 1], 1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], 1))))
        jaw = 0.0
        if jaw_mask is not None and inside and area > 1:
            m = np.zeros((IMG_H, IMG_W), np.uint8)
            cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 1)
            jaw = float((jaw_mask & (m > 0)).sum()) / max(1, int(m.sum()))
        res[name] = {"uv": uv, "cos": cosang, "area_px": area, "inside": inside, "jaw": jaw}
    return res


def view_score(r: dict, min_cos: float, max_jaw: float) -> float:
    """Usable view: inside the image, facing the camera at least min_cos, jaws off the face. Score = projected area
    weighted by the cosine (a larger, more frontal face shows a defect with more pixels)."""
    if not r["inside"] or r["cos"] < min_cos or r["jaw"] > max_jaw:
        return 0.0
    return r["area_px"] * r["cos"]


def candidate_poses(q_spot_hover, offsets, lifts, rolls):
    """Reachable top-down poses around the hover above the spot: TCP offsets by differential IK, then wrist rolls
    (the roll turns the tool about its own axis, so the TCP stays; ik_rel's yaw about the base z is not reachable for a
    tilted 5-axis tool)."""
    lo, hi = kin.CAL.norm_bounds(3)
    for dx in offsets:
        for dy in offsets:
            for dz in lifts:
                q = kin.ik_rel(q_spot_hover, [dx, dy, dz])
                if q is None:
                    continue
                for r in rolls:
                    if lo[4] <= r <= hi[4]:
                        qr = list(q)
                        qr[4] = float(r)
                        yield (dx, dy, dz, r), qr


def plan_views(q_spot_hover, faces: dict, jaw_mask, search: dict) -> dict:
    best = {}
    for key, q in candidate_poses(q_spot_hover, search["offsets_mm"], search["lifts_mm"], search["rolls"]):
        proj = project_faces(q, faces, jaw_mask)
        for name, r in proj.items():
            s = view_score(r, search["min_cos"], search["max_jaw"])
            if s > best.get(name, {}).get("score", 0.0):
                best[name] = {"score": round(s, 1), "q": [round(v, 2) for v in q], "key": key,
                              "cos": round(r["cos"], 3), "area_px": round(r["area_px"]), "jaw": round(r["jaw"], 3)}
    return best



# ---- runtime: measure the brick in the top view, align the projected faces, cut the face polygons ----

def lab_distance_to_mat(img: np.ndarray, brick_mask: np.ndarray, band_px: int) -> tuple[np.ndarray, dict]:
    """Lab distance of every pixel to the mat, the mat being sampled in a band around brick_mask (median colour).
    Returns the distance image and the mat statistics."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    outer = cv2.dilate(brick_mask.astype(np.uint8), k) > 0
    band = outer & ~(cv2.dilate(brick_mask.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_px + 1, band_px + 1))) > 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    if band.sum() < 50:
        return np.zeros(img.shape[:2], np.float32), {"ok": False}
    med = np.median(lab[band], axis=0)
    d = np.linalg.norm(lab - med, axis=2)
    return d, {"ok": True, "mat_lab": [round(float(x), 1) for x in med], "band_px": int(band.sum())}


def silhouette(dist: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, float]:
    """Brick pixels inside roi: Otsu split of the distance-to-mat values (no fixed cutoff). Returns the mask and the
    separation quality (between-class variance / total variance, 0..1; low means brick and mat look alike)."""
    v = dist[roi]
    if v.size < 50 or float(v.max()) <= 0:
        return np.zeros(dist.shape, bool), 0.0
    v8 = np.clip(v / float(v.max()) * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(v8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = t / 255.0 * float(v.max())
    hi, lo = v[v > thr], v[v <= thr]
    tot = float(v.var())
    sep = 0.0 if tot <= 0 or hi.size == 0 or lo.size == 0 else float(
        hi.size * lo.size / v.size ** 2 * (hi.mean() - lo.mean()) ** 2 / tot)
    m = np.zeros(dist.shape, bool)
    m[roi] = v > thr
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return m, round(sep, 3)


def poly_mask(shape, uv) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.round(np.asarray(uv)).astype(np.int32), 1)
    return m > 0


def measure_top(img: np.ndarray, q, prior_xy, yaw0: float, sp: dict) -> dict:
    """Brick centre (base xy, mm), yaw (rad, modulo 90 deg) and footprint (mm) from a top view: project the prior
    footprint, segment the brick against the mat around it, fit the minimum-area rectangle and map its corners to the
    brick-top plane with the camera model."""
    fp = sp["footprint_mm"]
    z_top = kin.MAT_Z + sp["height_mm"]
    faces = brick_faces(prior_xy, yaw0, (fp, fp), sp["height_mm"], kin.MAT_Z)
    uv = np.array([wc.project(q, p) for p in faces["top"]["pts"]])
    size_px = float(np.sqrt(max(1.0, cv2.contourArea(uv.astype(np.float32)))))
    roi_r = int(round(0.9 * size_px))
    prior = poly_mask(img.shape, uv)
    roi = cv2.dilate(prior.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * roi_r + 1,) * 2)) > 0
    dist, mat = lab_distance_to_mat(img, roi, int(round(0.4 * size_px)))
    if not mat["ok"]:
        return {"ok": False, "why": "no mat around the brick"}
    sil, sep = silhouette(dist, roi)
    n, lab, st, cen = cv2.connectedComponentsWithStats(sil.astype(np.uint8))
    if n <= 1:
        return {"ok": False, "why": "no brick pixels", "sep": sep}
    c0 = uv.mean(axis=0)
    k = 1 + int(np.argmin([np.hypot(*(cen[i] - c0)) - 1e-3 * st[i, 4] for i in range(1, n)]))
    ys, xs = np.nonzero(lab == k)
    rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    corners = cv2.boxPoints(rect)
    w3 = np.array([wc.pixel_to_mat(q, c, z_top) for c in corners])
    ctr = w3.mean(axis=0)
    e1, e2 = w3[1] - w3[0], w3[2] - w3[1]
    yaw = math.atan2(e1[1], e1[0])
    yaw = (yaw - yaw0 + math.pi / 4) % (math.pi / 2) - math.pi / 4 + yaw0  # nearest to the prior modulo 90 deg
    return {"ok": True, "xy": [round(float(ctr[0]), 1), round(float(ctr[1]), 1)], "yaw": round(yaw, 4),
            "size_mm": sorted([round(float(np.linalg.norm(e1)), 1), round(float(np.linalg.norm(e2)), 1)]),
            "sep": sep, "mat": mat, "px_area": int(st[k, 4])}


def align_shift(img: np.ndarray, proj: dict, search_px: int) -> dict:
    """Shift (px) of the projected visible faces that best matches the brick silhouette in this view (IoU)."""
    vis = [r["uv"] for r in proj.values() if r["cos"] > 0 and r["inside"]]
    if not vis:
        return {"ok": False, "why": "no visible face"}
    union = np.zeros(img.shape[:2], bool)
    for uv in vis:
        union |= poly_mask(img.shape, uv)
    size_px = float(np.sqrt(max(1, int(union.sum()))))
    roi = cv2.dilate(union.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * search_px + 1,) * 2)) > 0
    dist, mat = lab_distance_to_mat(img, roi, int(round(0.4 * size_px)))
    if not mat["ok"]:
        return {"ok": False, "why": "no mat around the brick"}
    sil, sep = silhouette(dist, roi)
    best = (-1.0, 0, 0)
    ys, xs = np.nonzero(union)
    H, W = img.shape[:2]
    for dy in range(-search_px, search_px + 1, 2):
        for dx in range(-search_px, search_px + 1, 2):
            y2, x2 = ys + dy, xs + dx
            ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
            inter = int(sil[y2[ok], x2[ok]].sum())
            iou = inter / float(union.sum() + sil.sum() - inter + 1e-6)
            if iou > best[0]:
                best = (iou, dx, dy)
    return {"ok": True, "iou": round(best[0], 3), "shift": [best[1], best[2]], "sep": sep, "mat": mat}



# ---- runtime planning at any brick location (relative view library), occlusion by neighbours ----

LIB_F = os.path.join(APP, "config", "survey_library.json")


def library() -> list:
    try:
        with open(LIB_F) as f:
            return json.load(f)["keys"]
    except (OSError, ValueError, KeyError):
        return []


def box_poly(q, xy, yaw, size_xy, height, inflate_mm: float) -> np.ndarray:
    """Convex hull (image) of a box standing on the mat, inflated in the plane by inflate_mm (position uncertainty)."""
    fs = brick_faces(xy, yaw, (size_xy[0] + 2 * inflate_mm, size_xy[1] + 2 * inflate_mm), height, kin.MAT_Z)
    pts = [p for f in fs.values() for p in f["pts"]]
    uv = np.array([wc.project(q, p) for p in pts], np.float32)
    return cv2.convexHull(uv).reshape(-1, 2)


def occluded_frac(face_uv, q, others, inflate_mm: float, height: float) -> float:
    """Fraction of a face polygon covered by the image hulls of neighbouring bricks (each: dict with xy, yaw,
    size_mm) that stand between the camera and the face."""
    if not others:
        return 0.0
    fm = poly_mask((IMG_H, IMG_W), face_uv)
    if fm.sum() == 0:
        return 1.0
    occ = np.zeros_like(fm)
    _, cb = wc.camera_pose(q)
    for o in others:
        hull = box_poly(q, o["xy"], o.get("yaw", 0.0), o["size_mm"], height, inflate_mm)
        occ |= poly_mask((IMG_H, IMG_W), hull)
    return float((occ & fm).sum()) / float(fm.sum())


def views_here(q_hover, xy, yaw, sp: dict, jaw_mask, keys: list, others: list, inflate_mm: float,
               min_cos: float, max_jaw: float, max_occ: float) -> dict:
    """Views for a brick at xy (yaw) from the hover q_hover: every library key is tried (differential IK + roll), each
    face gets its best unoccluded view, then a greedy cover merges faces that one pose shows well enough (at least
    half of that face's best score). Returns {"views": [{"q", "faces"}], "missing": [faces without a view]}."""
    fp = sp["footprint_mm"]
    faces = brick_faces(xy, yaw, (fp, fp), sp["height_mm"], kin.MAT_Z)
    lo, hi = kin.CAL.norm_bounds(3)
    cands = []
    seen = {}
    for dx, dy, dz, roll in [(0, 0, 0, float(q_hover[4]))] + [tuple(k) for k in keys]:  # the hover itself costs no move
        k = (dx, dy, dz)
        if k not in seen:
            seen[k] = list(q_hover)[:5] if k == (0, 0, 0) else kin.ik_rel(q_hover, [dx, dy, dz])
        q = seen[k]
        if q is None or not (lo[4] <= roll <= hi[4]):
            continue
        q = list(q)
        q[4] = float(roll)
        proj = project_faces(q, faces, jaw_mask)
        sc = {}
        for name, r in proj.items():
            s = view_score(r, min_cos, max_jaw)
            if s > 0 and occluded_frac(r["uv"], q, others, inflate_mm, sp["height_mm"]) <= max_occ:
                sc[name] = s
        if sc:
            cands.append((q, sc))
    best = {f: max((c[1].get(f, 0.0) for c in cands), default=0.0) for f in FACES}
    missing = [f for f in FACES if best[f] <= 0]
    todo = {f for f in FACES if best[f] > 0}
    views = []
    while todo:
        q, sc = max(cands, key=lambda c: (sum(1 for f in todo if c[1].get(f, 0) >= 0.5 * best[f]),
                                          sum(c[1].get(f, 0) for f in todo)))
        cov = [f for f in todo if sc.get(f, 0) >= 0.5 * best[f]]
        if not cov:
            break
        views.append({"q": [round(v, 2) for v in q], "faces": cov})
        todo -= set(cov)
    return {"views": views, "missing": missing + sorted(todo)}
