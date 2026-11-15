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
IMG_W, IMG_H = (int(v) for v in wc.M["image_size"])  # the frame size the camera model was fitted at


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

MAT_SIGNIFICANCE = 6.0  # a pixel is "not mat" when it departs from the mat by 6 robust sigmas of the mat's own spread


MAT_DARK_SHARE = 0.85  # share of the band, darkest first, that samples the mat colour: leaves out a light brick reaching in, keeps the cloth's lighter folds as mat


def lab_distance_to_mat(img: np.ndarray, brick_mask: np.ndarray, band_px: int) -> tuple[np.ndarray, dict]:
    """How far every pixel is from the mat, the mat being sampled in a band around brick_mask: colour (a*, b*) in any
    direction, lightness only upward. A shadow is the mat made darker with its own colour, so it stays mat; a brick
    face, even in shade, differs in colour or is lighter than a dark mat. Returns the distance image and the mat
    statistics, including the band's robust spread of that distance (the noise the threshold is scaled by)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    outer = cv2.dilate(brick_mask.astype(np.uint8), k) > 0
    band = outer & ~(cv2.dilate(brick_mask.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_px + 1, band_px + 1))) > 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    if band.sum() < 50:
        return np.zeros(img.shape[:2], np.float32), {"ok": False}
    lb = lab[band]
    if MAT_DARK_SHARE < 1.0:  # a brick lighter than the mat reaching into the band (prior off, bricks close) must not
        # become the "mat": the mat is sampled from the darker part of the band (on a light mat set the share to 1)
        lb = lb[lb[:, 0] <= np.percentile(lb[:, 0], 100.0 * MAT_DARK_SHARE)]
        band = band & (lab[..., 0] <= float(lb[:, 0].max()))
    med = np.median(lb, axis=0)
    dl = np.maximum(lab[..., 0] - med[0], 0.0)
    d = np.sqrt(dl * dl + (lab[..., 1] - med[1]) ** 2 + (lab[..., 2] - med[2]) ** 2)
    db = d[band]
    sig = float(1.4826 * np.median(np.abs(db - np.median(db))))
    return d, {"ok": True, "mat_lab": [round(float(x), 1) for x in med], "band_px": int(band.sum()),
               "d_med": round(float(np.median(db)), 2), "d_sigma": round(sig, 2)}


def silhouette(dist: np.ndarray, roi: np.ndarray, mat: dict | None = None) -> tuple[np.ndarray, float]:
    """Brick pixels inside roi: distance to the mat above the mat's median plus MAT_SIGNIFICANCE robust sigmas (the
    band's own noise; at least one Lab unit). Returns the mask and the separation quality (between-class variance /
    total variance of the distances in the roi, 0..1; low means brick and mat look alike)."""
    v = dist[roi]
    if v.size < 50 or float(v.max()) <= 0:
        return np.zeros(dist.shape, bool), 0.0
    if mat and mat.get("ok"):
        thr = mat["d_med"] + MAT_SIGNIFICANCE * max(mat["d_sigma"], 1.0)
    else:
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


def project_batch(q, P) -> np.ndarray:
    """Pixels of many base-frame points (N x 3, mm) seen from joints q (the camera model of wristcam, vectorised)."""
    Rbc, cb = wc.camera_pose(q)
    X = (np.asarray(P, float) - cb) @ Rbc
    x, y = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    d = 1.0 + wc.M["k1"] * (x * x + y * y)
    return np.stack([wc.M["f"] * x * d + wc.M["cx"], wc.M["f"] * y * d + wc.M["cy"]], 1)


def box_silhouette(q, cb, xy, yaw, size_xy, height, window) -> np.ndarray:
    """Mask (inside window = (x0, y0, w, h)) of the faces of a box on the mat that face the camera centre cb."""
    faces = brick_faces(xy, yaw, size_xy, height, kin.MAT_Z)
    x0, y0, w, h = window
    m = np.zeros((h, w), np.uint8)
    for f in faces.values():
        pts = np.array(f["pts"])
        if float(f["n"] @ (cb - pts.mean(axis=0))) <= 0:
            continue
        uv = project_batch(q, pts) - np.array([x0, y0])
        cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 1)
    return m > 0


def fit_box(comp: np.ndarray, q, xy0, yaw0: float, sp: dict) -> dict:
    """Analysis by synthesis: for every footprint of whole stud counts (1 to 4 per side) and both orientations, the box
    pose (x, y, yaw) whose projected visible faces best overlap the brick's silhouette (pattern search on IoU). The
    footprint and pose with the best IoU win."""
    _, cb = wc.camera_pose(q)
    ys, xs = np.nonzero(comp)
    pad = 20
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(comp.shape[1], xs.max() + pad), min(comp.shape[0], ys.max() + pad)
    win = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    obs = comp[y0:y1, x0:x1]
    n_obs = float(obs.sum())
    pitch = sp["footprint_mm"] / sp["studs"][0]
    h = sp["height_mm"]

    def iou(x, y, a, size):
        m = box_silhouette(q, cb, (x, y), a, size, h, win)
        inter = float((m & obs).sum())
        return inter / (float(m.sum()) + n_obs - inter + 1e-6)

    best = None
    for ns in range(1, 5):
        for nl in range(ns, 5):
            for turn in (0.0, math.pi / 2):
                size = (ns * pitch, nl * pitch)
                x, y, a = float(xy0[0]), float(xy0[1]), yaw0 + turn
                cur = iou(x, y, a, size)
                for step_mm, step_rad in ((4.0, 0.12), (2.0, 0.06), (1.0, 0.03), (0.5, 0.015)):
                    improved = True
                    while improved:
                        improved = False
                        for dx, dy, da in ((step_mm, 0, 0), (-step_mm, 0, 0), (0, step_mm, 0), (0, -step_mm, 0),
                                           (0, 0, step_rad), (0, 0, -step_rad)):
                            v = iou(x + dx, y + dy, a + da, size)
                            if v > cur + 1e-6:
                                x, y, a, cur, improved = x + dx, y + dy, a + da, v, True
                if best is None or cur > best["iou"]:
                    best = {"iou": cur, "xy": [x, y], "yaw": a, "studs": [ns, nl], "size_mm": [ns * pitch, nl * pitch]}
    return best


def measure_top(img: np.ndarray, q, prior_xy, yaw0: float, sp: dict) -> dict:
    """Brick centre (base xy, mm), yaw (rad, modulo 90 deg) and footprint (whole studs) from one view: segment the
    brick against the mat around the prior, then fit the box model to the silhouette (fit_box). A blob that runs to
    the edge of its search area touches a neighbour and is not measured."""
    fp = sp["footprint_mm"]
    faces = brick_faces(prior_xy, yaw0, (fp, fp), sp["height_mm"], kin.MAT_Z)
    uv = project_batch(q, np.array(faces["top"]["pts"]))
    size_px = float(np.sqrt(max(1.0, cv2.contourArea(uv.astype(np.float32)))))
    roi_r = int(round(0.9 * size_px))
    prior = poly_mask(img.shape, uv)
    roi = cv2.dilate(prior.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * roi_r + 1,) * 2)) > 0
    dist, mat = lab_distance_to_mat(img, roi, int(round(0.4 * size_px)))
    if not mat["ok"]:
        return {"ok": False, "why": "no mat around the brick"}
    sil, sep = silhouette(dist, roi, mat)
    n, lab, st, cen = cv2.connectedComponentsWithStats(sil.astype(np.uint8))
    if n <= 1:
        return {"ok": False, "why": "no brick pixels", "sep": sep}
    c0 = uv.mean(axis=0)
    k = 1 + int(np.argmin([np.hypot(*(cen[i] - c0)) - 1e-3 * st[i, 4] for i in range(1, n)]))
    comp = lab == k
    edge = cv2.dilate(roi.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    edge &= ~(cv2.erode(roi.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    if (comp & edge).any():  # the blob runs out of the search area: merged with a neighbour, not one brick
        return {"ok": False, "why": "brick blob reaches the edge of its search area (touching a neighbour)", "sep": sep}
    fit = fit_box(comp, q, prior_xy, yaw0, sp)
    yaw = (fit["yaw"] - yaw0 + math.pi / 4) % (math.pi / 2) - math.pi / 4 + yaw0  # nearest to the prior modulo 90 deg
    out = {"ok": fit["iou"] >= sp["fit_min_iou"], "xy": [round(v, 1) for v in fit["xy"]], "yaw": round(yaw, 4),
           "size_mm": [round(v, 1) for v in fit["size_mm"]], "studs": fit["studs"], "iou": round(fit["iou"], 3),
           "sep": sep, "mat": mat, "px_area": int(comp.sum())}
    if not out["ok"]:
        out["why"] = f"no footprint explains the silhouette (best IoU {out['iou']} < {sp['fit_min_iou']})"
    else:
        out["shape"] = ("2x2" if fit["studs"] == sorted(sp["studs"]) else "not_2x2",
                        f"{fit['studs'][0]}x{fit['studs'][1]} studs (silhouette IoU {out['iou']})")
    return out


def align_shift(img: np.ndarray, proj: dict, search_px: int) -> dict:
    """Shift (px) of the projected visible faces that best matches the brick silhouette in this view: the shift with
    the largest overlap (cross-correlation of the two masks), and the IoU there."""
    vis = [r["uv"] for r in proj.values() if r["cos"] > 0]
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
    sil, sep = silhouette(dist, roi, mat)
    ys, xs = np.nonzero(union)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    H, W = img.shape[:2]
    pad = np.zeros((H + 2 * search_px, W + 2 * search_px), np.float32)
    pad[search_px:search_px + H, search_px:search_px + W] = sil
    win = pad[y0:y1 + 2 * search_px, x0:x1 + 2 * search_px]
    tpl = union[y0:y1, x0:x1].astype(np.float32)
    cc = cv2.matchTemplate(win, tpl, cv2.TM_CCORR)
    _, best, _, (bx, by) = cv2.minMaxLoc(cc)
    dx, dy = int(bx) - search_px, int(by) - search_px
    inter = float(best)
    iou = inter / float(union.sum() + sil.sum() - inter + 1e-6)
    return {"ok": True, "iou": round(iou, 3), "shift": [dx, dy], "sep": sep, "mat": mat,
            "at_limit": abs(dx) >= search_px or abs(dy) >= search_px}


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


def order_views(q_start, views: list, q_end) -> list:
    """Visit order with the least joint travel (the arm moves joints together, so a hop costs its largest joint
    change), from q_start through every view back to q_end. Brute force: a survey has at most a handful of views."""
    import itertools
    hop = lambda a, b: max(abs(float(x) - float(y)) for x, y in zip(list(a)[:5], list(b)[:5]))  # noqa: E731
    best, best_cost = list(views), None
    for perm in itertools.permutations(views):
        path = [q_start] + [v["q"] for v in perm] + [q_end]
        cost = sum(hop(a, b) for a, b in zip(path, path[1:]))
        if best_cost is None or cost < best_cost:
            best, best_cost = list(perm), cost
    return best


def target_silhouette(img: np.ndarray, proj: dict, faces3d: dict, search_px: int) -> dict:
    """The surveyed brick in one view: align the projected visible faces to the image (align_shift), keep the
    not-mat component(s) under the aligned projection, and report the mat around it (for the camera noise) and the
    mean face scale (mm^2 per px over the visible faces)."""
    al = align_shift(img, proj, search_px)
    if not al.get("ok"):
        return {"ok": False, "why": al.get("why")}
    dx, dy = al["shift"]
    vis = {f: r for f, r in proj.items() if r["cos"] > 0}
    union = np.zeros(img.shape[:2], bool)
    for r in vis.values():
        union |= poly_mask(img.shape, r["uv"] + np.array([dx, dy], float))
    size_px = float(np.sqrt(max(1, int(union.sum()))))
    roi = cv2.dilate(union.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * search_px + 1,) * 2)) > 0
    dist, mat = lab_distance_to_mat(img, roi, int(round(0.4 * size_px)))
    sil, sep = silhouette(dist, roi, mat)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sil.astype(np.uint8))
    keep = np.zeros(img.shape[:2], bool)
    for i in range(1, n):
        comp = lab == i
        if (comp & union).sum() >= 0.5 * comp.sum():  # mostly under the brick: the brick, not a neighbour
            keep |= comp
    band = roi & ~cv2.dilate(union.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~sil
    area_mm2 = sum(faces3d[f]["area_mm2"] for f in vis)
    proj_px = sum(float(cv2.contourArea(np.asarray(r["uv"], np.float32))) for r in vis.values())
    rim = max(1, int(round(0.02 * size_px)))  # boundary pixels mix brick and mat: left out of the colour test
    inner = cv2.erode(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim + 1,) * 2)) > 0
    return {"ok": True, "mask": inner, "sil": keep, "mat_mask": band, "shift": [dx, dy], "iou": al["iou"],
            "at_limit": al["at_limit"],
            "sep": sep, "mm2_per_px": area_mm2 / proj_px if proj_px > 0 else 0.0,
            "faces": sorted(f for f, r in vis.items() if r["inside"])}
