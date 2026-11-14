"""Per-face defect test that does not depend on the brick's colour or the defect's colour.

A clean DUPLO face is one plastic colour under smooth shading. Within the face, every pixel's chroma (Lab a*, b*)
stays close to the face's own base chroma, and its lightness stays close to a planar shading model of the face.
Tape, stickers, ink, paint or dirt of any colour break one of the two. Each face is compared only with itself, so
nothing here knows brick colours or tape colours.

Per-pixel deviations are expressed in units of the face's own robust spread (median absolute deviation), floored by
the sensor noise level from the model file, so one cutoff works for dark and bright plastics. Connected anomalous
pixels are measured in mm^2 on the face (the face's area in mm^2 is known from the brick model). The cutoffs and the
area limit come from a model file written by fit_faces.py from labelled survey views; without a model file every
face is UNCERTAIN.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

CLIP = 250  # a channel at the sensor ceiling carries no colour: such pixels are left out (sensor property)


def load_model(path: str) -> dict | None:
    try:
        with open(path) as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    need = ("k_chroma", "k_light", "noise_chroma", "noise_light", "erode_frac", "max_region_mm2", "min_valid_frac")
    return m if all(k in m for k in need) else None


def face_mask(shape, poly, erode_frac: float) -> np.ndarray:
    """Pixels of the face polygon, eroded by erode_frac of the polygon's shorter side (keeps edges and small
    projection errors out of the test)."""
    m = np.zeros(shape[:2], np.uint8)
    pts = np.asarray(poly, np.float32)
    cv2.fillConvexPoly(m, np.round(pts).astype(np.int32), 1)
    sides = [float(np.linalg.norm(pts[i] - pts[(i + 1) % len(pts)])) for i in range(len(pts))]
    r = int(round(erode_frac * min(sides)))
    if r > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return m > 0


def _mad_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else 0.0


def _plane_fit(u: np.ndarray, v: np.ndarray, y: np.ndarray, iters: int = 3) -> np.ndarray:
    """Robust planar fit y ~ c0 + c1 u + c2 v (refit without the points beyond 2.5 robust sigmas)."""
    A = np.stack([np.ones_like(u), u, v], 1)
    keep = np.ones(y.size, bool)
    c = np.zeros(3)
    for _ in range(iters):
        if keep.sum() < 10:
            break
        c, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        r = y - A @ c
        s = _mad_sigma(r[keep])
        if s <= 0:
            break
        keep = np.abs(r) <= 2.5 * s
    return A @ c


def deviations(img: np.ndarray, mask: np.ndarray) -> dict | None:
    """Per-pixel chroma and lightness deviations of one face, in raw Lab units, with the face's robust spreads."""
    valid = mask & (img.max(axis=2) < CLIP)
    ys, xs = np.nonzero(valid)
    if ys.size < 30:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[ys, xs, 0], lab[ys, xs, 1], lab[ys, xs, 2]
    a0, b0 = float(np.median(a)), float(np.median(b))
    dC = np.hypot(a - a0, b - b0)
    u, v = xs.astype(np.float32), ys.astype(np.float32)
    rL = L - _plane_fit(u - u.mean(), v - v.mean(), L)
    return {"ys": ys, "xs": xs, "dC": dC, "dL": np.abs(rL), "sC": _mad_sigma(dC), "sL": _mad_sigma(rL),
            "base_lab": [float(np.median(L)), a0, b0], "n_valid": int(ys.size), "n_mask": int(mask.sum())}


def region_mm2(img: np.ndarray, poly, area_mm2: float, d: dict, k_chroma: float, k_light: float,
               noise_chroma: float, noise_light: float) -> tuple[float, float, list | None]:
    """Largest connected anomalous region (mm^2), the anomalous fraction and the region's bounding box."""
    sC = max(d["sC"], noise_chroma)
    sL = max(d["sL"], noise_light)
    bad = (d["dC"] > k_chroma * sC) | (d["dL"] > k_light * sL)
    amap = np.zeros(img.shape[:2], np.uint8)
    amap[d["ys"][bad], d["xs"][bad]] = 1
    n, _, st, _ = cv2.connectedComponentsWithStats(amap, connectivity=8)
    face_px = float(cv2.contourArea(np.asarray(poly, np.float32)))
    mm2_per_px = area_mm2 / face_px if face_px > 0 else 0.0
    if n <= 1:
        return 0.0, float(bad.mean()), None
    k = 1 + int(np.argmax(st[1:, 4]))
    return float(st[k, 4]) * mm2_per_px, float(bad.mean()), [int(t) for t in st[k, :4]]


def score(img: np.ndarray, poly, area_mm2: float, model: dict) -> dict:
    """Largest connected anomalous region on the face (mm^2) and the face verdict under the model."""
    mask = face_mask(img.shape, poly, model["erode_frac"])
    d = deviations(img, mask)
    if d is None or d["n_valid"] < model["min_valid_frac"] * max(1, d["n_mask"]):
        return {"verdict": "UNCERTAIN", "reason": "too few usable pixels", "n_valid": 0 if d is None else d["n_valid"]}
    reg, frac, bbox = region_mm2(img, poly, area_mm2, d, model["k_chroma"], model["k_light"],
                                 model["noise_chroma"], model["noise_light"])
    out = {"region_mm2": round(reg, 2), "bad_frac": round(frac, 4), "sC": round(d["sC"], 3), "sL": round(d["sL"], 3),
           "base_lab": [round(x, 1) for x in d["base_lab"]], "n_valid": d["n_valid"]}
    out["verdict"] = "DEFECT" if reg > model["max_region_mm2"] else "CLEAN"
    if bbox is not None and out["verdict"] == "DEFECT":
        out["region_bbox"] = bbox
    return out
