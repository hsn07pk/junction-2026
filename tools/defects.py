"""Surface defect test for one surveyed brick, from all views of the survey together. Nothing here knows a brick
colour or a tape colour.

Registration (register_view): the brick model, projected through the eye-in-hand camera model, is aligned to each
view first by the silhouette (shift), then by a small similarity transform (shift, rotation, scale) that brings the
model's edges onto the image's edges (truncated chamfer), kept only when it does not lower the silhouette overlap.

Sampling: every face a view shows well (cos to the camera at least min_cos) is resampled onto the face's own surface
grid (texels of texel_mm, each texel centre projected through the camera model and the registration). Texels near
the face's image boundary (registration residual), under the jaws or at the sensor ceiling are left out.

On that grid a clean DUPLO face is one plastic under smooth shading:
- colour: chromaticity (R, G) / (R + G + B) does not change with shading for a linear camera; the camera's tone curve
  bends it with brightness, so the body colour is a line c_b(l) against log intensity l, fitted robustly over every
  face of every view. Each view's colour balance is referred to the top face first (the same plastic under the same
  light in every view), so a camera that re-balances between views does not look like a defect;
- lightness: l minus a robust planar shading model of that face in that view.

Window means over squares of the smallest defect side (min_defect_mm) are compared with the body line and with the
shading plane, in units of their expected spread: the window-to-window spread on that face in that view (noise and
fine structure, robust to tape on less than half the face) combined with the model misfit of clean plastic (about a
level of quantisation and JPEG coding per flat face, in 8-bit levels), fitted by calibrate_defects.py on surveys of
clean bricks only. A window counts as
anomalous only when every view that tests it agrees (minimum over views): a highlight, a shadow or a registration
slip shows in one view, tape shows in all. The threshold follows from the number of windows tested and the accepted
false alarms per brick (a contrario detection), not from the bricks used for calibration."""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import warnings

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP  # noqa: E402
import survey as sv  # noqa: E402

CFG_F = os.path.join(APP, "config", "inspection.json")
MODEL_F = os.path.join(APP, "config", "defect_model.json")
CLIP = 250  # a channel at the sensor ceiling carries no colour (sensor property)
MODEL_KEYS = ("kappa_chroma", "kappa_light_top", "kappa_light_side")


def config() -> dict:
    with open(CFG_F) as f:
        return json.load(f)


def load_model(path: str = MODEL_F) -> dict | None:
    try:
        with open(path) as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    return m if all(isinstance(m.get(k), (int, float)) for k in MODEL_KEYS) else None


def _mad(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else 0.0


def camera_noise(img: np.ndarray, mask: np.ndarray) -> float:
    """Per-channel sensor noise (robust sigma, 8-bit units) on a uniform region of this image (the mat around the
    brick)."""
    v = img[mask].astype(np.float32)
    if len(v) < 50:
        return 1.0
    return max(0.5, float(np.median([_mad(v[:, c]) for c in range(3)])))


# ---------------------------------------------------------------- registration

def apply_T(uv: np.ndarray, c, T) -> np.ndarray:
    """Similarity transform about c: shift (dx, dy), rotation th (rad), scale s."""
    dx, dy, th, s = T
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return np.asarray(c) + s * (np.asarray(uv, float) - c) @ R.T + np.array([dx, dy])


def _edge_points(q, faces3d: dict, vis, step_px: float = 1.5) -> np.ndarray:
    """Image points along every edge of the visible faces (shared edges once), through the camera model."""
    seen, out = set(), []
    for f in vis:
        P = np.asarray(faces3d[f]["pts"], float)
        for a in range(4):
            p0, p1 = P[a], P[(a + 1) % 4]
            key = tuple(sorted((tuple(np.round(p0, 2)), tuple(np.round(p1, 2)))))
            if key in seen:
                continue
            seen.add(key)
            ends = sv.project_batch(q, np.stack([p0, p1]))
            n = max(8, int(np.linalg.norm(ends[1] - ends[0]) / step_px))
            t = (np.arange(n) + 0.5) / n
            out.append(sv.project_batch(q, p0[None] + t[:, None] * (p1 - p0)[None]))
    return np.concatenate(out) if out else np.zeros((0, 2))


def _edge_map(img: np.ndarray, roi: np.ndarray, noise: float) -> np.ndarray:
    """Edges in any colour channel inside roi; the hysteresis levels are at least 10 x the gradient noise (about the
    pixel noise after the 1 px blur and the 3x3 Sobel) and at the strongest decile of the roi's gradients."""
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    e = np.zeros(img.shape[:2], bool)
    for c in range(3):
        ch = np.ascontiguousarray(blur[..., c])
        mag = np.hypot(cv2.Sobel(ch, cv2.CV_32F, 1, 0), cv2.Sobel(ch, cv2.CV_32F, 0, 1))
        hi = max(float(np.percentile(mag[roi], 90)) if roi.any() else 0.0, 10.0 * noise)
        e |= cv2.Canny(ch, hi / 2.0, hi, L2gradient=True) > 0
    return e & roi


def edge_margin(dist: np.ndarray, hull: np.ndarray, noise: float, step_px: float = 3.0, reach: int = 8) -> float | None:
    """How far inside the brick's registered outline (px) the image stops being a mix of brick and mat: profiles of
    `dist` (a distance to the mat colour) are sampled along the outline's inward normals, each normalised between its
    own mat and brick levels; the margin covers optics
    blur, JPEG ringing and chroma subsampling, and registration slip at once, measured in this very view. Profiles without clear
    contrast (a neighbouring stud, a dark brick on the dark mat) are left out; None when none is left. The margin is
    where the median profile settles within 5 % of the brick level for good (JPEG ringing included)."""
    H, W = dist.shape
    pts, nrm = [], []
    m = len(hull)
    area2 = sum(hull[i][0] * hull[(i + 1) % m][1] - hull[(i + 1) % m][0] * hull[i][1] for i in range(m))
    for i in range(m):
        a, b = hull[i], hull[(i + 1) % m]
        e = b - a
        L = float(np.linalg.norm(e))
        if L < 2 * step_px:
            continue
        n_in = np.array([-e[1], e[0]]) / L * (1.0 if area2 > 0 else -1.0)  # inward normal of a CCW/CW polygon
        for t in np.arange(step_px, L - step_px, step_px):
            pts.append(a + e * t / L)
            nrm.append(n_in)
    if len(pts) < 10:
        return None
    pts, nrm = np.array(pts, np.float32), np.array(nrm, np.float32)
    offs = np.arange(-reach, reach + 1, 1.0, dtype=np.float32)
    P = pts[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    prof = cv2.remap(dist.astype(np.float32), P[..., 0], P[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    k = max(2, reach // 3)
    lo, hi = prof[:, :k].mean(axis=1), prof[:, -k:].mean(axis=1)
    keep = (hi - lo > 10.0 * max(noise, 0.5)) & (P[:, 0, 0] >= 0) & (P[:, -1, 0] < W) & (P[:, 0, 1] >= 0) & (P[:, -1, 1] < H)
    if keep.sum() < 10:
        return None
    norm = (prof[keep] - lo[keep, None]) / (hi[keep] - lo[keep])[:, None]
    med = np.median(norm, axis=0)
    tail = np.maximum.accumulate(np.abs(med - 1.0)[::-1])[::-1]  # worst deviation from each offset inward
    inside = np.nonzero((offs >= 0) & (tail <= 0.05))[0]
    return float(offs[inside[0]]) if inside.size else float(reach)


def register_view(img: np.ndarray, q, faces3d: dict, search_px: int, jaw=None) -> dict:
    """Align the projected brick model to one view. Returns the transform (applied to projected pixels), the
    silhouette overlap, the edge residual and what the defect test needs from this view (visible faces with their
    cosines, the mat around the brick and its noise)."""
    proj = sv.project_faces(q, faces3d, jaw)
    ts = sv.target_silhouette(img, proj, faces3d, search_px)
    if not ts.get("ok"):
        return {"ok": False, "why": ts.get("why")}
    vis = [f for f, r in proj.items() if r["cos"] > 0]
    pts = _edge_points(q, faces3d, vis)
    polys = {f: proj[f]["uv"] for f in vis}
    c = np.mean(np.concatenate(list(polys.values())), axis=0)
    sil = ts["sil"]
    H, W = img.shape[:2]

    def iou(T):
        m = np.zeros((H, W), bool)
        for uv in polys.values():
            m |= sv.poly_mask(img.shape, apply_T(uv, c, T))
        inter = float((m & sil).sum())
        return inter / (float(m.sum() + sil.sum()) - inter + 1e-6)

    T0 = (float(ts["shift"][0]), float(ts["shift"][1]), 0.0, 1.0)
    out = {"ok": True, "T": list(T0), "c": c.tolist(), "iou": round(iou(T0), 3), "at_limit": ts["at_limit"],
           "mat_mask": ts["mat_mask"], "noise": camera_noise(img, ts["mat_mask"]), "proj": proj, "refined": False}
    if len(pts) < 20:
        return out
    roi = cv2.dilate(sil.astype(np.uint8), np.ones((2 * 6 + 1,) * 2, np.uint8)) > 0
    dt = cv2.distanceTransform((~_edge_map(img, roi, out["noise"])).astype(np.uint8), cv2.DIST_L2, 3)
    tau = 5.0

    def cost(P):  # P: (..., N, 2) points -> mean truncated distance to the nearest image edge
        x = np.clip(np.round(P[..., 0]).astype(int), 0, W - 1)
        y = np.clip(np.round(P[..., 1]).astype(int), 0, H - 1)
        d = np.minimum(dt[y, x], tau)
        outside = (P[..., 0] < 0) | (P[..., 0] >= W) | (P[..., 1] < 0) | (P[..., 1] >= H)
        return np.where(outside, tau, d).mean(axis=-1)

    shifts = np.array([(a, b) for a in range(-4, 5) for b in range(-4, 5)], float)
    best = (float(cost(apply_T(pts, c, T0))), T0)
    for th in np.radians([-2.5, -1.25, 0.0, 1.25, 2.5]):
        for s in (0.97, 0.985, 1.0, 1.015, 1.03):
            base = apply_T(pts, c, (T0[0], T0[1], th, s))
            cs = cost(base[None] + shifts[:, None, :])
            i = int(np.argmin(cs))
            if cs[i] < best[0] - 1e-9:
                best = (float(cs[i]), (T0[0] + shifts[i, 0], T0[1] + shifts[i, 1], float(th), float(s)))
    T = list(best[1])
    for step in (1.0, 0.5, 0.25):
        improved = True
        while improved:
            improved = False
            for k, d in ((0, step), (0, -step), (1, step), (1, -step), (2, math.radians(0.5 * step)),
                         (2, -math.radians(0.5 * step)), (3, 0.005 * step), (3, -0.005 * step)):
                Tn = list(T)
                Tn[k] += d
                v = float(cost(apply_T(pts, c, Tn)))
                if v < best[0] - 1e-9:
                    best, T, improved = (v, tuple(Tn)), Tn, True
    c0 = float(cost(apply_T(pts, c, T0)))
    i1 = iou(T)
    if best[0] < c0 and i1 >= out["iou"] - 0.01:
        out.update(T=[float(v) for v in T], iou=round(i1, 3), refined=True)
    # each face's own small shift on top of the view's transform: the box seen in perspective from a slightly wrong
    # brick pose, or a camera model a pixel off near the image edge, is not exactly a similarity transform away
    sh = np.array([(a, b) for a in np.arange(-2.0, 2.01, 0.5) for b in np.arange(-2.0, 2.01, 0.5)])
    out["face_shift"] = {}
    for f in vis:
        pf = apply_T(_edge_points(q, {f: faces3d[f]}, [f], 1.0), c, out["T"])
        cs = cost(pf[None] + sh[:, None, :])
        i = int(np.argmin(cs))
        out["face_shift"][f] = [float(sh[i, 0]), float(sh[i, 1])] if cs[i] < cs[len(sh) // 2] - 0.05 else [0.0, 0.0]
    corners = np.unique(np.round(np.concatenate([np.asarray(faces3d[f]["pts"], float) for f in vis]), 3), axis=0)
    hull = cv2.convexHull(apply_T(sv.project_batch(q, corners), c, out["T"]).astype(np.float32))[:, 0, :]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    mm = ts["mat_mask"]
    if mm.sum() >= 50:
        med = np.median(lab[mm], axis=0)
        dL = np.abs(lab[..., 0] - med[0])
        dC = np.hypot(lab[..., 1] - med[1], lab[..., 2] - med[2])
        nL = max(_mad(lab[..., 0][mm]), 0.5)
        nC = max(math.hypot(_mad(lab[..., 1][mm]), _mad(lab[..., 2][mm])), 0.5)
        out["margin_L"] = edge_margin(dL, hull, nL)
        out["margin_C"] = edge_margin(dC, hull, nC)
    d = np.minimum(dt[np.clip(np.round(apply_T(pts, c, out["T"])[:, 1]).astype(int), 0, H - 1),
                      np.clip(np.round(apply_T(pts, c, out["T"])[:, 0]).astype(int), 0, W - 1)], tau)
    inl = d <= 2.0
    out.update(chamfer_px=round(float(d.mean()), 2), edge_inliers=round(float(inl.mean()), 3),
               resid_px=round(float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else tau, 2))
    return out


# ---------------------------------------------------------------- sampling the faces

def _seg_dist(p: np.ndarray, a, b) -> np.ndarray:
    """Distance (px) from points p (..., 2) to the segment a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def trim_edges(ok: np.ndarray, edge_d: np.ndarray, edge_id: np.ndarray, resid: np.ndarray, base: float,
               cap_px: float) -> np.ndarray:
    """Per face edge, drop 1 px bands next to that edge while a band's median residual still departs from the face's
    interior by more than 3 noise units of a band median (the neighbouring face or the mat mixing in where the view's
    outline margin was not enough, e.g. an edge the registration left a pixel off). At most cap_px beyond base, so a
    defect wider than that along an edge is still tested. resid: (..., k) residual vectors."""
    r = resid.reshape(resid.shape[:2] + (-1,))
    interior = ok & (edge_d >= base + cap_px)
    if interior.sum() < 50:
        interior = ok
    if interior.sum() < 10:
        return ok
    med0 = np.median(r[interior], axis=0)
    sig = max(float(np.mean([_mad(r[..., k][interior]) for k in range(r.shape[-1])])), 1e-6)
    out = ok.copy()
    for e in range(4):
        sel = ok & (edge_id == e)
        for d0 in np.arange(base, base + cap_px, 1.0):
            band = sel & (edge_d >= d0) & (edge_d < d0 + 1.0)
            n = int(band.sum())
            if n < 5:
                continue
            if float(np.linalg.norm(np.median(r[band], axis=0) - med0)) <= 3.0 * 1.25 * sig / math.sqrt(n):
                break
            out &= ~band
    return out


HIGHLIGHT_MAX_PX = 1500  # a clipped blob up to this size is a glossy highlight; larger ones are a bright face itself


def clipped_halo(img: np.ndarray) -> np.ndarray:
    """Glossy highlights and the halo around each: blobs where every channel is at the sensor ceiling (the lamp seen
    in the plastic, white whatever the brick's colour), small enough to be a highlight; a highlight bright enough to
    clip its core spills over about as far again as its core radius. A face that is itself at the ceiling over a large
    area (a saturated colour channel, or an over-exposed white brick) stays data: it is uniform, and a mark on it
    still differs from it."""
    clip_all = (img.min(axis=2) >= CLIP).astype(np.uint8)
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats(clip_all, connectivity=8)
    small = np.isin(lab0, [i for i in range(1, n0) if st0[i, 4] <= HIGHLIGHT_MAX_PX])
    clip = small.astype(np.uint8)
    out = cv2.dilate(clip, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(clip, connectivity=8)
    H, W = clip.shape
    for i in range(1, n):
        k = max(1, int(math.ceil(2.0 * math.sqrt(st[i, 4] / math.pi))))
        x0, y0, w, h = st[i, :4]
        xa, ya, xb, yb = max(0, x0 - k), max(0, y0 - k), min(W, x0 + w + k), min(H, y0 + h + k)
        sub = (lab[ya:yb, xa:xb] == i).astype(np.uint8)
        out[ya:yb, xa:xb] |= cv2.dilate(sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return out > 0


def face_grid(face: dict, texel_mm: float) -> tuple[np.ndarray, int, int]:
    """Texel centres (nv x nu x 3, base frame mm) of a face: u along corner 0 -> 1, v along corner 0 -> 3."""
    P = np.asarray(face["pts"], float)
    o, eu, ev = P[0], P[1] - P[0], P[3] - P[0]
    nu = max(2, int(round(np.linalg.norm(eu) / texel_mm)))
    nv = max(2, int(round(np.linalg.norm(ev) / texel_mm)))
    uu, vv = (np.arange(nu) + 0.5) / nu, (np.arange(nv) + 0.5) / nv
    return o + uu[None, :, None] * eu + vv[:, None, None] * ev, nu, nv


def sample_face(img: np.ndarray, q, face: dict, reg: dict, texel_mm: float, jaw=None) -> dict:
    G, nu, nv = face_grid(face, texel_mm)
    fs = np.asarray((reg.get("face_shift") or {}).get(face.get("name"), (0.0, 0.0)), float)
    uv = (apply_T(sv.project_batch(q, G.reshape(-1, 3)), reg["c"], reg["T"]) + fs).reshape(nv, nu, 2).astype(np.float32)
    tex = cv2.remap(img, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0).astype(np.float32)
    H, W = img.shape[:2]
    xi = np.clip(np.round(uv[..., 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(uv[..., 1]).astype(int), 0, H - 1)
    inside = (uv[..., 0] >= 1) & (uv[..., 0] <= W - 2) & (uv[..., 1] >= 1) & (uv[..., 1] <= H - 2)
    poly = apply_T(sv.project_batch(q, np.asarray(face["pts"], float)), reg["c"], reg["T"]) + fs
    ed = np.stack([_seg_dist(uv, poly[k], poly[(k + 1) % 4]) for k in range(4)], -1)  # px to each face edge
    clip = clipped_halo(img)
    valid = inside & ~clip[yi, xi]
    if jaw is not None and jaw.shape == clip.shape:
        valid &= ~jaw[yi, xi]
    px_per_mm = math.sqrt(max(1.0, cv2.contourArea(poly.astype(np.float32))) / max(1e-6, face["area_mm2"]))
    return {"tex": tex, "valid": valid, "edge_d": ed.min(axis=-1), "edge_id": ed.argmin(axis=-1), "nu": nu, "nv": nv,
            "px_per_mm": px_per_mm}


def _lag_sigma(m: np.ndarray, ok: np.ndarray, w: int) -> float | None:
    """Robust sigma of window means, from differences between windows one window apart (both testable). A defect larger
    than a window leaves most such pairs inside it or outside it, so this measures noise and fine structure at the
    window scale, not the defect itself."""
    d = []
    if m.shape[0] > w:
        d.append((m[w:] - m[:-w])[ok[w:] & ok[:-w]])
    if m.shape[1] > w:
        d.append((m[:, w:] - m[:, :-w])[ok[:, w:] & ok[:, :-w]])
    d = np.concatenate(d) if d else np.zeros(0)
    return _mad(d) / math.sqrt(2.0) if d.size >= 10 else None


def stud_mask(q, top: dict, reg: dict, nu: int, nv: int, sp: dict, margin_px: float, px_per_mm: float) -> np.ndarray:
    """Texels of the top face whose pixel does not show the flat top plane in this view: inside a stud's base, or
    behind a stud (the camera ray to the texel meets the stud's wall or top first), widened by the registration
    margin. The flat-plane shading model does not hold there; the colour test still uses them (same plastic)."""
    P = np.asarray(top["pts"], float)
    o, eu, ev = P[0], P[1] - P[0], P[3] - P[0]
    uu, vv = (np.arange(nu) + 0.5) / nu, (np.arange(nv) + 0.5) / nv
    X = o + uu[None, :, None] * eu + vv[:, None, None] * ev
    _, cb = sv.wc.camera_pose(q)
    d = X - cb
    z_top = float(P[0][2])
    r, h = sp["stud_diameter_mm"] / 2.0, sp["stud_height_mm"]
    nx, ny = sp["studs"]
    hidden = np.zeros((nv, nu), bool)
    for i in range(nx):
        for j in range(ny):
            c = o + eu * (i + 0.5) / nx + ev * (j + 0.5) / ny
            hidden |= np.hypot(X[..., 0] - c[0], X[..., 1] - c[1]) <= r
            t = (z_top + h - cb[2]) / d[..., 2]  # the stud's top disc
            px, py = cb[0] + t * d[..., 0] - c[0], cb[1] + t * d[..., 1] - c[1]
            hidden |= (t > 0) & (t < 1) & (px * px + py * py <= r * r)
            ox, oy = cb[0] - c[0], cb[1] - c[1]  # the stud's wall: |(o + t d)_xy| = r with z in [z_top, z_top + h]
            A = d[..., 0] ** 2 + d[..., 1] ** 2
            B = 2 * (ox * d[..., 0] + oy * d[..., 1])
            C = ox * ox + oy * oy - r * r
            disc = B * B - 4 * A * C
            ok = disc >= 0
            sq = np.sqrt(np.where(ok, disc, 0.0))
            for root in ((-B - sq) / (2 * A + 1e-12), (-B + sq) / (2 * A + 1e-12)):
                z = cb[2] + root * d[..., 2]
                hidden |= ok & (root > 0) & (root < 1) & (z >= z_top) & (z <= z_top + h)
    k = int(math.ceil(margin_px / max(1e-6, px_per_mm) / (np.linalg.norm(eu) / nu)))
    if k > 0:
        hidden = cv2.dilate(hidden.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    return hidden


def _box(x: np.ndarray, w: int) -> np.ndarray:
    return cv2.boxFilter(x.astype(np.float32), -1, (w, w), normalize=False, borderType=cv2.BORDER_CONSTANT)


def _plane(valid: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Robust planar fit y ~ a + b u + c v over the valid texels: start from the texels within 3 robust sigmas of the
    median (a defect on less than half the face is left out from the start), then refit on the texels within 2.5 robust
    sigmas of the current plane (a shading gradient is taken back in as the plane follows it)."""
    vs, us = np.nonzero(valid)
    A = np.stack([np.ones(us.size), us - us.mean(), vs - vs.mean()], 1)
    t = y[valid]
    med, s0 = float(np.median(t)), max(_mad(t), 1e-6)
    keep = np.abs(t - med) <= 3.0 * s0
    c = np.array([med, 0.0, 0.0])
    for _ in range(5):
        if keep.sum() < 10:
            break
        c, *_ = np.linalg.lstsq(A[keep], t[keep], rcond=None)
        r = t - A @ c
        s = _mad(r[keep])
        if s <= 0:
            break
        keep = np.abs(r) <= 2.5 * s
    vv, uu = np.mgrid[0:valid.shape[0], 0:valid.shape[1]]
    return c[0] + c[1] * (uu - us.mean()) + c[2] * (vv - vs.mean())


def _body_line(ell: np.ndarray, rg: np.ndarray, slope_floor: float) -> dict:
    """Robust line rg = a + b (l - l0) (Tukey biweight IRLS). The slope is damped as if l spread at least slope_floor,
    so a brick seen at nearly one brightness gets a flat line."""
    l0 = float(np.median(ell))
    x = ell - l0
    a, b = np.median(rg, axis=0), np.zeros(2)
    for _ in range(10):
        res = rg - a - np.outer(x, b)
        s = np.maximum(1.4826 * np.median(np.abs(res - np.median(res, axis=0)), axis=0), 1e-4)
        u = np.abs(res) / (4.685 * s)
        w = np.prod(np.where(u < 1, (1 - u * u) ** 2, 0.0), axis=1)
        sw = w.sum()
        if sw <= 0:
            break
        sx, sxx = (w * x).sum(), (w * x * x).sum() + sw * slope_floor ** 2
        for k in range(2):
            sy, sxy = (w * rg[:, k]).sum(), (w * x * rg[:, k]).sum()
            det = sw * sxx - sx * sx
            b[k] = (sw * sxy - sx * sy) / det
            a[k] = (sy - b[k] * sx) / sw
    return {"l0": l0, "a": [float(v) for v in a], "b": [float(v) for v in b]}


# ---------------------------------------------------------------- observe (images -> per-window deviations)

def observe(views: list, faces3d: dict, cfg: dict, search_px: int, jaw=None, sp: dict | None = None) -> dict:
    """views: [{"img", "q", "name"}]. Registers every view, samples every face it shows well, fits the body colour
    and returns per face and view the window-mean deviations and their spreads (judge() turns them into verdicts)."""
    sp = sp or sv.spec()
    tm, w = cfg["texel_mm"], max(2, int(round(cfg["min_defect_mm"] / cfg["texel_mm"])))
    regs, samples = {}, []
    for v in views:
        if v["img"].shape[:2] != (sv.IMG_H, sv.IMG_W):  # the camera model holds for the frame size it was fitted at
            regs[v["name"]] = {"ok": False, "why": f"frame {v['img'].shape[1]}x{v['img'].shape[0]}, camera model "
                                                   f"{sv.IMG_W}x{sv.IMG_H}"}
            continue
        reg = register_view(v["img"], v["q"], faces3d, search_px, jaw)
        regs[v["name"]] = {k: reg.get(k) for k in ("ok", "why", "T", "iou", "refined", "chamfer_px", "edge_inliers",
                                                   "resid_px", "at_limit", "noise", "margin_L", "margin_C",
                                                   "face_shift")}
        if not reg.get("ok") or reg["iou"] < cfg["min_iou"] or reg["at_limit"] or \
                reg.get("edge_inliers", 1.0) < cfg["min_edge_inliers"]:
            continue
        resid = float(reg.get("resid_px", 2.0))
        mL = reg.get("margin_L")
        mL = (mL if mL is not None else cfg["edge_px"]) + resid
        mC = reg.get("margin_C")
        mC = max(mC if mC is not None else cfg["edge_px"], mL - resid) + resid
        margin = mL
        for f, pr in reg["proj"].items():
            if pr["cos"] < cfg["min_cos"] or not pr["inside"]:
                continue
            s = sample_face(v["img"], v["q"], dict(faces3d[f], name=f), reg, tm, jaw)
            s["valid_c"] = s["valid"] & (s["edge_d"] >= mC)
            s["valid"] &= s["edge_d"] >= mL
            s["mL"], s["mC"] = mL, mC
            if s["valid"].sum() < w * w:
                continue
            s.update(face=f, view=v["name"], cos=pr["cos"], noise=reg["noise"])
            s["flat"] = ~stud_mask(v["q"], faces3d[f], reg, s["nu"], s["nv"], sp, margin, s["px_per_mm"]) \
                if f == "top" else np.ones_like(s["valid"])
            samples.append(s)
    # colour balance per view, referred to the top face (same plastic, same light in every view)
    tops = {s["view"]: s for s in samples if s["face"] == "top"}
    ref, meds = None, {}
    if tops:
        for vn, s in tops.items():
            t = s["tex"][s["valid"]]
            t = t[t.sum(axis=1) > 0]
            if len(t) > 50:
                m = np.median(t / t.sum(axis=1, keepdims=True), axis=0)
                meds[vn] = m
        if meds:
            ref = np.median(np.stack(list(meds.values())), axis=0)
    for s in samples:
        gain = (ref / meds[s["view"]]) if ref is not None and s["view"] in meds else np.ones(3)
        t = s["tex"] * gain[None, None, :]
        I = t.sum(axis=2)
        dark = cfg["chroma_dark_snr"] * math.sqrt(3.0) * s["noise"]
        s["okc"] = s["valid_c"] & (I > dark)
        s["ell"] = np.log(np.maximum(I, 1.0))
        s["rg"] = np.stack([t[..., 2] / np.maximum(I, 1e-6), t[..., 1] / np.maximum(I, 1e-6)], -1)
    pts_l = [s["ell"][s["okc"]] for s in samples]
    pts_c = [s["rg"][s["okc"]] for s in samples]
    if not samples or sum(len(p) for p in pts_l) < 100:
        return {"ok": False, "why": "no face sampled well enough", "regs": regs, "w": w}
    ell, rg = np.concatenate(pts_l), np.concatenate(pts_c)
    step = max(1, len(ell) // 40000)
    body = _body_line(ell[::step], rg[::step], cfg["body_slope_floor"])
    faces = {}
    min_n = cfg["min_window_valid"] * w * w
    for s in samples:
        x = s["ell"] - body["l0"]
        e = s["rg"] - (np.array(body["a"])[None, None] + x[..., None] * np.array(body["b"])[None, None])
        cap = cfg["edge_trim_mm"] * s["px_per_mm"]
        s["okc"] = trim_edges(s["okc"], s["edge_d"], s["edge_id"], e, s["mC"], cap)
        nC = _box(s["okc"], w)
        winC = nC >= min_n
        mc = np.stack([_box(np.where(s["okc"], e[..., k], 0.0), w) for k in range(2)], -1) / np.maximum(nC, 1)[..., None]
        okL = s["valid"] & s["flat"]
        if okL.sum() < w * w:
            continue
        okL = trim_edges(okL, s["edge_d"], s["edge_id"], s["ell"] - _plane(okL, s["ell"]), s["mL"], cap)
        if okL.sum() < w * w:
            continue
        eL = np.where(okL, s["ell"] - _plane(okL, s["ell"]), 0.0)
        nL = _box(okL, w)
        winL = nL >= min_n
        mL = _box(eL, w) / np.maximum(nL, 1)
        if winL.sum() < cfg["min_windows"]:
            continue
        sC = None
        if winC.sum() >= cfg["min_windows"]:
            sk = [_lag_sigma(mc[..., k], winC, w) for k in range(2)]
            sC = max(float(np.mean(sk)), 1e-6) if None not in sk else None
        dl = mL - (float(np.median(mL[winL])))
        sL = _lag_sigma(dl, winL, w)
        I = np.exp(s["ell"])
        faces.setdefault(s["face"], []).append({
            "view": s["view"], "cos": round(float(s["cos"]), 3), "px_per_mm": round(s["px_per_mm"], 2),
            "dc": np.where(winC, np.hypot(mc[..., 0], mc[..., 1]), np.nan) if sC is not None else None,
            "sC": sC, "dl": np.where(winL, np.abs(dl), np.nan), "sL": max(sL if sL is not None else _mad(dl[winL]), 1e-6),
            "Ic": np.where(winC, _box(np.where(s["okc"], I, 0.0), w) / np.maximum(nC, 1), np.nan),
            "Il": np.where(winL, _box(np.where(okL, I, 0.0), w) / np.maximum(nL, 1), np.nan),
            "shape": (s["nv"], s["nu"])})
    n_win = sum(max(1.0, (s["nv"] - w + 1) * (s["nu"] - w + 1) / float(w * w))
                for s in {s["face"]: s for s in samples}.values())
    n_tests = 2.0 * max(n_win, 1.0)
    eps = cfg["false_alarms_per_brick"]
    z_c = math.sqrt(2.0 * math.log(n_tests / eps))  # 2-D (Rayleigh) tail of a window's colour deviation
    z_l = statistics.NormalDist().inv_cdf(1.0 - eps / (2.0 * n_tests))  # two-sided Gaussian tail of its lightness
    return {"ok": True, "faces": faces, "body": body, "regs": regs, "w": w, "z_star": [z_c, z_l],
            "texel_mm": tm, "n_tests": n_tests}


# ---------------------------------------------------------------- judge (deviations -> verdicts)

def fused(obs: dict, face: str, model: dict) -> dict:
    """Per texel window of one face: the smallest standardised deviation over the views that test it (colour and
    lightness). A window's expected spread is its noise scale combined with the model misfit of clean plastic,
    kappa / I with kappa in 8-bit levels and I = R + G + B of the window: an 8-bit, JPEG-coded frame leaves each flat
    face's colour and brightness a level or so off, which is a larger chromaticity and log-intensity error on a darker
    face."""
    st = obs["faces"].get(face) or []
    k_c = float(model.get("kappa_chroma", 0.0))
    k_l = float(model.get("kappa_light_top" if face == "top" else "kappa_light_side", 0.0))
    zc, zl = [], []
    for v in st:
        if v["dc"] is not None:
            zc.append(v["dc"] / np.sqrt(v["sC"] ** 2 + (k_c / np.maximum(v["Ic"], 1.0)) ** 2))
        zl.append(v["dl"] / np.sqrt(v["sL"] ** 2 + (k_l / np.maximum(v["Il"], 1.0)) ** 2))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fc = np.nanmin(np.stack(zc), axis=0) if zc else None
        fl = np.nanmin(np.stack(zl), axis=0) if zl else None
    return {"zc": fc, "zl": fl, "n_views": len(st)}


def judge(obs: dict, model: dict | None, cfg: dict) -> dict:
    """Verdict per face: DEFECT when a window's colour or lightness departs beyond the a-contrario threshold in every
    view that tests it; CLEAN when the face was tested over at least min_face_coverage of its windows and none
    departs; UNCERTAIN otherwise (not seen well enough, or no model fitted yet)."""
    out = {}
    if not obs.get("ok"):
        return {f: {"verdict": "UNCERTAIN", "reason": obs.get("why", "no observation")} for f in sv.FACES}
    warnings.simplefilter("ignore", RuntimeWarning)
    z_c, z_l = obs["z_star"]
    w, tm = obs["w"], obs["texel_mm"]
    for f in sv.FACES:
        st = obs["faces"].get(f)
        if not st:
            out[f] = {"verdict": "UNCERTAIN", "reason": "face not seen in any usable view"}
            continue
        fz = fused(obs, f, model or {})
        nv, nu = st[0]["shape"]
        inner = np.zeros((nv, nu), bool)  # the face's interior: edge bands are left out of the coverage (see edge_trim)
        h = w // 2 + int(math.ceil((cfg["edge_trim_mm"] + cfg["coverage_band_mm"]) / tm))
        if nv > 2 * h and nu > 2 * h:
            inner[h:nv - h, h:nu - h] = True
        else:
            inner[:] = True
        tested = np.zeros((nv, nu), bool)  # tested by colour or by lightness in at least one view
        for z in (fz["zc"], fz["zl"]):
            if z is not None:
                tested |= ~np.isnan(z)
        cov = float((tested & inner).sum()) / max(1, int(inner.sum()))
        with np.errstate(all="ignore"):
            zc_max = float(np.nanmax(fz["zc"])) if fz["zc"] is not None and np.isfinite(fz["zc"]).any() else 0.0
            zl_max = float(np.nanmax(fz["zl"])) if fz["zl"] is not None and np.isfinite(fz["zl"]).any() else 0.0
            bad = np.zeros((nv, nu), bool)
            if fz["zc"] is not None:
                bad |= np.nan_to_num(fz["zc"], nan=0.0) >= z_c
            if fz["zl"] is not None:
                bad |= np.nan_to_num(fz["zl"], nan=0.0) >= z_l
        r = {"coverage": round(cov, 3), "n_views": fz["n_views"], "zc_max": round(zc_max, 2), "zl_max": round(zl_max, 2),
             "z_star": [round(z_c, 2), round(z_l, 2)], "views": [v["view"] for v in st]}
        if bad.any():
            vs, us = np.nonzero(bad)
            r.update(verdict="DEFECT" if model else "UNCERTAIN",
                     region_mm2=round(float(bad.sum()) * tm * tm + (w * tm) ** 2 - tm * tm, 1),
                     region_uv_mm=[round(float(us.min()) * tm, 1), round(float(vs.min()) * tm, 1),
                                   round(float(us.max() + 1) * tm, 1), round(float(vs.max() + 1) * tm, 1)],
                     by="colour" if (fz["zc"] is not None and np.nanmax(np.where(bad, fz["zc"], np.nan)) >= z_c)
                     else "lightness")
        elif cov < cfg["min_face_coverage"]:
            r.update(verdict="UNCERTAIN", reason=f"only {cov:.0%} of the face tested")
        else:
            r["verdict"] = "CLEAN" if model else "UNCERTAIN"
        if not model:
            r["reason"] = "no defect model fitted yet (calibrate_defects.py)"
        out[f] = r
    return out


def inspect(views: list, faces3d: dict, model: dict | None, cfg: dict | None = None, search_px: int = 24,
            jaw=None, sp: dict | None = None) -> tuple[dict, dict]:
    """Observe and judge. Returns ({face: verdict record}, summary for the episode log)."""
    cfg = cfg or config()
    obs = observe(views, faces3d, cfg, search_px, jaw, sp)
    res = judge(obs, model, cfg)
    summ = {"regs": obs.get("regs"), "body": obs.get("body"), "z_star": obs.get("z_star"), "ok": obs.get("ok"),
            "why": obs.get("why"), "model": None if model is None else {k: model.get(k) for k in MODEL_KEYS}}
    return res, summ
