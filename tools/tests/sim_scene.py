"""A synthetic wrist-camera view of the mat for offline tests: bricks (boxes with studs, per-face shading, optional
tape patches on any face), plates (discs) and a dark mat, rendered through the eye-in-hand camera model at any arm
pose. The image formation imitates the real camera where it matters to the inspection: light is added in linear
units (Lambertian faces, a glossy reflection of a lamp where the mirror geometry puts it, optional soft shadows),
then optics blur, vignetting, an exposure that differs per frame, shot and read noise, an sRGB tone curve, 8-bit
quantisation and a JPEG round trip (the camera streams MJPEG). Faces are filled far to near (painter's order), so
neighbours occlude faces like on the table. The renderer shares the camera model with the code under test: it checks
the inspection's logic, not the camera calibration. The lamp is a test value; the studs follow the brick spec."""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kin  # noqa: E402
import survey as sv  # noqa: E402

LIGHT = np.array([0.3, 0.2, 0.93]) / np.linalg.norm([0.3, 0.2, 0.93])  # diffuse key light direction
LAMP = np.array([120.0, 60.0, 900.0])  # point lamp seen in glossy faces (base frame, mm)
AMBIENT, DIFFUSE = 0.30, 0.70


def brick(xy, yaw, colour_bgr, size=(31.8, 31.8), tapes=()):
    """tapes: (face name, u0, u1, v0, v1, bgr) patches in face fractions (u along corner 0 -> 1, v along 0 -> 3 of the
    face, as survey.brick_faces orders them: on a side face v goes up)."""
    return {"xy": list(xy), "yaw": yaw, "bgr": colour_bgr, "size": size, "tapes": list(tapes)}


def to_linear(bgr) -> np.ndarray:
    v = np.asarray(bgr, float) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def to_display(E: np.ndarray) -> np.ndarray:
    E = np.clip(E, 0.0, 1.0)
    return np.where(E <= 0.0031308, 12.92 * E, 1.055 * np.power(E, 1 / 2.4) - 0.055) * 255.0


def _shade(n) -> float:
    return AMBIENT + DIFFUSE * max(0.0, float(np.asarray(n, float) @ LIGHT))


def _proj(q, P) -> np.ndarray:
    return sv.project_batch(q, np.asarray(P, float))


def _polys(q, b, height, cb):
    """(distance, polygon px, linear colour, kind) of everything one brick shows, for painter's order."""
    f = sv.brick_faces(b["xy"], b["yaw"], b["size"], height, kin.MAT_Z)
    col = to_linear(b["bgr"])
    out = []
    for name, face in f.items():
        p = np.array(face["pts"])
        ctr = p.mean(axis=0)
        if float(face["n"] @ (cb - ctr)) <= 0:
            continue
        sh = _shade(face["n"])
        d = float(np.linalg.norm(cb - ctr))
        out.append((d, _proj(q, p), col * sh, ("face", name, b, face)))
        o, eu, ev = p[0], p[1] - p[0], p[3] - p[0]
        for t in b["tapes"]:
            if t[0] == name:
                u0, u1, v0, v1 = t[1:5]
                quad = [o + eu * u + ev * v for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
                out.append((d - 1e-3, _proj(q, quad), to_linear(t[5]) * sh, ("tape", name, b, face)))
    top = f["top"]
    if float(top["n"] @ (cb - np.mean(top["pts"], axis=0))) > 0:  # studs, when the top faces the camera
        c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
        spec = sv.spec()
        pitch = spec["footprint_mm"] / spec["studs"][0]
        STUD_D, STUD_H = spec["stud_diameter_mm"], spec["stud_height_mm"]
        nx, ny = max(1, int(round(b["size"][0] / pitch))), max(1, int(round(b["size"][1] / pitch)))
        z0 = kin.MAT_Z + height
        ang = np.linspace(0, 2 * math.pi, 25)
        p = np.array(top["pts"])
        o, eu, ev = p[0], p[1] - p[0], p[3] - p[0]
        for i in range(nx):
            for j in range(ny):
                lx, ly = (i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch
                cx, cy = b["xy"][0] + c * lx - s * ly, b["xy"][1] + s * lx + c * ly
                ctr = np.array([cx, cy, z0 + STUD_H / 2])
                d = float(np.linalg.norm(cb - ctr)) - 1e-2
                rel = np.array([cx, cy, z0]) - o
                u, v = rel @ eu / (eu @ eu), rel @ ev / (ev @ ev)
                tape = [t for t in b["tapes"] if t[0] == "top" and t[1] <= u <= t[2] and t[3] <= v <= t[4]]
                base = to_linear(tape[-1][5]) if tape else col  # tape on the top covers the studs it spans
                r = STUD_D / 2
                for k in range(24):
                    a0, a1 = ang[k], ang[k + 1]
                    nrm = np.array([math.cos((a0 + a1) / 2), math.sin((a0 + a1) / 2), 0.0])
                    q4 = [np.array([cx + r * math.cos(a), cy + r * math.sin(a), z]) for a, z in
                          ((a0, z0), (a1, z0), (a1, z0 + STUD_H), (a0, z0 + STUD_H))]
                    toc = cb - np.mean(q4, axis=0)
                    if float(nrm @ toc) > 0:
                        out.append((d + 1e-3 * (1 - float(nrm @ toc) / np.linalg.norm(toc)), _proj(q, q4),
                                    base * _shade(nrm), ("stud", None, b, None)))
                disc = [np.array([cx + r * math.cos(a), cy + r * math.sin(a), z0 + STUD_H]) for a in ang[:-1]]
                out.append((d - 2e-3, _proj(q, disc), base * _shade([0, 0, 1]), ("stud_top", None, b, None)))
    return out


def _specular(q, face, cb, img_lin, mask_px, amp, radius_mm):
    """Add the lamp's mirror image on a glossy planar face where the mirror geometry puts it."""
    p = np.array(face["pts"])
    n = face["n"] / np.linalg.norm(face["n"])
    Lm = LAMP - 2 * float((LAMP - p[0]) @ n) * n
    den = float((Lm - cb) @ n)
    if abs(den) < 1e-9:
        return
    t = float((p[0] - cb) @ n) / den
    if not 0 < t < 1:
        return
    S = cb + t * (Lm - cb)
    o, eu, ev = p[0], p[1] - p[0], p[3] - p[0]
    u, v = float((S - o) @ eu / (eu @ eu)), float((S - o) @ ev / (ev @ ev))
    if not (0 <= u <= 1 and 0 <= v <= 1):
        return
    c = _proj(q, [S])[0]
    e = _proj(q, [S + radius_mm * eu / np.linalg.norm(eu)])[0]
    r = max(1.0, float(np.linalg.norm(e - c)))
    H, W = img_lin.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    blob = amp * np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * r * r))
    img_lin += (blob * mask_px)[..., None].astype(np.float32)


def render(q, bricks, plates=(), noise=2.5, seed=0, mat_bgr=(24, 22, 22), exposure_jitter=0.12, jpeg_quality=85,
           specular=1.6, shadow=None, blur=0.7):
    """One wrist frame at joints q. shadow: (u, v, radius_px, depth) soft dark disc (a shadow of the arm in this view)."""
    rng = np.random.default_rng(seed)
    H, W = sv.IMG_H, sv.IMG_W
    img = np.empty((H, W, 3), np.float32)
    img[:] = to_linear(mat_bgr)
    _, cb = sv.wc.camera_pose(q)
    for xy, r, bgr in plates:
        ring = [np.array([xy[0] + r * math.cos(a), xy[1] + r * math.sin(a), kin.MAT_Z])
                for a in np.linspace(0, 2 * math.pi, 90, endpoint=False)]
        cv2.fillPoly(img, [np.round(_proj(q, ring)).astype(np.int32)], to_linear(bgr).tolist())
    height = sv.spec()["height_mm"]
    items = []
    for b in bricks:
        items += _polys(q, b, height, cb)
    for d, uv, col, kind in sorted(items, key=lambda t: -t[0]):
        poly = np.round(uv).astype(np.int32)
        cv2.fillConvexPoly(img, poly, col.tolist())
        if kind[0] == "face" and specular > 0:
            m = np.zeros((H, W), np.float32)
            cv2.fillConvexPoly(m, poly, 1.0)
            _specular(q, kind[3], cb, img, m, specular, 2.5)
    if shadow is not None:
        u, v, r, depth = shadow
        yy, xx = np.mgrid[0:H, 0:W]
        dd = np.hypot(xx - u, yy - v)
        img *= (1.0 - depth * np.clip((r - dd) / (0.3 * r) + 0.5, 0, 1))[..., None].astype(np.float32)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    yy, xx = np.mgrid[0:H, 0:W]
    rr = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    img = img * ((1.0 - 0.2 * rr)[..., None] * (1.0 + exposure_jitter * rng.uniform(-1, 1)))
    k = (noise / 255.0) ** 2 / 0.05  # shot + read noise in linear units: mid grey gets about `noise` 8-bit levels
    img = img + rng.normal(0, 1, img.shape) * np.sqrt(np.maximum(img, 0) * k * 0.2 + k * 0.002)
    out = np.clip(np.round(to_display(img)), 0, 255).astype(np.uint8)
    if jpeg_quality:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out
