"""Underside inspection: show the held brick's bottom to the fixed close camera and judge it.

The arm turns the gripper so that its approach axis (along which the held brick's bottom faces) points at the close
camera (kin.ik_dir; on this five-axis arm the camera's azimuth from the pan axis fixes the plane of the pose), at the
reachable pose that shows the bottom largest. In the close frame the bottom is found near where the camera model
(closecam.py) puts it: the region that differs from the local background, squared up by its minimum-area rectangle.
A DUPLO bottom is open: a dark cavity inside a lit rim. Tape over the opening turns the cavity into a surface lit like
the rim (the cavity-to-rim brightness ratio rises toward 1) or, dark tape, darker than any shaded cavity, and tape of
another colour also shows as a chroma change from the rim. Both limits come from clean bricks only (calibrate: --fit), never from the tapes.

Off unless config/underside.json enables it and both the close-camera model and the underside model are fitted; then
the bottom is a required face. Usage: python -I -B underside.py --fit [LABELS.json] [--write]"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP, LOG_ROOT  # noqa: E402
import closecam as cc  # noqa: E402
import kin  # noqa: E402
import survey as sv  # noqa: E402

CFG_F = os.path.join(APP, "config", "underside.json")
MODEL_F = os.path.join(APP, "config", "underside_model.json")
TEX = 64  # the bottom face is resampled to TEX x TEX for the test


def config() -> dict:
    return json.load(open(CFG_F))


def model() -> dict | None:
    try:
        return json.load(open(MODEL_F))
    except (OSError, ValueError):
        return None


def enabled() -> bool:
    try:
        return bool(config().get("enabled")) and cc.load() is not None and model() is not None
    except (OSError, ValueError):
        return False


def bottom(q, sp: dict, tip_mm: float):
    """Centre, outward normal and corners (base frame, mm) of the held brick's bottom face at joints q: the brick was
    gripped with the jaw tips tip_mm above its bottom, its faces along the jaws."""
    p, R = kin.fk(q)
    a, ex, ey = R[:, 2], R[:, 0], R[:, 1]
    c = np.asarray(p, float) + tip_mm * a
    h = sp["footprint_mm"] / 2.0
    corners = np.array([c + sx * h * ex + sy * h * ey for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return c, a, corners


def plan(M: dict, q_ref, sp: dict, cfg: dict, tip_mm: float):
    """The reachable pose whose held brick's bottom faces the camera and shows largest, inside the image."""
    C = np.asarray(M["centre_mm"], float)
    az = math.atan2(C[1] - kin.PAN_XY_MM[1], C[0] - kin.PAN_XY_MM[0])
    best = None
    for rho in cfg["radius_mm"]:
        for dz in cfg["height_mm"]:
            P = np.array([kin.PAN_XY_MM[0] + rho * math.cos(az), kin.PAN_XY_MM[1] + rho * math.sin(az), kin.MAT_Z + dz])
            a = C - P
            a /= np.linalg.norm(a)
            for _ in range(2):  # aim from the bottom's centre, not the TCP
                a = C - (P + tip_mm * a)
                a /= np.linalg.norm(a)
            if math.degrees(math.asin(max(-1.0, min(1.0, a[2])))) > cfg["max_up_deg"]:
                continue
            q = kin.ik_dir(P, a, q_ref)
            if q is None:
                continue
            c, n, corners = bottom(q, sp, tip_mm)
            if not in_view(M, corners):
                continue
            cosv = float(n @ (C - c) / np.linalg.norm(C - c))
            uv = cc.project(M, corners)
            size = math.sqrt(max(cv2.contourArea(uv.astype(np.float32)), 0.0))
            if cosv >= cfg["min_cos"] and (best is None or size > best[1]["size_px"]):
                best = (q, {"tcp_mm": [round(float(v), 1) for v in P], "cos": round(cosv, 3), "size_px": round(size, 1)})
    return best


def in_view(M: dict, P: np.ndarray, margin: int = 20) -> bool:
    W, H = M["image_size"]
    uv = cc.project(M, P)
    return bool(np.all(cc.in_front(M, P)) and np.all((uv[:, 0] >= margin) & (uv[:, 0] <= W - margin) &
                                                     (uv[:, 1] >= margin) & (uv[:, 1] <= H - margin)))


def find_bottom(img: np.ndarray, pred: np.ndarray) -> dict:
    """The bottom face near the predicted quad: pixels that differ from the background around it (Otsu on the Lab
    distance to the background's median), the component that overlaps the prediction most, its minimum-area rectangle."""
    size = math.sqrt(max(cv2.contourArea(pred.astype(np.float32)), 1.0))
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.round(pred).astype(np.int32), 1)
    roi = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(1.2 * size) | 1,) * 2)) > 0
    band = roi & ~(cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.8 * size) | 1,) * 2)) > 0)
    if band.sum() < 50:
        return {"ok": False, "why": "no background around the predicted bottom"}
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.linalg.norm(lab - np.median(lab[band], axis=0), axis=2)
    v = d[roi]
    t, _ = cv2.threshold(np.clip(v / max(float(v.max()), 1e-6) * 255, 0, 255).astype(np.uint8).reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = np.zeros(img.shape[:2], np.uint8)
    fg[roi] = (v > t / 255.0 * float(v.max())).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labm, st, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return {"ok": False, "why": "nothing but background where the bottom should be"}
    k = 1 + int(np.argmax([((labm == i) & (m > 0)).sum() for i in range(1, n)]))
    comp = (labm == k).astype(np.uint8)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cs, -1, 1, -1)  # the cavity is part of the face
    rect = cv2.minAreaRect(max(cs, key=cv2.contourArea))
    quad = cv2.boxPoints(rect)
    inter = float(((filled > 0) & (m > 0)).sum())
    iou = inter / float((filled > 0).sum() + m.sum() - inter + 1e-6)
    return {"ok": True, "quad": quad, "iou": round(iou, 3), "rect_wh": [round(float(x), 1) for x in rect[1]]}


def measure(img: np.ndarray, quad: np.ndarray) -> dict:
    """Cavity-to-rim brightness ratio and the chroma difference of the cavity from the rim (Lab a*, b*), on the bottom
    resampled square: rim = outer band (0.80-0.95 of the half side), cavity = the middle (within 0.5)."""
    dst = np.array([[0, 0], [TEX - 1, 0], [TEX - 1, TEX - 1], [0, TEX - 1]], np.float32)
    H = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst)
    tex = cv2.warpPerspective(img, H, (TEX, TEX), flags=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(tex, cv2.COLOR_BGR2LAB).astype(np.float32)
    g = np.abs(np.linspace(-1, 1, TEX))
    r = np.maximum(g[None, :], g[:, None])
    rim, mid = (r >= 0.80) & (r <= 0.95), r <= 0.5
    L_rim, L_mid = float(np.median(lab[..., 0][rim])), float(np.median(lab[..., 0][mid]))
    ab_rim, ab_mid = np.median(lab[..., 1:][rim], axis=0), np.median(lab[..., 1:][mid], axis=0)
    spread = float(np.mean([1.4826 * np.median(np.abs(lab[..., c][rim] - np.median(lab[..., c][rim]))) for c in (1, 2)]))
    return {"cavity_ratio": round(L_mid / max(L_rim, 1e-6), 3), "chroma_diff": round(float(np.linalg.norm(ab_mid - ab_rim)), 2),
            "rim_chroma_spread": round(spread, 2), "rim_L": round(L_rim, 1)}


def verdict(m: dict, um: dict | None) -> str:
    if um is None:
        return "UNCERTAIN"
    bad = not (um["cavity_ratio_min"] < m["cavity_ratio"] < um["cavity_ratio_max"]) or \
        m["chroma_diff"] >= um["chroma_diff_max"]
    return "DEFECT" if bad else "CLEAN"


def inspect(ga, sp: dict, log=print) -> dict:
    """In the loop, with a brick held and the arm lifted: show its bottom to the close camera, judge, come back up.
    ga: the motion/camera client (grasp_auto)."""
    cfg, M, um = config(), cc.load(), model()
    q0 = ga.state()["present"][:5]
    tip = kin.cell.get("tip_final_mm")
    pl = plan(M, q0, sp, cfg, tip) if M else None
    if pl is None:
        return {"verdict": "UNCERTAIN", "reason": "no reachable pose shows the bottom to the close camera"}
    q, info = pl
    p0, _ = kin.fk(q0)
    lift = kin.MAT_Z + min(cfg["height_mm"]) - float(p0[2])
    if lift > 0:  # straight up first: the turn never sweeps the brick low
        import pick_any as pa
        q0 = pa.move_rel(q0, 0.0, [0, 0, lift], speed=cfg["speed"]) or q0
    ga.move(list(q) + [0.0], speed=cfg["speed"], comp=True)
    img = ga.fresh("close", time.time() + 0.3)
    qa = ga.state()["present"][:5]
    ga.move(list(q0) + [0.0], speed=cfg["speed"], comp=True)
    if img is None:
        return {"verdict": "UNCERTAIN", "reason": "no close frame", "pose": info}
    _, _, corners = bottom(qa, sp, tip)
    found = find_bottom(img, cc.project(M, corners))
    if not found["ok"] or found["iou"] < cfg["min_iou"]:
        return {"verdict": "UNCERTAIN", "reason": f"bottom not found ({found.get('why') or found.get('iou')})", "pose": info}
    m = measure(img, found["quad"])
    out = dict(m, verdict=verdict(m, um), pose=info, iou=found["iou"], q=[round(float(v), 2) for v in qa])
    out["_img"] = img
    log(f"underside: {out['verdict']} (cavity/rim {m['cavity_ratio']}, chroma {m['chroma_diff']})")
    return out


def fit(records: list, cfg: dict) -> dict | None:
    """Limits from clean bottoms only. The cavity's brightness relative to the rim is set by the bottom's geometry and
    the light, not by the plastic's colour: tape over the opening makes it a lit surface (ratio toward 1) or, dark tape,
    darker than any shaded cavity (toward 0). The limits sit halfway between the clean bottoms' range and those ends;
    the chroma limit is the design margin times the largest clean difference (at least the rim's own spread)."""
    clean = [r for r in records if r["truth"] == "GOOD"]
    if not clean:
        return None
    ratios = [r["m"]["cavity_ratio"] for r in clean]
    cmax = max(max(r["m"]["chroma_diff"] for r in clean), max(r["m"]["rim_chroma_spread"] for r in clean))
    return {"cavity_ratio_min": round(0.5 * min(ratios), 3), "cavity_ratio_max": round(0.5 * (max(ratios) + 1.0), 3),
            "chroma_diff_max": round(cfg["margin"] * cmax, 2), "clean_cavity_ratio": [min(ratios), max(ratios)],
            "clean_chroma_diff_max": cmax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("labels", nargs="?", default=os.path.join(LOG_ROOT, "sort", "labels.json"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if not a.fit:
        sys.exit(__doc__)
    cfg = config()
    lab = json.load(open(a.labels))
    recs = []
    for ep, bid in sorted(lab.get("episodes", {}).items()):
        try:
            d = json.load(open(os.path.join(LOG_ROOT, "sort", ep, "episode.json")))
        except (OSError, ValueError):
            continue
        u = d.get("underside") or {}
        if "cavity_ratio" in u and bid in lab.get("bricks", {}):
            recs.append({"ep": ep, "brick": bid, "truth": lab["bricks"][bid], "m": u})
    um = fit(recs, cfg)
    if um is None or not any(r["truth"] == "DEFECT" for r in recs):
        sys.exit("need underside records of clean and of taped bricks")
    wrong = []
    for b in sorted({r["brick"] for r in recs}):
        mb = fit([r for r in recs if r["brick"] != b], cfg)
        for r in (r for r in recs if r["brick"] == b):
            pred = {"CLEAN": "GOOD", "DEFECT": "DEFECT"}.get(verdict(r["m"], mb) if mb else "UNCERTAIN", "UNCERTAIN")
            if pred != r["truth"]:
                wrong.append((r["ep"], b, r["truth"], pred))
    missed = [r["ep"] for r in recs if r["truth"] == "DEFECT" and verdict(r["m"], um) != "DEFECT"]
    um["provenance"] = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "records": len(recs),
                        "holdout_wrong": wrong, "missed": missed}
    print(json.dumps(um, indent=1))
    if a.write:
        if wrong or missed:
            sys.exit("not written: a held-out record is wrong or a taped bottom is missed")
        with open(MODEL_F + ".tmp", "w") as f:
            json.dump(um, f, indent=1)
        os.replace(MODEL_F + ".tmp", MODEL_F)
        print("wrote", MODEL_F)


if __name__ == "__main__":
    main()
