#!/usr/bin/env python3
"""Pick-anywhere: wrist-camera visual servoing + differential IK (kin.ik_rel, small moves from the current
configuration, so absolute FK offsets cancel) + brick-size height check. Never lets the jaws reach the mat
(TIP_FINAL 8 mm; touching the mat loosens the fixed jaw).

Frames: base frame from the URDF FK (mm). Wrist image Jacobian M (px per mm of TCP motion) from calib_M.py,
rotated with the tool yaw. TARGET = brick centre pixel when the jaws are centred at the hover (jaw_check.py).
"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grasp_auto as ga  # noqa: E402
import kin  # noqa: E402

R = ga.R
HOVER_DZ = kin.HOVER_DZ
TIP_FINAL = kin.cell.get("tip_final_mm")
SCAN_UP = kin.cell.get("scan_up_mm")
SERVO_SPEED = 12.0
SCALF = os.path.join(ga.APP, "tools", "scan_calib.json")
PLATE_HUES = ((85, 112), (38, 85))     # blue, green plates (OpenCV hue)


def target():
    return np.array(ga.TARGET_PX, dtype=float)


COLOUR_CLASSES = {  # hue classes so touching bricks of different colours become separate blobs (17:54Z pink+cream)
    "red": lambda H, S, V: ((H <= 8) | (H >= 165)) & (S >= 90) & (V >= 60),
    "orange": lambda H, S, V: (H >= 9) & (H <= 16) & (S >= 90) & (V >= 70),
    "yellow": lambda H, S, V: (H >= 17) & (H <= 38) & (S >= 45) & (V >= 90),
    "green": lambda H, S, V: (H >= 39) & (H <= 85) & (S >= 70) & (V >= 60),
    "blue": lambda H, S, V: (H >= 86) & (H <= 130) & (S >= 90) & (V >= 60),
    "white": lambda H, S, V: (S < 45) & (V >= 150),
}


JAW_LIVE_MIN_PX = 1500   # a jaw-red region entering from the bottom edge larger than this is a jaw
JAW_LIVE_GROW_PX = 15    # and is widened by this much (its shaded edge is less red)
JAW_LIVE_REACH_PX = 40   # how far a loose jaw moves from the last jaw check's mask between checks


def jaw_mask(img, hsv=None):
    """The jaws as they are in this wrist frame: the static mask from the last jaw check joined with the jaw-red
    regions that enter from the bottom edge of the view (a loose jaw moves out of the static mask)."""
    static = ga.JAW_MASK if ga.JAW_MASK is not None and ga.JAW_MASK.shape == img.shape[:2] else \
        np.zeros(img.shape[:2], bool)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if hsv is None else hsv
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((((H >= 165) | (H <= 4)) & (S >= 85) & (V >= 55)) | static).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    h = img.shape[0]
    keep = [i for i in range(1, n) if st[i, 4] > JAW_LIVE_MIN_PX and st[i, 1] + st[i, 3] >= h - 3]
    live = np.isin(lab, keep) if keep else np.zeros(red.shape, bool)
    live = cv2.dilate(live.astype(np.uint8), np.ones((JAW_LIVE_GROW_PX, JAW_LIVE_GROW_PX), np.uint8)) > 0
    if static.any():  # the jaws move at most this far from where the last jaw check saw them: a red brick touching a
        # jaw stays a brick beyond that
        live &= cv2.dilate(static.astype(np.uint8), np.ones((2 * JAW_LIVE_REACH_PX + 1,) * 2, np.uint8)) > 0
    return live | static


def jaw_mask_views(imgs):
    """One jaw mask for a set of wrist frames of the same survey: the jaws as they were in any of them."""
    m = None
    for im in imgs:
        if im is None:
            continue
        j = jaw_mask(im)
        m = j if m is None else (m | j if m.shape == j.shape else m)
    return m if m is not None else ga.JAW_MASK


def blobs(img, min_area=2500, max_area=60000):
    """Brick-like blobs in a wrist frame (jaws masked as they are in this frame), one connected component per colour
    class, with plate-surround flag."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    jaw = jaw_mask(img, hsv)
    plate = np.zeros(H.shape, bool)
    for lo, hi in PLATE_HUES:
        plate |= (H >= lo) & (H < hi) & (S >= 70) & (V >= 70)
    out = []
    for cname, fn in COLOUR_CLASSES.items():
        fg = fn(H, S, V) & ~jaw
        fg8 = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        fg8 = cv2.morphologyEx(fg8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        # 18:16Z: one white brick split into top/side pieces and the servo alternated between them; label on a
        # 15 px bridge so the pieces of one brick stay one blob, measure on the real pixels only.
        bridge = cv2.dilate(fg8, np.ones((15, 15), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(bridge)
        for i in range(1, n):
            if int(st[i, 4]) < min_area:
                continue
            m = (lab == i) & (fg8 > 0)
            a = int(m.sum())
            if a < min_area or a > max_area:
                continue
            ys, xs = np.nonzero(m)
            rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            (cx, cy), (sw, sh), ang = rect
            yaw = ang % 90.0
            if yaw > 45:
                yaw -= 90.0
            ring = cv2.dilate(m.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~m
            plate_frac = float(plate[ring].mean()) if ring.any() else 0.0
            box = cv2.boxPoints(rect)
            e1, e2 = box[1] - box[0], box[2] - box[1]
            lng = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
            x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            border = x <= 2 or y <= 2 or x + w >= img.shape[1] - 2 or y + h >= img.shape[0] - 2
            out.append({"centre": [round(float(cx), 1), round(float(cy), 1)], "size": [round(float(sw), 1), round(float(sh), 1)],
                        "short": round(float(min(sw, sh)), 1), "area": a, "yaw": round(float(yaw), 1),
                        "aspect": round(float(max(sw, sh) / max(1.0, min(sw, sh))), 2),
                        "long_vertical": bool(abs(lng[1]) >= abs(lng[0])), "plate_frac": round(plate_frac, 2),
                        "border": bool(border), "hue": int(np.median(H[m])), "sat": int(np.median(S[m])),
                        "val": int(np.median(V[m])), "colour": cname})
    return out


def brick_at_target(img, min_area=2500):
    """The brick blob nearest the servo target in a wrist frame, with its distance to the target (px): what the
    calibration tools follow while they move the arm."""
    if img is None:
        return None
    b = nearest(blobs(img, min_area=min_area), target())
    if b is None:
        return None
    return dict(b, dist_to_target=round(math.hypot(b["centre"][0] - target()[0], b["centre"][1] - target()[1]), 1))


def nearest(bl, px, max_d=1e9, ok=lambda b: True):
    c = [b for b in bl if ok(b)]
    if not c:
        return None
    b = min(c, key=lambda b: math.hypot(b["centre"][0] - px[0], b["centre"][1] - px[1]))
    return b if math.hypot(b["centre"][0] - px[0], b["centre"][1] - px[1]) <= max_d else None


_LAST = {"img": None}


def frame():
    img = ga.fresh("wrist", time.time() + 0.25)
    _LAST["img"] = img
    return img


def frame_last():
    """The last frame returned by frame() (what the failed search looked at)."""
    return _LAST["img"]


def m_base(q):
    return kin.m_base(kin.psi_of(q))


def move_rel(q, g, d, speed=6.0, comp=True, clamp=None):
    qn = kin.ik_rel(q, d)
    if qn is None:
        return None
    c = {"op": "move", "q": qn + [g], "speed": speed, "comp": comp, "accel": ga.ACCEL}
    if clamp is not None:
        c["comp_clamp"] = clamp
    st = ga.cmd(c)
    if st["status"] != "ok":
        raise ga.Stop(f"move_rel {d} -> {st['status']} {st['msg']}")
    return qn


COMPAT = {"red": ("red", "orange"), "orange": ("orange", "red", "yellow"), "yellow": ("yellow", "white", "orange"),
          "white": ("white", "yellow"), "green": ("green",), "blue": ("blue",)}


def _ok(want, short_ref, strict):
    def ok(x):
        if x["plate_frac"] >= 0.3:
            return False
        if want and x.get("colour") not in ((want,) if strict else COMPAT.get(want, (want,))):
            return False
        if short_ref and not (0.6 * short_ref <= x["short"] <= 1.6 * short_ref):
            return False
        return True
    return ok


LOST_LOG = []  # (utc, px, blobs) of searches that failed inside the radius; the loop writes them to the episode


def find(bl, px, max_d, want=None, short_ref=None):
    """Blob nearest px of the tracked brick's colour class (exact class first, then a compatible one), within max_d.
    If nothing is within max_d, the whole frame is searched for the same colour class (and size, when short_ref is
    given): the brick moved further than expected (large yaw, a servo step at another height), it did not vanish.
    A neighbour of the same class nearer the expected position still wins, as before."""
    b = nearest(bl, px, max_d=max_d, ok=_ok(want, short_ref, True))
    if b is None and want:
        b = nearest(bl, px, max_d=max_d, ok=_ok(want, short_ref, False))
    if b is not None and want:  # 05:56Z: a cream brick's yellow side sliver won over its white top face (yaw 88 deg)
        near = [x for x in bl if _ok(want, short_ref, False)(x)
                and math.hypot(x["centre"][0] - b["centre"][0], x["centre"][1] - b["centre"][1]) < max(x["short"], b["short"])]
        big = max(near, key=lambda x: x["area"]) if near else None
        if big is not None and big["area"] > 2.5 * b["area"]:
            b = big
    if b is None and want:
        def same(x):  # exact colour class; size within +-20 % when the tracked size is known
            return (_ok(want, None, True)(x)
                    and (short_ref is None or 0.8 * short_ref <= x["short"] <= 1.2 * short_ref))
        b = nearest(bl, px, max_d=1e9, ok=same)
        try:
            os.makedirs(f"{ga.RUN_ROOT}/lost", exist_ok=True)
            cv2.imwrite(f"{ga.RUN_ROOT}/lost/{time.strftime('%H%M%S', time.gmtime())}.jpg", frame_last())
        except Exception:  # noqa: BLE001
            pass
        LOST_LOG.append((time.strftime("%H:%M:%SZ", time.gmtime()), [round(float(v), 1) for v in px], round(float(max_d)),
                         [(x.get("colour"), int(x.get("area", 0)), [round(c) for c in x["centre"]], x.get("short")) for x in bl],
                         None if b is None else [round(c) for c in b["centre"]]))
    return b


SERVO_MIN_AREA = 1500  # 17:59Z: a brick half under the static jaw mask fell below 2500 px and was "not visible"


SERVO_CORR_FILE = os.path.join(ga.RUN_ROOT, "servo_corr.json")
SERVO_CORR = {"C": np.eye(2), "samples": [], "loaded": False}


def servo_corr():
    """2x2 correction C of the calibrated wrist Jacobian (image motion seen = C @ M @ step), learned online."""
    if not SERVO_CORR["loaded"]:
        SERVO_CORR["loaded"] = True
        try:
            d = json.load(open(SERVO_CORR_FILE))
            SERVO_CORR["C"] = np.array(d["C"], float)
            SERVO_CORR["samples"] = [(np.array(u, float), np.array(p, float)) for u, p in d["samples"]][-40:]
        except (OSError, ValueError, KeyError):
            pass
    return SERVO_CORR["C"]


def learn_servo_corr(u, dp, ridge=200.0):
    """U: image motion the calibrated Jacobian predicts for the step taken; dp: image motion seen. Keeps the last 40
    steps that moved the brick at least 8 px and agree with the current model within 40 px (a bigger miss is a
    tracking jump, not the camera), and refits C by ridge least squares pulled toward the identity."""
    C = servo_corr()
    if np.linalg.norm(u) < 8 or np.linalg.norm(dp - C @ u) > 40:
        return C
    SERVO_CORR["samples"] = (SERVO_CORR["samples"] + [(np.asarray(u, float), np.asarray(dp, float))])[-40:]
    U = np.array([s[0] for s in SERVO_CORR["samples"]])
    D = np.array([s[1] for s in SERVO_CORR["samples"]])
    C = np.linalg.solve(U.T @ U + ridge * np.eye(2), U.T @ D + ridge * np.eye(2)).T
    SERVO_CORR["C"] = C
    try:
        tmp = SERVO_CORR_FILE + ".tmp"
        json.dump({"C": C.tolist(), "samples": [[a.tolist(), b.tolist()] for a, b in SERVO_CORR["samples"]],
                   "t": time.strftime("%H:%M:%SZ", time.gmtime())}, open(tmp, "w"))
        os.replace(tmp, SERVO_CORR_FILE)
    except OSError:
        pass
    return C


def servo(q, g, start_px, tol=7.0, max_it=6, log=print, want=None):
    """Bring the tracked brick (colour class `want`) to TARGET at the current height. Returns (q, blob, converged).
    Each step also teaches the Jacobian correction (learn_servo_corr), so a knocked camera is followed."""
    px = np.array(start_px, float)
    b = None
    prev = None  # (predicted image motion of the last step, brick centre before it)
    for it in range(max_it):
        bl = blobs(frame(), min_area=SERVO_MIN_AREA)
        b = find(bl, px, 140, want)
        if b is None:
            return q, None, False
        if prev is not None:
            learn_servo_corr(prev[0], np.array(b["centre"], float) - prev[1])
        e = target() - np.array(b["centre"])
        log(f"servo it{it}: {b.get('colour')} brick {b['centre']} err {e.round(1).tolist()} short {b['short']}")
        if np.linalg.norm(e) <= tol:
            return q, b, True
        M = m_base(q)
        d = 0.65 * np.linalg.solve(servo_corr() @ M, e)  # damped: the 17:32Z servo overshot at gain 1
        n = np.linalg.norm(d)
        if n > 30:
            d *= 30 / n
        qn = move_rel(q, g, [d[0], d[1], 0.0], speed=SERVO_SPEED)
        if qn is None:
            d = d / 2
            qn = move_rel(q, g, [d[0], d[1], 0.0], speed=SERVO_SPEED)
            if qn is None:
                return q, b, False
        q = qn
        prev = (M @ d, np.array(b["centre"], float))
        px = np.array(b["centre"], float) + servo_corr() @ M @ d  # expected position after the step
    bl = blobs(frame(), min_area=SERVO_MIN_AREA)
    b = find(bl, target(), 140, want)
    return q, b, b is not None and np.linalg.norm(target() - np.array(b["centre"])) <= 2 * tol


def align_yaw(q, g, b, log=print):
    yaw = b["yaw"]
    if b["aspect"] > 1.4 and not b["long_vertical"]:
        yaw = yaw + 90.0 if yaw < 0 else yaw - 90.0
    if abs(yaw) <= 2.5:
        return q, False
    if abs(yaw) > 45.0:  # only a blob of two touching bricks reads like this: grasp as it is rather than lose it
        log(f"yaw {yaw:.1f} deg not aligned (merged blob)")
        return q, False
    nr = q[4] - yaw / ga.YAW_PER_ROLL
    if abs(nr) > 97 and b["aspect"] <= 1.4:
        alt = yaw + 90.0 if yaw < 0 else yaw - 90.0
        nr = q[4] - alt / ga.YAW_PER_ROLL
    nr = float(np.clip(nr, -97, 97))
    qn = list(q)
    qn[4] = round(nr, 2)
    st = ga.cmd({"op": "move", "q": qn + [g], "speed": max(25.0, SERVO_SPEED), "comp": True, "accel": max(120.0, ga.ACCEL)})
    if st["status"] != "ok":
        raise ga.Stop(f"yaw -> {st['status']}")
    log(f"yaw {yaw:.1f} deg -> roll {qn[4]}")
    return qn, True


def load_scan():
    return json.load(open(SCALF)) if os.path.exists(SCALF) else None
