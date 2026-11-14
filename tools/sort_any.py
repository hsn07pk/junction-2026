#!/usr/bin/env python3
"""Autonomous sort loop, PICK ANYWHERE (bricks at random places on the mat).

Scan pose (wrist camera high over the work area) -> choose the brick nearest the view target (not in a plate) ->
differential-IK move above it -> hover servo (wrist cam) + yaw align -> size-checked straight descent (jaws stop
8 mm above the mat) -> close -> lift check -> fixed inspection pose in front of the close camera, multi-view rolls ->
decide (OD-10, I-52) -> plate path -> back to scan. Every cycle recorded (30 Hz joints + MKVs) and logged:
R/logs/sort/<RUN>/ep_<NNNN>/. Stop: touch R/run/sort_stop. Pause: touch R/run/sort_pause.
usage: sort_any.py --run NAME [--n 150] [--speed 1.0]
"""
import argparse
import subprocess
import json
import math
import os
import sys
import time
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "anomaly", "det2_f2"))
import grasp_auto as ga  # noqa: E402
import kin  # noqa: E402
import pick_any as pa  # noqa: E402
import sort_loop as sl  # noqa: E402  (close_verdict, decide, skin/hands helpers, run_fast)
import inspector as DET2  # noqa: E402
import closeloc as cl
import faces as fcs  # noqa: E402
import survey as srv  # noqa: E402
from paths import APP  # noqa: E402  # noqa: E402

R = ga.R
SPOT_HOVER = [-12.62, 30.0, -19.33, 69.73, -45.65]
PLATES_XY = {"blue": (187.1, -86.1), "green": (194.6, 218.3)}
PLATE_KEEPOUT = 35.0  # 17:45Z: 85 excluded the cream brick 4 cm outside the green rim; plate_frac handles in-plate bricks
CFG = {"transit": 42.0, "hover": 18.0, "servo": 12.0, "desc": 10.0, "final": 4.0}
EP = {}
EPIMG = {}  # this episode's first frames, taken at the scan pose with the arm out of the close view
RUN = ""
CLOSE_MAP = cl.ClosePlaneMap(os.path.join(ga.RUN_ROOT, "close_pairs.jsonl"))


def log(msg):
    line = time.strftime("%H:%M:%SZ ", time.gmtime()) + msg
    print(line, flush=True)
    with open(f"{RUN}/loop.log", "a") as f:
        f.write(line + "\n")


def scan_cal():
    c = pa.load_scan()
    if not c or c.get("q_scan") is None:
        raise SystemExit("scan_calib.json missing: run scan_calib.py with a brick under the hover")
    return c


def go_scan(holding=False):
    c = scan_cal()
    g = 0.0 if holding else 25.0
    st = ga.state()
    p, _ = kin.fk(st["present"][:5])
    if p[2] < kin.MAT_Z + 45:  # low: go straight up first; a failed lift must not turn into a low sweep
        up = kin.MAT_Z + 75 - p[2]
        q = pa.move_rel(st["present"][:5], g, [0, 0, up], speed=CFG["hover"])
        if q is None:
            q = st["present"][:5]
            for _ in range(int(math.ceil(up / 20.0))):
                q = pa.move_rel(q, g, [0, 0, min(20.0, up)], speed=CFG["hover"])
                if q is None:
                    raise ga.Stop("lift before the scan move failed (IK): not sweeping from a low pose")
    tmove(c["q_scan"] + [g], CFG["transit"], comp=True)


SCAN_PAIRS = f"{ga.RUN_ROOT}/scan_pairs.jsonl" if ga.SIM_ROOT else f"{R}/app/tools/scan_pairs.jsonl"  # (scan px, servo-verified hover xy) from every centred approach
SCAN_C = np.eye(2)


def jac_xy_from_scan(q_scan, px, c):
    M = kin.m_base(kin.psi_of(q_scan)) * c["scale"]
    return np.linalg.solve(M, np.array(c["target_scan"]) - np.array(px, float))


SCAN_H = None
SCAN_HULL = None  # scan-image area covered by the pairs the homography was fitted on


def fit_scan_corr():
    """Learned scan map from servo-verified approaches (scan_pairs.jsonl), refitted every cycle.
    >= 15 pairs: RANSAC homography scan px -> base xy (perspective; the 18:15Z brick near the image bottom was 48 mm
    off any linear model). Fewer: similarity correction (scale + rotation) of the scan Jacobian with one round of
    outlier rejection (> 25 mm). 17:58Z seed fit: scale 0.70, rot -9.5 deg."""
    global SCAN_C, SCAN_H, SCAN_HULL, SCAN_RESID_MM
    c = scan_cal()
    p, _ = kin.fk(c["q_scan"])
    PX, XY = [], []
    if os.path.exists(SCAN_PAIRS):
        for line in open(SCAN_PAIRS):
            try:
                d = json.loads(line)
                PX.append([float(d["px"][0]), float(d["px"][1])])
                XY.append([float(d["xy"][0]), float(d["xy"][1])])
            except Exception:  # noqa: BLE001
                continue
    SCAN_H, SCAN_HULL = None, None
    if len(PX) >= 15:
        H, inl = cv2.findHomography(np.array(PX, np.float32), np.array(XY, np.float32), cv2.RANSAC, 15.0)
        if H is not None and int(inl.sum()) >= 10:
            pr = cv2.perspectiveTransform(np.array(PX, np.float32)[None], H)[0]
            res = np.linalg.norm(pr - np.array(XY), axis=1)[inl.ravel() > 0]
            if float(np.median(res)) <= 10:
                SCAN_H = H
                SCAN_HULL = cv2.convexHull(np.array(PX, np.float32))
                SCAN_RESID_MM = float(np.median(res))
                log(f"scan map: homography from {int(inl.sum())}/{len(PX)} pairs, median residual {float(np.median(res)):.1f} mm "
                    "(used inside the area its pairs cover; similarity elsewhere)")
    if len(PX) < 4:
        SCAN_C = np.eye(2)
        SCAN_RESID_MM = None
        return
    J = np.array([jac_xy_from_scan(c["q_scan"], px, c) for px in PX])
    A = np.array(XY) - p[:2]

    def sim(idx):
        M, y = [], []
        for dj, da in zip(J[idx], A[idx]):
            M += [[dj[0], -dj[1]], [dj[1], dj[0]]]
            y += [da[0], da[1]]
        (a, bb), *_ = np.linalg.lstsq(np.array(M), np.array(y), rcond=None)
        return np.array([[a, -bb], [bb, a]])

    idx = np.arange(len(J))
    C = sim(idx)
    res = np.linalg.norm(J @ C.T - A, axis=1)
    keep = idx[res <= 25.0]
    if len(keep) >= 4 and len(keep) < len(idx):
        C = sim(keep)
        res = np.linalg.norm(J @ C.T - A, axis=1)
    SCAN_C = C if float(np.median(res)) <= 15 else np.eye(2)
    if SCAN_H is None:
        SCAN_RESID_MM = float(np.median(res))
    log(f"scan map: similarity from {len(keep)}/{len(J)} pairs, scale {math.hypot(C[0, 0], C[1, 0]):.3f} "
        f"rot {math.degrees(math.atan2(C[1, 0], C[0, 0])):.1f} deg, median residual {float(np.median(res)):.1f} mm")


def record_scan_pair(px, q_hover):
    ph, _ = kin.fk(q_hover)
    with open(SCAN_PAIRS, "a") as f:
        f.write(json.dumps({"px": [float(px[0]), float(px[1])], "xy": [round(float(ph[0]), 1), round(float(ph[1]), 1)],
                            "t": time.strftime("%H:%M:%SZ", time.gmtime()), "run": os.path.basename(RUN),
                            "ep": EP.get("episode"), "target_px": [float(v) for v in ga.TARGET_PX]}) + "\n")


def brick_xy_from_scan(q_scan, b, c):
    """Base-frame offset (mm) of a brick seen from the scan pose relative to the point under the jaws (scan Jacobian
    with the learned similarity correction)."""
    if SCAN_H is not None and SCAN_HULL is not None and cv2.pointPolygonTest(
            SCAN_HULL, (float(b["centre"][0]), float(b["centre"][1])), True) >= -40.0:
        p, _ = kin.fk(q_scan)
        xy = cv2.perspectiveTransform(np.array([[b["centre"]]], np.float32), SCAN_H)[0, 0]
        return np.array([xy[0] - p[0], xy[1] - p[1]])
    return SCAN_C @ jac_xy_from_scan(q_scan, b["centre"], c)  # outside the pairs: no perspective extrapolation


REACH_CACHE = {}  # (q_scan, 5 mm cell) -> bool


def grasp_reachable(q_scan, d):
    """Whether the approach and the descent can reach a brick at base offset d (mm) from the point under the jaws at
    the scan pose. Replays, without moving, the IK chain approach() and grasp() run: down to the hover height, the
    60/30/15 mm steps toward the brick, then the two descent stages. 3 of 6 approach failures in runs any_1731/1740
    were IK failures on bricks inside the old 335 mm radius rule; at hover height the reach ends near 300 mm."""
    key = (tuple(round(v, 1) for v in q_scan), round(d[0] / 5.0), round(d[1] / 5.0))
    if key in REACH_CACHE:
        return REACH_CACHE[key]
    ok = False
    q = kin.ik_rel(list(q_scan), [0.0, 0.0, -pa.SCAN_UP])
    rem = np.array(d, float)
    for _ in range(8):
        if q is None:
            break
        n = float(np.linalg.norm(rem))
        if n < 3:
            break
        for lim in (60.0, 30.0, 15.0):
            dd = rem if n <= lim else rem * lim / n
            qn = kin.ik_rel(q, [dd[0], dd[1], 0.0])
            if qn is not None:
                q, rem = qn, rem - dd
                break
        else:
            q = None
    if q is not None and float(np.linalg.norm(rem)) < 3:
        tip = float(kin.fk(q)[0][2] - kin.MAT_Z)
        q1 = kin.ik_rel(q, [0.0, 0.0, -(tip - 26.0)])
        ok = q1 is not None and kin.ik_rel(q1, [0.0, 0.0, -(26.0 - pa.TIP_FINAL)]) is not None
    REACH_CACHE[key] = ok
    return ok


PLATE_MIN_AREA = 8000  # px at the scan pose: plates measured 9178-27306 px, bricks 850-6400 px (2x2 and 2x4 smaller than this)


def plate_regions(img):
    """Plate pixels in a wrist scan frame: plate-coloured regions of at least PLATE_MIN_AREA that touch the image
    border (both plates sit at the sides of the work area), dilated 15 px. Independent of the learned scan map."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue = np.zeros(H.shape, np.uint8)
    for lo, hi in pa.PLATE_HUES:
        hue |= ((H >= lo) & (H < hi) & (S >= 70) & (V >= 70)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(hue)
    h, w = H.shape
    plates = np.zeros(H.shape, np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area >= PLATE_MIN_AREA and (x == 0 or y == 0 or x + bw >= w or y + bh >= h):
            plates[lab == i] = 1
    return cv2.dilate(plates, np.ones((15, 15), np.uint8))


def candidates(img, q_scan, c):
    out = []
    p, _ = kin.fk(q_scan)
    plates = plate_regions(img)
    for b in pa.blobs(img, min_area=700, max_area=40000):
        if b["plate_frac"] >= 0.3 or (b.get("colour") in ("blue", "green") and b["plate_frac"] >= 0.15):
            continue  # inside a plate, or a plate rim fragment
        cx, cy = int(round(b["centre"][0])), int(round(b["centre"][1]))
        if 0 <= cy < plates.shape[0] and 0 <= cx < plates.shape[1] and plates[cy, cx]:
            continue  # part of a plate, found in the image (no table map involved)
        d = brick_xy_from_scan(q_scan, b, c)
        xy = (p[0] + d[0], p[1] + d[1])
        if min(math.hypot(xy[0] - v[0], xy[1] - v[1]) for v in PLATES_XY.values()) < PLATE_KEEPOUT:
            continue
        r = math.hypot(xy[0] - kin.PAN_XY_MM[0], xy[1] - kin.PAN_XY_MM[1])
        if r < 115 or r > 360 or not grasp_reachable(q_scan, d):  # at the base, or out of reach
            continue
        b["d_mm"] = [round(float(d[0]), 1), round(float(d[1]), 1)]
        b["xy"] = [round(float(xy[0]), 1), round(float(xy[1]), 1)]
        out.append(b)
    now = time.time()
    out = [b for b in out if not any(math.hypot(b["xy"][0] - x, b["xy"][1] - y) < 25 and now - t < 300 for x, y, t in FAILED_XY)]
    out.sort(key=lambda b: (b["border"], math.hypot(*b["d_mm"])))  # whole bricks first; edge bricks get re-centred
    return out


def wait_for_bricks():
    c = scan_cal()
    t0 = time.time()
    last = None
    while True:
        for f in ("sort_stop",):
            if os.path.exists(f"{ga.RUN_ROOT}/{f}"):
                raise SystemExit("stop requested")
        if os.path.exists(f"{ga.RUN_ROOT}/sort_pause") or sl.hands_seen(2):
            last = None
            time.sleep(0.4)
            continue
        cand = candidates(pa.frame(), c["q_scan"], c)
        if cand:
            b = cand[0]
            if last is not None and math.hypot(b["centre"][0] - last[0], b["centre"][1] - last[1]) < 8:
                return b, cand, round(time.time() - t0, 1)
            last = b["centre"]
        else:
            last = None
        time.sleep(0.15)


MIN_TIP_MM = 3.0  # lowest jaw-tip height above the mat (FK) any descent may reach
FAILED_XY = []   # (x, y, t) of bricks whose approach failed; skipped for 5 min


def approach(b, c):
    """Scan estimate -> descend to the hover height at the scan location -> differential-IK steps (<= 60 mm) at the
    hover height toward the estimate (more reach margin than at the scan height, 17:45Z IK failures) -> wrist servo
    + yaw align. Returns (q, blob) or (None, reason)."""
    g = 25.0
    q = pa.move_rel(list(c["q_scan"]), g, [0.0, 0.0, -pa.SCAN_UP], speed=CFG["hover"])
    if q is None:
        return None, "ik fail descending to hover"
    d = np.array(b["d_mm"], float)
    for step in range(8):
        n = float(np.linalg.norm(d))
        if n < 3:
            break
        moved = False
        for lim in (60.0, 30.0, 15.0):
            dd = d if n <= lim else d * lim / n
            qn = pa.move_rel(q, g, [dd[0], dd[1], 0.0], speed=CFG["hover"])
            if qn is not None:
                q, d, moved = qn, d - dd, True
                break
        if not moved:
            return None, f"ik fail moving toward brick (remaining {d.round(1).tolist()} mm)"
    want = b.get("colour")  # track this brick's colour class from scan to grasp (17:56Z neighbour switch)
    EP["target_colour"] = want
    bl = pa.blobs(pa.frame(), min_area=pa.SERVO_MIN_AREA)
    bh = pa.find(bl, pa.target(), 260, want)
    if bh is None:  # second look from 30 mm higher (wider view), centre there, then come back down
        qu = pa.move_rel(q, g, [0.0, 0.0, 30.0], speed=CFG["hover"])
        if qu is not None:
            bu = pa.find(pa.blobs(pa.frame(), min_area=1000), pa.target(), 320, want)
            if bu is not None:
                log(f"raised look: {want} brick at {bu['centre']}")
                qu, bu, _ = pa.servo(qu, g, bu["centre"], log=log, want=want, tol=15.0)
            qd = pa.move_rel(qu, g, [0.0, 0.0, -30.0], speed=CFG["hover"])
            if qd is not None:
                q = qd
        bh = pa.find(pa.blobs(pa.frame(), min_area=pa.SERVO_MIN_AREA), pa.target(), 260, want)
    if bh is None:
        fr = pa.frame()
        cv2.imwrite(f"{ga.OUT}/crops/hover_not_found.jpg", fr)
        seen = [(x.get("colour"), int(x.get("area", 0)), [round(v) for v in x["centre"]], x.get("short"))
                for x in pa.blobs(fr, min_area=pa.SERVO_MIN_AREA)]
        log(f"hover search failed: wanted {want}; blobs now {seen}")
        return None, f"{want} brick not visible at hover"
    q, bh, ok = pa.servo(q, g, bh["centre"], log=log, want=want)
    if bh is None:
        return None, "servo lost brick"
    q, rot = pa.align_yaw(q, g, bh, log=log)
    if rot:
        bh = pa.find(pa.blobs(pa.frame()), pa.target(), 260, want, short_ref=bh["short"])
        if bh is None:
            return None, "brick lost after yaw"
        q, bh, ok = pa.servo(q, g, bh["centre"], log=log, want=want)
    if not ok or bh is None:
        return None, "could not centre at hover"
    if pa.LOST_LOG:  # the tracker had to search beyond its radius: it may hold another brick now, so no pair
        log("scan pair not recorded: the brick was re-found outside the search radius during this approach")
    else:
        try:
            record_scan_pair(b["centre"], q)
        except Exception as e:  # noqa: BLE001
            log(f"scan pair not recorded: {e}")
    return q, bh


def close_pre_verdict(img, want_hue=None):
    """DET2 close tape rule for the brick under the jaws, located from the jaw tips in the close image (works
    anywhere the close camera sees the gripper). None if the jaws are not in the close view."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((H <= 8) | (H >= 170)) & (S >= 100) & (V >= 50)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    tips = []
    for i in range(1, n):
        if st[i, 1] <= 3 and st[i, 4] > 800:
            ys, xs = np.nonzero(lab == i)
            yb = ys.max()
            tips.append((float(xs[ys >= yb - 6].mean()), float(yb)))
    if not tips:
        return None
    tips.sort(key=lambda t: -t[1])
    mx = float(np.mean([t[0] for t in tips[:2]]))
    my = float(max(t[1] for t in tips[:2])) + 40.0
    try:
        seg = DET2.segment_close(sl.mask_jaws(img))
    except Exception as e:  # noqa: BLE001
        return {"verdict": "UNCERTAIN", "reasons": [f"det2_error {e!r}"]}
    best = None
    for rec in seg["bricks"]:
        cx, cy = rec["centre_px"]
        d = math.hypot(cx - mx, cy - my)
        if best is None or d < best[0]:
            best = (d, rec)
    if best is None or best[0] > 130:
        return {"verdict": "UNCERTAIN", "reasons": ["no_brick_under_jaws_in_close_view"], "jaw_mid": [mx, my]}
    rec = best[1]
    verdict, reasons = DET2.triage(rec, DET2.load_thresholds())
    x, y, w, h = rec["bbox"]
    out = {"verdict": verdict, "reasons": reasons, "colour": rec.get("colour"), "tape_score": rec.get("tape_score"),
           "bbox": [int(x), int(y), int(w), int(h)], "dist": round(best[0], 1), "jaw_mid": [round(mx), round(my)],
           "hue": int(rec.get("hsv_med", [0])[0])}
    if want_hue is not None:
        dh = abs(out["hue"] - want_hue)
        dh = min(dh, 180 - dh)
        out["hue_diff"] = int(dh)
        if dh > 10:  # 17:41Z: the close view matched a different brick (taped cream) to the picked orange one
            out["verdict_raw"] = verdict
            out["verdict"] = "UNCERTAIN"
            out["reasons"] = list(reasons) + [f"close_brick_hue_mismatch_{int(dh)}"]
    return out


def rec_verdict(rec, method):
    """DET2 triage of one segmented brick, in the verdict format used by decide()."""
    verdict, reasons = DET2.triage(rec, DET2.load_thresholds())
    x, y, w, h = rec["bbox"]
    return {"verdict": verdict, "reasons": list(reasons), "colour": rec.get("colour"), "tape_score": rec.get("tape_score"),
            "bbox": [int(x), int(y), int(w), int(h)], "centre_px": [round(float(c), 1) for c in rec["centre_px"]],
            "hue": int(rec.get("hsv_med", [0])[0]), "method": method}


def _hue_check(out, want_hue):
    if want_hue is None or out.get("hue") is None:
        return out
    dh = abs(out["hue"] - want_hue)
    dh = min(dh, 180 - dh)
    out["hue_diff"] = int(dh)
    if dh > 10:  # the close view matched a different brick than the one centred in the wrist view
        out["verdict_raw"] = out["verdict"]
        out["verdict"] = "UNCERTAIN"
        out["reasons"] = list(out["reasons"]) + [f"close_brick_hue_mismatch_{int(dh)}"]
    return out


def pre_close_verdict(cimg, want_hue=None):
    """Close-camera verdict on the brick about to be picked. The jaws are the only new red parts since this episode's
    first close frame; the brick under them is judged on that first frame, where the gripper does not cover it. No
    fixed pixel is involved, so a moved close camera does not break it. Falls back to the jaw-tip rule (jaws entering
    from the image top) when the first frame is missing or no jaws are found."""
    before = (EPIMG.get("pre") or {}).get("close")
    if cimg is None:
        return None
    if before is None:
        out = close_pre_verdict(cimg, want_hue)
        if out is not None:
            out["method"] = "jaw_top_rule"
            if out.get("bbox"):
                sl.save_crop(cimg, out["bbox"], f"{ga.OUT}/crops/pre_close.png")
        return out
    tips = cl.jaw_tips_by_diff(cimg, before)
    if len(tips) < 2:
        out = close_pre_verdict(cimg, want_hue) or {"verdict": "UNCERTAIN", "reasons": ["no_jaws_in_close_view"]}
        out.update({"method": "jaw_top_rule", "jaw_tips_diff": [[round(x), round(y)] for x, y in tips]})
        if out.get("bbox"):
            sl.save_crop(cimg, out["bbox"], f"{ga.OUT}/crops/pre_close.png")
        return out
    mx = (tips[0][0] + tips[1][0]) / 2.0
    my = max(tips[0][1], tips[1][1])
    try:
        bricks = DET2.segment_close(sl.mask_jaws(before))["bricks"]
    except Exception as e:  # noqa: BLE001
        return {"verdict": "UNCERTAIN", "reasons": [f"det2_error {e!r}"], "method": "diff_first_frame"}
    rec, d = cl.brick_below(bricks, mx, my)
    if rec is None:
        return {"verdict": "UNCERTAIN", "reasons": ["no_brick_under_jaws_in_close_view"], "method": "diff_first_frame",
                "jaw_tips": [[round(x), round(y)] for x, y in tips]}
    out = rec_verdict(rec, "diff_first_frame")
    out.update({"jaw_tips": [[round(x), round(y)] for x, y in tips], "dist": d})
    sl.save_crop(before, out["bbox"], f"{ga.OUT}/crops/pre_close.png")
    return _hue_check(out, want_hue)


def grasp(q, bh, retry=False, tag=""):
    """Descend on the centred brick, close, lift, check the hold. tag="re_" for the re-pick at the station: its
    snapshots and fields get that prefix, so the first pick's record stays intact."""
    g = 25.0
    tip = pa.tip_from_size(bh["short"])
    fk_tip = float(kin.fk(q)[0][2] - kin.MAT_Z)
    use = tip if abs(tip - fk_tip) <= 15 else fk_tip
    tip_final = pa.TIP_FINAL
    if retry:  # a leaning upside-down brick looked big (size 51 mm, FK 58 mm) and the jaws closed too high
        use = min(max(tip, fk_tip), fk_tip + 6.0)
        tip_final = pa.TIP_FINAL - 2.0
    use = min(use, fk_tip + tip_final - MIN_TIP_MM)  # jaws end at tip_final + fk_tip - use: never below MIN_TIP_MM
    EP[tag + "grasp_use_mm" + ("_retry" if retry else "")] = round(use, 1)
    EP[tag + "hover_tip_size_mm"] = round(tip, 1)
    shape = ("not_2x2", f"aspect {bh['aspect']}") if bh["aspect"] > 1.4 else (
        ("not_2x2", f"narrow short {bh['short']}px vs {pa.SIZE_K / (use + pa.SIZE_C):.0f}") if bh["short"] < 0.72 * pa.SIZE_K / (use + pa.SIZE_C)
        else ("2x2", f"aspect {bh['aspect']} short {bh['short']}"))
    EP[tag + "shape"], EP[tag + "shape_why"] = shape
    cv2.imwrite(f"{ga.OUT}/crops/{tag}wrist_hover.png", pa.frame())
    cimg = ga.fresh("close", time.time() + 0.2)
    if cimg is not None:
        cv2.imwrite(f"{ga.OUT}/crops/{tag}hover_close.jpg", cimg)
    if not tag:
        EP["pre_close"] = pre_close_verdict(cimg, want_hue=bh.get("hue"))
    q1 = pa.move_rel(q, g, [0, 0, -(use - 26.0)], speed=CFG["desc"])
    if q1 is None:
        return False, "ik fail stage 1"
    b1 = pa.nearest(pa.blobs(pa.frame(), max_area=90000), pa.target(), max_d=260)
    ga.snap(tag + "stage1")
    q2 = pa.move_rel(q1, g, [0, 0, -(26.0 - tip_final)], speed=CFG["final"])
    if q2 is None:
        return False, "ik fail final descent"
    gimgs = ga.snap(tag + "grasp_open")
    bo = pa.nearest(pa.blobs(gimgs["wrist"], max_area=200000), pa.target(), max_d=400) if gimgs.get("wrist") is not None else None
    EP[tag + "open_short"] = bo["short"] if bo else None
    ga.cmd({"op": "grip", "value": 0.0, "speed": 20})
    time.sleep(0.5)
    gr = ga.state()["present"][5]
    EP[tag + "grip_closed"] = gr
    ga.snap(tag + "closed")
    ql = pa.move_rel(q2, 0.0, [0, 0, 45.0], speed=CFG["desc"])
    if ql is None:
        return False, "ik fail lift"
    imgs = ga.snap(tag + "lifted")
    gr2 = ga.state()["present"][5]
    held = pa.nearest(pa.blobs(imgs["wrist"], max_area=150000), pa.target(), max_d=300)
    EP.update({tag + "grip_lifted": gr2, tag + "lift_q": ql, tag + "stage1_blob": b1, tag + "lifted_blob": held})
    ratio = (held["short"] / EP[tag + "open_short"]) if (held and EP.get(tag + "open_short")) else None
    EP[tag + "held_size_ratio"] = round(ratio, 2) if ratio else None
    if gr2 < 1.1:  # an empty close reads 0.59, a held 2x2/2x4 reads 1.57-2.42
        ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
        return False, f"nothing held (gripper {gr2}, size ratio {ratio})"
    return True, shape


STATION_PX_FILE = os.path.join(ga.RUN_ROOT, "station_px.json")


def station_px():
    try:
        return json.load(open(STATION_PX_FILE))["px"]
    except (OSError, ValueError, KeyError):
        return None


def learn_station_px(px):
    old = station_px()
    new = px if old is None else [0.7 * old[0] + 0.3 * px[0], 0.7 * old[1] + 0.3 * px[1]]
    tmp = STATION_PX_FILE + ".tmp"
    json.dump({"px": [round(new[0], 1), round(new[1], 1)], "t": time.strftime("%H:%M:%SZ", time.gmtime())}, open(tmp, "w"))
    os.replace(tmp, STATION_PX_FILE)


def station_verdict(st_img, put_xy):
    """Verdict on the brick just put down at the station, never on another brick. The station brick is the brick that
    appeared since this episode's first close frame (changes connected to the image top edge = people, ignored), with
    a colour compatible with what the wrist camera centred; when the difference is weak (e.g. the brick was picked
    from the station and put back) the brick at the learned station pixel is used. No clean candidate -> UNCERTAIN.
    The picked brick's original view (the brick that vanished from the first frame) is kept as first_frame_view.
    Clean detections give table-position -> close-pixel pairs for the close map."""
    before = (EPIMG.get("pre") or {}).get("close")
    want = EP.get("target_colour")
    if st_img is None or before is None:
        return {"verdict": "UNCERTAIN", "reasons": ["no first frame or station frame"], "method": "none"}
    ch = cl.drop_top_edge(cl.changed_mask(st_img, before))
    try:
        seg_st = [r for r in DET2.segment_close(sl.mask_jaws(st_img))["bricks"] if cl.colour_ok(r.get("colour"), want)]
        seg_pre = [r for r in DET2.segment_close(sl.mask_jaws(before))["bricks"] if cl.colour_ok(r.get("colour"), want)]
    except Exception as e:  # noqa: BLE001
        return {"verdict": "UNCERTAIN", "reasons": [f"det2_error {e!r}"], "method": "none"}
    van, fv, fv2 = cl.pick_changed(seg_pre, ch, min_frac=0.5)
    if van is not None:
        ffv = rec_verdict(van, "vanished_first_frame")
        ffv["changed_frac"] = [fv, fv2]
        EP["first_frame_view"] = ffv
        sl.save_crop(before, ffv["bbox"], f"{ga.OUT}/crops/first_frame_view.png")
        lq = EP.get("lift_q")
        if lq is not None and fv >= 0.7:
            gx, gy = (float(x) for x in kin.fk(lq)[0][:2])
            CLOSE_MAP.add(gx, gy, ffv["centre_px"][0], ffv["centre_px"][1], "grasp")
    sp = station_px()
    app, fa, fa2 = cl.pick_changed(seg_st, ch, min_frac=0.45)
    if app is not None:
        d = None if sp is None else float(np.hypot(app["centre_px"][0] - sp[0], app["centre_px"][1] - sp[1]))
        if d is None or d <= 120 or fa >= 0.7:  # far from the learned spot only with a strong change (camera moved)
            v = rec_verdict(app, "appeared")
            v.update({"changed_frac": [fa, fa2], "station_px_dist": None if d is None else round(d, 1)})
            if fa >= 0.6:
                learn_station_px(v["centre_px"])
                if put_xy is not None:
                    CLOSE_MAP.add(float(put_xy[0]), float(put_xy[1]), v["centre_px"][0], v["centre_px"][1], "station")
            return v
    if sp is not None:
        rec, d = cl.nearest(seg_st, sp[0], sp[1], 90)
        if rec is not None:
            v = rec_verdict(rec, "station_px")
            v.update({"changed_frac": [fa, fa2], "station_px_dist": d})
            return v
    return {"verdict": "UNCERTAIN", "reasons": ["station brick not found (no clean change, no brick at the learned spot)"],
            "method": "none", "changed_frac": [fa, fa2]}


def station_inspect(k):
    """Put the held brick down at the station, retract to the scan pose, judge it in the close view (station_verdict),
    re-pick it (snapshots and fields under the re_ prefix). Returns (verdict dict, ok)."""
    c = scan_cal()
    hi = kin.ik_rel(SPOT_HOVER, [0, 0, 35.0])
    p, _ = kin.fk(ga.state()["present"][:5])
    if p[2] < kin.MAT_Z + 80:
        pa.move_rel(ga.state()["present"][:5], 0.0, [0, 0, kin.MAT_Z + 85 - p[2]], speed=CFG["desc"])
    if (EPIMG.get("pre") or {}).get("close") is None:  # loop restarted holding a brick: no first frame yet
        ga.move(c["q_scan"] + [0.0], speed=CFG["transit"], comp=True)
        time.sleep(0.4)
        EPIMG["pre"] = ga.snap("carry_pre")
        log("station: no first close frame this episode; took one at the scan pose (brick held)")
    ga.move(hi + [0.0], speed=CFG["transit"], comp=True)
    qd = pa.move_rel(hi, 0.0, [0, 0, -(35.0 + pa.HOVER_DZ - pa.TIP_FINAL - 1.0)], speed=CFG["desc"])
    if qd is None:  # never put down: the brick is still held, so the cycle decides on the evidence it has
        return {"verdict": "UNCERTAIN", "reasons": ["station put-down IK failed"], "method": "none"}, True
    ga.cmd({"op": "grip", "value": 25.0, "speed": 25})
    time.sleep(0.3)
    put_xy = kin.fk(qd)[0][:2]
    pa.move_rel(qd, 25.0, [0, 0, 40.0], speed=CFG["desc"])
    ga.move(c["q_scan"] + [25.0], speed=CFG["transit"], comp=True)
    time.sleep(0.4)
    imgs = ga.snap("station")
    v = station_verdict(imgs.get("close"), put_xy)
    if imgs.get("close") is not None:
        cv2.imwrite(f"{ga.OUT}/crops/station_close_full.jpg", imgs["close"])
        sl.save_crop(imgs["close"], v.get("bbox"), f"{ga.OUT}/crops/station_close.png")
    log(f"ep {k}: station close view [{v.get('method')} {v.get('changed_frac')}] {v.get('colour')} {v.get('verdict')} "
        f"{v.get('reasons')} tape {v.get('tape_score')}")
    cand = candidates(imgs["wrist"], c["q_scan"], c)
    if not cand:
        return v, False
    q, bh = approach(cand[0], c)
    if q is None:
        return v, False
    ok, info = grasp(q, bh, tag="re_")
    return v, ok


FACE_MODEL_F = os.path.join(APP, "config", "face_model.json")
ALIGN_SEARCH_PX = int(round(6 * float(srv.wc.M["reproj_median_px"])))  # 6 x the camera model's median reprojection error
SCAN_RESID_MM = None  # median residual (mm) of the current scan map, set by fit_scan_corr: neighbour position uncertainty


def survey_plan():
    try:
        with open(srv.PLAN_F) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def judge_view(img, qa, faces3d, names, model):
    """Align the projected brick to this view and judge the named faces. qa: joints the arm reached."""
    proj = srv.project_faces(qa, faces3d, ga.JAW_MASK)
    al = srv.align_shift(img, proj, ALIGN_SEARCH_PX)
    out = {}
    for f in names:
        uv = proj[f]["uv"] + (np.array(al["shift"], float) if al.get("ok") else 0.0)
        r = fcs.score(img, uv, faces3d[f]["area_mm2"], model) if model else {"verdict": "UNCERTAIN", "reason": "no face model"}
        r.update({"cos": round(proj[f]["cos"], 3), "align": {k: al.get(k) for k in ("ok", "iou", "shift", "sep")},
                  "uv": np.round(uv, 1).tolist(), "q": [round(float(v), 2) for v in qa]})
        out[f] = r
    return out


def run_views(views, faces3d, model, sdir, tag):
    """Visit the views (least wrist-roll travel first), photograph, judge. Returns {face: result}."""
    os.makedirs(sdir, exist_ok=True)
    res = {}
    for i, v in enumerate(sorted(views, key=lambda v: v["q"][4])):
        ga.move(list(v["q"]) + [25.0], speed=CFG["transit"], comp=False)
        img = pa.frame()
        qa = ga.state()["present"][:5]
        if img is None:
            res.update({f: {"verdict": "UNCERTAIN", "reason": "no frame"} for f in v["faces"]})
            continue
        cv2.imwrite(f"{sdir}/{tag}{i}.jpg", img)
        for f, r in judge_view(img, qa, faces3d, v["faces"], model).items():
            r["view"] = f"{tag}{i}"
            res[f] = r
    return res


def survey_here(k, q_hover, bh, b, cand):
    """Survey the brick where it lies, before the grasp: views from the relative library that neighbouring bricks do
    not hide, each face judged; the arm returns to the hover. Faces without a clear view stay open for the spot."""
    sp = srv.spec()
    model = fcs.load_model(FACE_MODEL_F)
    t0 = time.time()
    img0 = pa.frame()
    q0 = ga.state()["present"][:5]
    z_top = kin.MAT_Z + sp["height_mm"]
    xy0 = [float(v) for v in srv.wc.pixel_to_mat(q0, bh["centre"], z_top)]
    yaw0 = kin.psi_of(q0)
    meas = srv.measure_top(img0, q0, xy0, yaw0, sp) if img0 is not None else {"ok": False, "why": "no frame"}
    xy, yaw = (meas["xy"], meas["yaw"]) if meas.get("ok") else (xy0, yaw0)
    fp = sp["footprint_mm"]
    inflate = 2.0 * SCAN_RESID_MM if SCAN_RESID_MM else fp
    others = [{"xy": c["xy"], "yaw": 0.0, "size_mm": (fp * max(1.0, float(c.get("aspect", 1.0))),) * 2}
              for c in cand if c is not b and math.hypot(c["xy"][0] - b["xy"][0], c["xy"][1] - b["xy"][1]) > 1.0]
    lib = srv.library()
    crit = (json.load(open(srv.LIB_F)).get("criteria", {}) if lib else {})
    plan = srv.views_here(list(q_hover), xy, yaw, sp, ga.JAW_MASK, lib, others, inflate,
                          crit.get("min_cos", 0.35), crit.get("max_jaw", 0.03), crit.get("max_occ", 0.02)) if lib else \
        {"views": [], "missing": list(srv.FACES)}
    faces3d = srv.brick_faces(xy, yaw, (fp, fp), sp["height_mm"], kin.MAT_Z)
    sdir = f"{ga.OUT}/survey"
    os.makedirs(sdir, exist_ok=True)
    res = {}
    here = [v for v in plan["views"] if max(abs(a - c) for a, c in zip(v["q"], q0)) < 0.5]
    if here and img0 is not None:  # the hover itself is one of the views: no move
        cv2.imwrite(f"{sdir}/here_hover.jpg", img0)
        for f, r in judge_view(img0, q0, faces3d, here[0]["faces"], model).items():
            r["view"] = "here_hover"
            res[f] = r
    moved = [v for v in plan["views"] if not (here and v is here[0])]
    if moved:
        res.update(run_views(moved, faces3d, model, sdir, "here"))
        ga.move(list(q_hover) + [25.0], speed=CFG["transit"], comp=True)
    out = {"where": "in_place", "measure": meas, "faces": res, "missing": plan["missing"], "n_views": len(plan["views"]),
           "neighbours": len(others), "inflate_mm": round(inflate, 1), "survey_s": round(time.time() - t0, 1)}
    log(f"ep {k}: survey here {len(plan['views'])} views in {out['survey_s']} s: " +
        " ".join(f"{f}={res.get(f, {}).get('verdict', '-')}" for f in srv.FACES) + f"; missing {plan['missing']}")
    return out


def survey_inspect(k):
    """Put the held brick down alone at the planned inspection spot, judge its top and four side faces from the
    planned views, re-pick it. Returns (survey record, ok); ok is False only when the brick was put down and the
    re-pick failed (the next cycle picks it up from the spot)."""
    plan = survey_plan()
    if plan is None:
        return {"error": "no survey plan (run plan_survey.py)"}, True
    model = fcs.load_model(FACE_MODEL_F)
    sp = srv.spec()
    t0 = time.time()
    p, _ = kin.fk(ga.state()["present"][:5])
    if p[2] < kin.MAT_Z + 80:
        pa.move_rel(ga.state()["present"][:5], 0.0, [0, 0, kin.MAT_Z + 85 - p[2]], speed=CFG["desc"])
    hi = kin.ik_rel(plan["q_hover"], [0, 0, 35.0])
    if hi is None:
        return {"error": "no pose above the inspection spot"}, True
    ga.move(hi + [0.0], speed=CFG["transit"], comp=True)
    qd = pa.move_rel(hi, 0.0, [0, 0, -(35.0 + pa.HOVER_DZ - pa.TIP_FINAL - 1.0)], speed=CFG["desc"])
    if qd is None:
        return {"error": "put-down IK failed at the inspection spot"}, True  # still held
    ga.cmd({"op": "grip", "value": 25.0, "speed": 25})
    time.sleep(0.3)
    prior_xy = [float(v) for v in kin.fk(qd)[0][:2]]
    yaw0 = kin.psi_of(qd)
    pa.move_rel(qd, 25.0, [0, 0, 40.0], speed=CFG["desc"])
    top = plan["views"]["top"]
    ga.move(list(top["q"]) + [25.0], speed=CFG["transit"], comp=False)
    img = pa.frame()
    qa = ga.state()["present"][:5]
    meas = srv.measure_top(img, qa, prior_xy, yaw0, sp) if img is not None else {"ok": False, "why": "no frame"}
    xy, yaw = (meas["xy"], meas["yaw"]) if meas.get("ok") else (prior_xy, yaw0)
    fp = sp["footprint_mm"]
    faces3d = srv.brick_faces(xy, yaw, (fp, fp), sp["height_mm"], kin.MAT_Z)
    sdir = f"{ga.OUT}/survey"
    os.makedirs(sdir, exist_ok=True)
    res = {}
    if img is not None:
        cv2.imwrite(f"{sdir}/spot_top.jpg", img)
        res.update({f: dict(r, view="spot_top") for f, r in judge_view(img, qa, faces3d, ["top"], model).items()})
    res.update(run_views([{"q": plan["views"][f]["q"], "faces": [f]} for f in srv.FACES if f != "top"],
                         faces3d, model, sdir, "spot"))
    out = {"where": "spot", "plan": plan.get("created_utc"), "spot_xy": plan.get("spot_xy"), "measure": meas,
           "faces": res, "survey_s": round(time.time() - t0, 1)}
    log(f"ep {k}: survey at the spot in {out['survey_s']} s: " + " ".join(f"{f}={res.get(f, {}).get('verdict', '-')}"
                                                                         for f in srv.FACES))
    ga.move(list(plan["q_hover"]) + [25.0], speed=CFG["transit"], comp=True)
    want = EP.get("target_colour")
    bh = pa.find(pa.blobs(pa.frame(), min_area=pa.SERVO_MIN_AREA), pa.target(), 260, want)
    if bh is None:
        return out, False
    q, bh, okc = pa.servo(list(plan["q_hover"]), 25.0, bh["centre"], log=log, want=want)
    if bh is None or not okc:
        return out, False
    q, rot = pa.align_yaw(q, 25.0, bh, log=log)
    if rot:
        bh = pa.find(pa.blobs(pa.frame()), pa.target(), 260, want, short_ref=bh["short"])
        if bh is None:
            return out, False
        q, bh, okc = pa.servo(q, 25.0, bh["centre"], log=log, want=want)
        if bh is None or not okc:
            return out, False
    ok, _ = grasp(q, bh, tag="re_")
    return out, ok


def merged_faces():
    """Face verdicts of this episode: the spot survey (complete, alone) overrides the in-place one."""
    f = dict((EP.get("survey") or {}).get("faces", {}))
    f.update((EP.get("survey_spot") or {}).get("faces", {}))
    return f


def tmove(q, speed, comp=False, accel=120.0):
    st = ga.cmd({"op": "move", "q": [float(x) for x in q], "speed": speed, "comp": comp, "accel": accel})
    if st["status"] != "ok":
        raise ga.Stop(f"tmove -> {st['status']} {st['msg']}")
    return st


WP = json.load(open(f"{R}/app/motion/gm/spot_fast.yaml"))["poses"]
PLATE_IN = {"blue": [4, 3, 2, 1, 0], "green": [4, 3, 2, 1, 0]}
PLATE_OUT = {"blue": [1, 2, 3], "green": [1, 2, 3, 4]}


def place(plate):
    """Short plate path (17:45Z): straight up, then only the plate end of the validated teleop path (wp4..wp0),
    release, back up wp1..wp3/4. FK heights: blue wp4 151 mm, green wp4 226 mm above the mat."""
    st = ga.state()
    q = st["present"][:5]
    p, _ = kin.fk(q)
    if p[2] < kin.MAT_Z + 85 and q[1] > -60:
        pa.move_rel(q, 0.0, [0, 0, kin.MAT_Z + 90 - p[2]], speed=CFG["desc"] * 2)
    for i in PLATE_IN[plate]:
        tmove(WP[f"{plate}_wp{i}_closed"], CFG["transit"])
    ga.cmd({"op": "grip", "value": 25.0, "speed": 30})
    time.sleep(0.2)
    ga.snap("released")
    for i in PLATE_OUT[plate]:
        tmove(WP[f"{plate}_wp{i}"], CFG["transit"])


def run_jaw_check(move=True):
    """Jaw_check.py at the current pose (the scan pose, gripper open). Returns (exit code, result dict)."""
    args = [f"{R}/envs/data-env/bin/python", "-B", "jaw_check.py"] + ([] if move else ["--no-move"])
    r = subprocess.run(args, cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, timeout=60)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip().startswith("{")]
    return r.returncode, (json.loads(lines[-1]) if lines else {"error": (r.stderr or r.stdout)[-300:]})


def apply_jaw_calibration():
    """Use the jaw mask and servo target that jaw_check.py has just written."""
    tools = os.path.dirname(os.path.abspath(__file__))
    calib = json.load(open(f"{tools}/jaw_calib.json"))
    ga.TARGET_PX = (float(calib["target_px"][0]), float(calib["target_px"][1]))
    jm = cv2.imread(f"{tools}/wrist_jaw_mask.png", cv2.IMREAD_GRAYSCALE)
    if jm is not None:
        ga.JAW_MASK = jm > 0


def jaw_check_now(where):
    """Two-roll jaw check at the scan pose. Rigid jaws (a knocked camera moves both tips together): new mask and servo
    target. Jaws moved relative to each other (loose or bent fixed jaw): returns False. Inconclusive: keeps the target."""
    rc, d = None, {}
    for attempt in range(3):  # a stop (hand in view) during the two rolls makes the check inconclusive: wait, retry
        try:
            rc, d = run_jaw_check(move=True)
        except Exception as e:  # noqa: BLE001
            rc, d = None, {"error": repr(e)}
        if rc in (0, 3):
            break
        log(f"jaw check ({where}) inconclusive ({str(d.get('error'))[-160:]}); retry {attempt + 1}/3 after a clear workspace")
        sl.wait_clear("jaw check")
    if rc == 3:
        log(f"JAW CHECK FAILED ({where}): {d.get('error')} tips {d.get('jaw_tips')}")
        return False, d
    if rc == 0:
        old = ga.TARGET_PX
        apply_jaw_calibration()
        log(f"jaw check ({where}): tips {d.get('jaw_tips')} (n {d.get('n_measurements')}, spread {d.get('spread_px')} px, "
            f"gripper {d.get('gripper')}) gap change {d.get('gap_change_px')} px, common shift {d.get('common_shift_px')} px, "
            f"servo target {old} -> {ga.TARGET_PX}" + (f"; WARNING {d['warning']}" if d.get("warning") else ""))
        if False and os.path.exists(SCAN_PAIRS):  # disabled: plate keep-out still depends on the scan map
            dst = SCAN_PAIRS.replace(".jsonl", time.strftime("_before_%Y%m%dT%H%M%SZ.jsonl", time.gmtime()))
            os.replace(SCAN_PAIRS, dst)
            log(f"wrist camera moved: scan pairs from the old camera pose archived ({os.path.basename(dst)}); "
                "the scan map re-learns from new servo-verified pairs")
    else:
        log(f"jaw check ({where}) inconclusive: {d.get('error')}; target stays {ga.TARGET_PX}")
    return True, d


def periodic_jaw_check(k, every=5):
    """Every `every` cycles (at the scan pose, before the scan): the start-up jaw check again. A loose or bent fixed
    jaw pauses the loop with a clear message (the fixed jaw works loose when it touches the ground)."""
    if k % every:
        return
    ok, d = jaw_check_now(f"cycle {k}")
    EP["jaw_check"] = d
    if not ok:
        open(f"{ga.RUN_ROOT}/sort_pause", "w").write(
            f"{time.strftime('%H:%M:%SZ', time.gmtime())} jaw check failed: {d.get('error')}\n")
        log("pausing: tighten the fixed jaw, then delete run/sort_pause")


def cycle(k):
    global EP
    EP = {"episode": k, "run": os.path.basename(RUN), "mode": "pick_anywhere", "truth": CFG.get("collect")}
    ep_dir = f"{RUN}/ep_{k:04d}"
    os.makedirs(f"{ep_dir}/crops", exist_ok=True)
    ga.OUT = ep_dir
    ga.SUM.clear()
    ga.SUM["phases"] = []
    T = {}
    fit_scan_corr()
    periodic_jaw_check(k)
    b, cand, waited = wait_for_bricks()
    T["wait_s"] = waited
    c = scan_cal()
    EP.update({"target_scan_blob": b, "n_candidates": len(cand), "brick_xy_est": b.get("xy")})
    EPIMG.clear()
    EPIMG["pre"] = ga.snap("pre")
    log(f"ep {k}: {len(cand)} brick(s) in view; going for {b['centre']} est {b['xy']} mm (d {b['d_mm']})")
    ga.cmd({"op": "rec_start", "path": f"{ep_dir}/joints.csv"})
    EP["t_start"] = time.time()
    t0 = time.time()
    q, bh = approach(b, c)
    if q is None:
        EP.update({"success": False, "failure": bh})
        FAILED_XY.append((b["xy"][0], b["xy"][1], time.time()))
        log(f"ep {k}: APPROACH FAILED: {bh} (brick at {b['xy']} skipped for 5 min)")
        go_scan(False)
        ga.cmd({"op": "rec_stop"})
        return
    EP["hover_q"], EP["hover_blob"] = q, bh
    EP["survey"] = survey_here(k, q, bh, b, cand)
    ok, info = grasp(q, bh)
    if not ok and str(info).startswith("nothing held"):
        EP["grasp_miss_1"] = info
        log(f"ep {k}: grasp missed ({info}); re-centring at hover and retrying lower")
        st = ga.state()
        qr = pa.move_rel(st["present"][:5], 25.0, [0.0, 0.0, pa.HOVER_DZ - (pa.TIP_FINAL + 45.0)], speed=CFG["hover"])
        want = EP.get("target_colour")
        b2 = pa.find(pa.blobs(pa.frame(), min_area=pa.SERVO_MIN_AREA), pa.target(), 260, want) if qr is not None else None
        if b2 is not None:
            qr, b2, okc = pa.servo(qr, 25.0, b2["centre"], log=log, want=want)
            if b2 is not None and okc:
                ok, info = grasp(qr, b2, retry=True)
                EP["grasp_retry_ok"] = ok
    T["pick_s"] = round(time.time() - t0, 1)
    if not ok:
        EP.update({"success": False, "failure": info})
        log(f"ep {k}: GRASP FAILED: {info}")
        ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
        go_scan(False)
        ga.cmd({"op": "rec_stop"})
        return
    shape = info
    t1 = time.time()
    pre = EP.get("pre_close") or {}
    views = []
    have = merged_faces()
    if shape[0] == "2x2" and pre.get("verdict") != "CONFIDENT_DEFECT" and \
            not any(r.get("verdict") == "DEFECT" for r in have.values()) and \
            any(have.get(f, {}).get("verdict") != "CLEAN" for f in srv.FACES):
        surv, ok2 = survey_inspect(k)
        EP["survey_spot"] = surv
        if not ok2:
            EP.update({"success": False, "failure": "re-pick at the inspection spot failed"})
            log(f"ep {k}: re-pick at the inspection spot failed")
            go_scan(False)
            ga.cmd({"op": "rec_stop"})
            return
    T["inspect_s"] = round(time.time() - t1, 1)
    pre = EP.get("pre_close") or {"verdict": "UNCERTAIN"}
    colour = pre.get("colour") or EP.get("target_colour") or "unknown"
    plate, label, conf, reasons = sl.decide_faces(pre, shape, merged_faces(), srv.FACES)
    EP.update({"views": views, "colour": colour, "plate": plate, "label": label, "decision_conf": conf,
               "decision_reasons": reasons, "label_source": "det2", "needs_review": conf not in ("confident", "held_only"),
               "task": f"move the {colour} brick to the {plate} plate"})
    log(f"ep {k}: {colour} {shape[0]} -> {label} -> {plate} ({conf}; {reasons})")
    t2 = time.time()
    place(plate)
    T["place_s"] = round(time.time() - t2, 1)
    t3 = time.time()
    go_scan(False)
    T["return_s"] = round(time.time() - t3, 1)
    EP["t_end"] = time.time()
    ga.cmd({"op": "rec_stop"})
    T["cycle_s"] = round(EP["t_end"] - t0, 1)
    EP.update({"success": True, "failure": None, "timings": T})
    log(f"ep {k}: DONE {label} -> {plate} in {T['cycle_s']} s {T}")


def save_ep(k):
    if not EP:
        return
    if pa.LOST_LOG:  # searches that had to look beyond the radius during this episode
        EP["search_beyond_radius"] = list(pa.LOST_LOG)
        pa.LOST_LOG.clear()
    EP["t_end"] = EP.get("t_end") or time.time()
    mon = f"{ga.LOG_ROOT}/monitor"
    names = sorted(os.listdir(mon)) if os.path.isdir(mon) else []
    newest = {v: ([x for x in names if x.startswith(v + "_") and x.endswith(".mkv")] or [None])[-1] for v in ("top", "close")}
    EP["cams"] = {"top": newest["top"] and "logs/monitor/" + newest["top"],
                  "close": newest["close"] and "logs/monitor/" + newest["close"],
                  "wrist_dir": "data/wrist/raw"}
    os.makedirs(f"{RUN}/ep_{k:04d}", exist_ok=True)
    json.dump(EP, open(f"{RUN}/ep_{k:04d}/episode.json", "w"), indent=1, default=str)
    with open(f"{RUN}/cycles.jsonl", "a") as f:
        f.write(json.dumps({x: EP.get(x) for x in ("episode", "t_start", "t_end", "task", "plate", "label", "decision_conf",
                                                  "decision_reasons", "colour", "shape", "success", "failure",
                                                  "grip_closed", "timings", "brick_xy_est", "mode")}, default=str) + "\n")


def clean_views(views):
    return [v for v in views if v.get("verdict") != "NOT_USED"
            and not any(str(r).startswith(("merged", "colour_split")) for r in v.get("reasons", []))]


def finish_held():
    """A brick is held: inspect (if no decision yet), decide, place."""
    plate = EP.get("plate")
    if not plate:
        views = []
        have = merged_faces()
        if not any(r.get("verdict") == "DEFECT" for r in have.values()) and \
                any(have.get(f, {}).get("verdict") != "CLEAN" for f in srv.FACES):
            surv, ok2 = survey_inspect(EP.get("episode", 0))
            EP["survey_spot"] = surv
            if not ok2:
                log("held brick put down at the inspection spot; re-pick failed - the next cycle picks it up")
                return None
        pre = EP.get("pre_close") or {"verdict": "UNCERTAIN"}
        shape = (EP.get("shape") or EP.get("re_shape") or "unknown", EP.get("shape_why") or EP.get("re_shape_why") or "")
        plate, label, conf, reasons = sl.decide_faces(pre, shape, merged_faces(), srv.FACES)
        colour = pre.get("colour") or EP.get("target_colour") or "unknown"
        EP.update({"views": views, "plate": plate, "label": label, "decision_conf": conf, "colour": colour,
                   "decision_reasons": reasons, "label_source": "det2", "needs_review": conf not in ("confident", "held_only"), "task": f"move the {colour} brick to the {plate} plate"})
        log(f"held brick -> {label} {plate} ({conf}; {reasons})")
    place(plate)
    return plate


def recover():
    """After an estop: wait until the workspace is clear, finish a held brick or open, then go to the scan pose.
    Returns False after 40 failed attempts: writes RUN/needs_operator and logs why. The caller must stop the loop."""
    last_error = "no attempt"
    for attempt in range(40):
        try:
            ga.cmd({"op": "rec_stop"})
            sl.wait_clear("estop")
            st = ga.state()
            if st["target"][5] < 5 and 0.8 <= st["present"][5] <= 8:
                plate = finish_held()
                EP["failure"] = (EP.get("failure") or "") + f" (recovered: {'placed on ' + plate if plate else 'left at station'})"
            else:
                ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
            go_scan(False)
            return True
        except ga.Stop as e:  # noqa: PERF203
            last_error = str(e)
            pause = min(2.0 ** min(attempt, 5), 30.0)  # 18:03Z: 40 retries burned in 1 s; now ~15 min before giving up
            log(f"recovery interrupted ({e}); retry {attempt + 1} in {pause:.0f} s")
            time.sleep(pause)
    reason = f"recovery gave up after 40 attempts (last error: {last_error})"
    write_needs_operator(reason)
    log(f"needs operator: {reason}")
    return False


def write_needs_operator(reason):
    """Operator marker in the run folder: utc, episode and last failure. The operator deletes it after the fix."""
    with open(f"{RUN}/needs_operator", "w") as f:
        f.write(f"utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        f.write(f"episode: {EP.get('episode', 'start-up')}\n")
        f.write(f"last failure: {EP.get('failure') or 'none recorded'}\n")
        f.write(f"reason: {reason}\n")


def run_one(k):
    try:
        cycle(k)
    except SystemExit as e:
        log(f"stopping: {e}")
        save_ep(k)
        return None
    except ga.Stop as e:
        EP.update({"success": False, "failure": f"stop: {e}"})
        log(f"ep {k}: STOP {e}; recovering")
        if not recover():
            save_ep(k)
            return None
    except Exception as e:  # noqa: BLE001
        EP.update({"success": False, "failure": f"exception {e!r}"})
        log(f"ep {k}: EXCEPTION {traceback.format_exc()[-900:]}")
        try:
            ga.cmd({"op": "rec_stop"})
            if not (ga.state()["target"][5] < 5 and 0.8 <= ga.state()["present"][5] <= 8):
                ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
                go_scan(False)
            else:
                finish_held()
                go_scan(False)
        except ga.Stop:
            if not recover():
                save_ep(k)
                return None
    save_ep(k)
    return k + 1


def main():
    global RUN, EP
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--collect", choices=("GOOD", "DEFECT"), default=None,
                    help="calibration run: every brick on the mat is of this kind (truth in episode.json for fit_faces.py)")
    ap.add_argument("--profile", choices=("safe", "fast"), default="safe",
                    help="fast = untested speed-ups (transit 45, hover 25, desc 14, final 6, servo 18); try it on a few "
                         "cycles and compare cycles.jsonl timings before using it for collection")
    a = ap.parse_args()
    RUN = f"{ga.LOG_ROOT}/sort/{a.run}"
    sl.RUN = RUN
    os.makedirs(RUN, exist_ok=True)
    if a.profile == "fast":
        CFG.update({"transit": 45.0, "hover": 25.0, "desc": 14.0, "final": 6.0, "servo": 18.0})
        pa.SERVO_SPEED = 18.0
    for key in ("transit", "hover", "servo", "desc"):
        CFG[key] = min(45.0, CFG[key] * a.speed)
    CFG["profile"] = a.profile
    CFG["collect"] = a.collect
    if os.path.exists(f"{ga.RUN_ROOT}/sort_stop"):
        os.remove(f"{ga.RUN_ROOT}/sort_stop")
    log(f"start {a.run} n={a.n} cfg={CFG} target={ga.TARGET_PX}")
    sl.wait_clear("start-up")  # no motion before a live hand watchdog and a clear workspace
    st0 = ga.state()
    if st0["target"][5] < 5 and 0.8 <= st0["present"][5] <= 8:
        k0 = a.start - 1
        EP = {"episode": k0, "run": os.path.basename(RUN), "mode": "carry_over", "t_start": time.time()}
        os.makedirs(f"{RUN}/ep_{k0:04d}/crops", exist_ok=True)
        ga.OUT = f"{RUN}/ep_{k0:04d}"
        log(f"holding a brick at start (gripper {st0['present'][5]}): inspect, decide, place")
        carried = True
        try:
            finish_held()
            EP["success"] = True
        except ga.Stop as e:
            log(f"carry-over interrupted ({e})")
            carried = recover()
        save_ep(k0)
        if not carried:
            return  # recover() wrote needs_operator and logged why: the cycle loop does not start
    try:
        go_scan(False)
    except ga.Stop:
        if not recover():
            return  # start-up recover() gave up: the cycle loop does not start
    ok, d = jaw_check_now("start-up")  # at the scan pose with the gripper open
    if not ok:
        write_needs_operator(f"jaw check: {d.get('error')}")
        return
    k = a.start
    while k is not None and k < a.start + a.n:
        k = run_one(k)
    log("loop end")


if __name__ == "__main__":
    main()
