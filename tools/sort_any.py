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
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grasp_auto as ga  # noqa: E402
import kin  # noqa: E402
import pick_any as pa  # noqa: E402
import sort_loop as sl  # noqa: E402  (workspace-clear rule, log, crops, decide_faces)
import defects as dfx  # noqa: E402
import survey as srv  # noqa: E402
import scene  # noqa: E402
import underside as us  # noqa: E402
import closecam as cc  # noqa: E402
from paths import APP  # noqa: E402

R = ga.R
SPOT_HOVER = list(kin.cell.get("spot_hover_q"))
PLATES_F = os.path.join(ga.RUN_ROOT, "plates_seen.json")  # where each plate was last seen, and bricks placed on it
CFG = {"transit": 42.0, "hover": 18.0, "servo": 12.0, "desc": 10.0, "final": 4.0, "grip": 20.0}  # set by --profile
PROFILES = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                                       "motion_profiles.json")))
EP = {}
EPIMG = {}  # this episode's first frames, taken at the scan pose with the arm out of the close view
RUN = ""


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


SCAN_PAIRS = f"{ga.RUN_ROOT}/scan_pairs.jsonl" if ga.SIM_ROOT else os.path.join(APP, "tools", "scan_pairs.jsonl")  # (scan px, servo-verified hover xy) from every centred approach
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
    """Base-frame offset (mm) of a brick seen from the scan pose relative to the point under the jaws: through the
    wrist camera model when there is one (it is re-fitted whenever the camera moves on the gripper, while the learned
    scan map below keeps pairs from before such a move), else the scan Jacobian with the learned correction."""
    if srv.wc.M is not None:
        qd = list(np.asarray(list(q_scan)[:5], float) + srv.wc._DQ)
        p, _ = kin.fk(qd)
        xy = srv.wc.pixel_to_mat(q_scan, b["centre"])
        return np.array([xy[0] - p[0], xy[1] - p[1]])
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




def plates_seen():
    try:
        return json.load(open(PLATES_F))
    except (OSError, ValueError):
        return {"plates": {}, "placed": {}}


def see_plates(img, q_scan):
    """The plates in this scan frame (scene.plates: by colour, and by shape when the rim shows enough of a circle),
    merged into the record of where each plate was last measured. Returns (plates measured now or before:
    {colour: {"xy", "r_mm", ...}}, the plate pixels of this frame)."""
    rec = plates_seen()
    try:
        now = scene.plates(img, q_scan, srv.spec(), ignore=pa.jaw_mask(img))
    except Exception as e:  # noqa: BLE001
        log(f"plate detection failed: {e!r}")
        now = {}
    pix = np.zeros(img.shape[:2], bool)
    changed = False
    for k, v in now.items():
        pix |= v["mask"]
        if v["reliable"]:
            rec["plates"][k] = {kk: vv for kk, vv in v.items() if kk != "mask"}
            rec["plates"][k]["t"] = time.time()
            changed = True
    if changed:
        with open(PLATES_F + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(PLATES_F + ".tmp", PLATES_F)
    return rec["plates"], pix


def near_plate(xy, plates, sp):
    """A brick at xy lies on or against a measured plate (inside its radius plus half the brick's diagonal)."""
    reach = sp["footprint_mm"] * math.sqrt(0.5)
    return any(math.hypot(xy[0] - v["xy"][0], xy[1] - v["xy"][1]) < v["r_mm"] + reach for v in plates.values())


def candidates(img, q_scan, c):
    out = []
    p, _ = kin.fk(q_scan)
    sp = srv.spec()
    plates, pix = see_plates(img, q_scan)
    slack = int(round(0.25 * math.sqrt(max(1.0, scene.brick_area_px(q_scan, p[:2], sp)))))  # a quarter brick, in px
    pix = cv2.dilate(pix.astype(np.uint8), np.ones((2 * slack + 1, 2 * slack + 1), np.uint8)) > 0
    for b in pa.blobs(img, min_area=700, max_area=40000):
        if b["plate_frac"] >= 0.3 or (b.get("colour") in ("blue", "green") and b["plate_frac"] >= 0.15):
            continue  # inside a plate, or a plate rim fragment
        cx, cy = int(round(b["centre"][0])), int(round(b["centre"][1]))
        if 0 <= cy < pix.shape[0] and 0 <= cx < pix.shape[1] and pix[cy, cx]:
            continue  # on a plate seen in this frame
        d = brick_xy_from_scan(q_scan, b, c)
        xy = (p[0] + d[0], p[1] + d[1])
        if near_plate(xy, plates, sp):
            continue
        r = math.hypot(xy[0] - kin.PAN_XY_MM[0], xy[1] - kin.PAN_XY_MM[1])
        r_lo, r_hi = kin.cell.get("reach_r_mm")
        if r < r_lo or r > r_hi or not grasp_reachable(q_scan, d):  # at the base, or out of reach
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


MIN_TIP_MM = kin.cell.get("min_tip_mm")  # lowest jaw-tip height above the mat (FK) any descent may reach
STAGE_MM = 26.0  # the grasp descent is fast down to jaw tips this high above the mat (above a 19.2 mm brick), slow after
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
    q_pre = list(q)
    q, rot = pa.align_yaw(q, g, bh, log=log)
    if rot:
        bh2 = pa.find(pa.blobs(pa.frame(), min_area=pa.SERVO_MIN_AREA), pa.target(), 400, want)
        if bh2 is None:  # the roll swung the brick out of sight or under a jaw: undo it and grasp unaligned
            log("brick not found after the yaw roll: back to the previous roll, grasping unaligned")
            ga.cmd({"op": "move", "q": q_pre + [g], "speed": max(25.0, pa.SERVO_SPEED), "comp": True,
                    "accel": max(120.0, ga.ACCEL)})
            q = q_pre
            bh2 = pa.find(pa.blobs(pa.frame(), min_area=pa.SERVO_MIN_AREA), pa.target(), 260, want)
            if bh2 is None:
                return None, "brick lost after yaw"
        q, bh, ok = pa.servo(q, g, bh2["centre"], log=log, want=want)
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


def grasp(q, bh, retry=False, tag=""):
    """Descend on the centred brick to the grasp height above the mat (FK), close, lift, check the hold (gripper
    reading against the cell calibration). tag="re_" for the re-pick at the inspection spot: its snapshots and fields
    get that prefix, so the first pick's record stays intact."""
    g = 25.0
    fk_tip = float(kin.fk(q)[0][2] - kin.MAT_Z)  # jaw tips above the mat at the hover (FK)
    end = max(pa.TIP_FINAL - (2.0 if retry else 0.0), MIN_TIP_MM)  # a retry (the first close met nothing) goes lower
    d1 = min(0.0, max(STAGE_MM, end) - fk_tip)  # fast down to the stage height, never up
    d2 = min(0.0, end - (fk_tip + d1))  # slow for the rest
    EP[tag + "grasp_end_mm" + ("_retry" if retry else "")] = round(end, 1)
    cv2.imwrite(f"{ga.OUT}/crops/{tag}wrist_hover.png", pa.frame())
    cimg = ga.fresh("close", time.time() + 0.2)
    if cimg is not None:
        cv2.imwrite(f"{ga.OUT}/crops/{tag}hover_close.jpg", cimg)
    q1 = pa.move_rel(q, g, [0, 0, d1], speed=CFG["desc"]) if d1 < 0 else list(q)
    if q1 is None:
        return False, "ik fail stage 1"
    b1 = pa.nearest(pa.blobs(pa.frame(), max_area=90000), pa.target(), max_d=260)
    ga.snap_async(tag + "stage1")
    q2 = pa.move_rel(q1, g, [0, 0, d2], speed=CFG["final"]) if d2 < 0 else list(q1)
    if q2 is None:
        return False, "ik fail final descent"
    ga.snap_async(tag + "grasp_open")
    ga.cmd({"op": "grip", "value": 0.0, "speed": CFG["grip"]})
    gr = ga.wait_grip()
    EP[tag + "grip_closed"] = gr
    ga.snap_async(tag + "closed")
    ql = pa.move_rel(q2, 0.0, [0, 0, 45.0], speed=CFG["desc"])
    if ql is None:
        return False, "ik fail lift"
    ga.snap_async(tag + "lifted")
    gr2 = ga.state()["present"][5]
    EP.update({tag + "grip_lifted": gr2, tag + "lift_q": ql, tag + "stage1_blob": b1})
    if gr2 < kin.cell.held_threshold():
        ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
        return False, f"nothing held (gripper {gr2} < {kin.cell.held_threshold():.2f})"
    return True, "held"




ALIGN_SEARCH_PX = int(round(float(dfx.config().get("align_search_factor", 6.0)) * float(srv.wc.M["reproj_median_px"])))  # 6 x the camera model's median reprojection error
SCAN_RESID_MM = None  # median residual (mm) of the current scan map, set by fit_scan_corr: brick position uncertainty


def survey_plan():
    try:
        with open(srv.PLAN_F) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pos_sigma_mm(sp):
    """Position uncertainty of a brick seen at the scan pose: the scan map's median residual; one stud pitch before
    the first fit."""
    return SCAN_RESID_MM if SCAN_RESID_MM else sp["footprint_mm"] / sp["studs"][0]


def keep_frame(img, path):
    """Write a survey frame as JPEG and return exactly what was written: the inspection judges the same pixels that
    calibrate_defects.py and calib_wristcam.py replay later."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return img
    buf.tofile(path)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def run_views(views, sdir, tag, q_from, q_to):
    """Visit the views in the order with the least joint travel and photograph each (joints read back at the frame).
    Returns the view records for the inspection."""
    os.makedirs(sdir, exist_ok=True)
    out = []
    for i, v in enumerate(srv.order_views(q_from, views, q_to)):
        ga.move(list(v["q"]) + [25.0], speed=CFG["transit"], comp=False)
        if CFG.get("settle_s"):  # let the camera stop moving after a fast stop before the photograph
            time.sleep(CFG["settle_s"])
        img = pa.frame()
        qa = ga.state()["present"][:5]
        if img is None:
            continue
        name = f"{tag}{i}"
        out.append({"img": keep_frame(img, f"{sdir}/{name}.jpg"), "q": [float(x) for x in qa], "name": name,
                    "faces": list(v["faces"])})
    return out


def survey_views(q_hover, meas, sp, model, sdir, tag):
    """Plan (library views at this brick's pose), photograph and inspect a measured brick from the hover above it. The
    hover frame the brick was measured in is a view too. All views are judged together (defects.inspect). Returns
    (face results, plan, inspection summary)."""
    fp = sp["footprint_mm"]
    faces3d = srv.brick_faces(meas["xy"], meas["yaw"], (fp, fp), sp["height_mm"], kin.MAT_Z)
    lib = srv.library()
    crit = json.load(open(srv.LIB_F)).get("criteria", {}) if lib else {}
    plan = srv.views_here(list(q_hover), meas["xy"], meas["yaw"], sp, ga.JAW_MASK, lib, [], 0.0,
                          crit.get("min_cos", 0.35), crit.get("max_jaw", 0.03), crit.get("max_occ", 0.02)) if lib else \
        {"views": [], "missing": list(srv.FACES)}
    os.makedirs(sdir, exist_ok=True)
    views = []
    if meas.get("img") is not None:
        views.append({"img": keep_frame(meas["img"], f"{sdir}/{tag}_hover.jpg"), "q": [float(x) for x in meas["q"]],
                      "name": f"{tag}_hover", "faces": ["top"]})
    rest = [v for v in plan["views"] if max(abs(a - c) for a, c in zip(v["q"], q_hover)) >= 0.5]
    if rest:
        views += run_views(rest, sdir, tag, q_hover, q_hover)
    with ThreadPoolExecutor(1) as ex:  # the inspection runs while the arm goes back to the hover
        job = ex.submit(dfx.inspect, views, faces3d, model, None, ALIGN_SEARCH_PX, ga.JAW_MASK)
        if rest:
            ga.move(list(q_hover) + [25.0], speed=CFG["transit"], comp=True)
        res, summ = job.result()
    summ["views"] = [{"name": v["name"], "q": v["q"], "faces": v["faces"]} for v in views]
    return res, plan, summ


def measure_here(q_prior_xy, yaw0, sp):
    """Top view from the current pose: frame, joints, brick measurement (survey.measure_top)."""
    img = pa.frame()
    q = ga.state()["present"][:5]
    m = srv.measure_top(img, q, q_prior_xy, yaw0, sp) if img is not None else {"ok": False, "why": "no frame"}
    m.update({"img": img, "q": q})
    return m


def lowest_view_elevation(views, xy, sp):
    """Smallest elevation (rad) of the camera above the brick top over the planned views."""
    z_top = kin.MAT_Z + sp["height_mm"]
    el = []
    for v in views:
        _, cb = srv.wc.camera_pose(v["q"])
        el.append(math.atan2(cb[2] - z_top, math.hypot(cb[0] - xy[0], cb[1] - xy[1])))
    return min(el) if el else math.pi / 2


def survey_here(k, q_hover, bh, b, cand):
    """Survey the brick where it lies, before the grasp, when nothing else stands close enough to hide a face in the
    planned views (checked on this cycle's scan image: any object, not only pickable bricks). The arm returns to
    the hover. Faces left open are surveyed at the inspection spot after the grasp."""
    sp = srv.spec()
    model = dfx.load_model()
    t0 = time.time()
    q0 = ga.state()["present"][:5]
    z_top = kin.MAT_Z + sp["height_mm"]
    xy0 = [float(v) for v in srv.wc.pixel_to_mat(q0, bh["centre"], z_top)]
    meas = measure_here(xy0, kin.psi_of(q0), sp)
    out = {"where": "in_place", "measure": {k_: v for k_, v in meas.items() if k_ not in ("img",)}, "faces": {}}
    if not meas.get("ok"):
        out.update(missing=list(srv.FACES), why=f"measurement: {meas.get('why')}")
        log(f"ep {k}: survey here skipped ({out['why']})")
        return out
    out["shape"] = meas["shape"]
    sdir = f"{ga.OUT}/survey"
    os.makedirs(sdir, exist_ok=True)
    lib = srv.library()
    crit = json.load(open(srv.LIB_F)).get("criteria", {}) if lib else {}
    plan = srv.views_here(list(q_hover), meas["xy"], meas["yaw"], sp, ga.JAW_MASK, lib, [], 0.0,
                          crit.get("min_cos", 0.35), crit.get("max_jaw", 0.03), crit.get("max_occ", 0.02)) if lib else \
        {"views": [], "missing": list(srv.FACES)}
    sig = pos_sigma_mm(sp)
    radius = sp["footprint_mm"] / 2 + sp["height_mm"] / math.tan(max(lowest_view_elevation(plan["views"], meas["xy"], sp),
                                                                     1e-3)) + 2 * sig
    scan = (EPIMG.get("pre") or {}).get("wrist")
    iso = scene.isolated(scan, scan_cal()["q_scan"], meas["xy"], meas["yaw"], sp, radius, 2 * sig, pa.jaw_mask(scan)) \
        if scan is not None else \
        {"isolated": False, "why": "no scan image"}
    out["isolation"] = iso
    if not iso.get("isolated"):
        out.update(missing=list(srv.FACES), why="not alone on the mat: surveyed at the inspection spot")
        log(f"ep {k}: survey here skipped: objects within {iso.get('radius_mm')} mm ({iso.get('objects_px')})")
        return out
    res, plan, summ = survey_views(q_hover, meas, sp, model, sdir, "here")
    out.update(faces=res, missing=plan["missing"], n_views=len(summ["views"]), inspection=summ,
               survey_s=round(time.time() - t0, 1))
    log(f"ep {k}: survey here {len(plan['views'])} views in {out['survey_s']} s, {meas['shape'][1]}: " +
        " ".join(f"{f}={res.get(f, {}).get('verdict', '-')}" for f in srv.FACES))
    return out


def free_spot(sp):
    """The first planned inspection spot that this cycle's scan image shows free (anything on it counts)."""
    plan = survey_plan()
    scan = (EPIMG.get("pre") or {}).get("wrist")
    if plan is None or scan is None:
        return None, "no plan or no scan image"
    q_scan = scan_cal()["q_scan"]
    spots = [plan["spot_xy"]] + [s[1:] for s in plan.get("ranked_spots", []) if s[1:] != plan["spot_xy"]]
    plates = plates_seen().get("plates", {})
    for xy in spots:
        if any(math.hypot(xy[0] - v["xy"][0], xy[1] - v["xy"][1]) < v["r_mm"] + 2.0 * sp["footprint_mm"]
               for v in plates.values()):
            continue  # the survey views around a brick this close to a plate would see the plate as a neighbour
        fr = scene.spot_free(scan, q_scan, xy, sp, 2 * pos_sigma_mm(sp), pa.jaw_mask(scan))
        if fr.get("free"):
            q_h = kin.ik(xy[0], xy[1], kin.MAT_Z + pa.HOVER_DZ, kin.psi_of(plan["q_hover"]), plan["q_hover"])
            if q_h is not None:
                return {"xy": xy, "q_hover": q_h}, None
    return None, "every planned spot is occupied or out of view"


def survey_inspect(k, roll=0.0, repick=True):
    """Put the held brick down alone at a free inspection spot, measure it from the hover above, survey every face,
    re-pick it. roll: wrist roll (joint units) added before the put-down, to set the brick down turned (calibration
    runs vary the pose this way); repick False leaves the brick at the spot. Returns (survey record, ok); ok is False
    only when the brick was put down and the re-pick failed (the next cycle picks it up from the spot)."""
    sp = srv.spec()
    model = dfx.load_model()
    t0 = time.time()
    spot, why = free_spot(sp)
    if spot is None:
        return {"error": why}, True  # still held: the decision goes on the evidence it has
    p, _ = kin.fk(ga.state()["present"][:5])
    if p[2] < kin.MAT_Z + 80:
        pa.move_rel(ga.state()["present"][:5], 0.0, [0, 0, kin.MAT_Z + 85 - p[2]], speed=CFG["desc"])
    hi = kin.ik_rel(spot["q_hover"], [0, 0, 35.0])
    if hi is None:
        return {"error": "no pose above the inspection spot"}, True
    lo_b, hi_b = kin.CAL.norm_bounds(3)
    if roll and lo_b[4] <= hi[4] + roll <= hi_b[4]:
        hi = list(hi)
        hi[4] += roll
    ga.move(hi + [0.0], speed=CFG["transit"], comp=True)
    qd = pa.move_rel(hi, 0.0, [0, 0, -(35.0 + pa.HOVER_DZ - pa.TIP_FINAL - 1.0)], speed=CFG["desc"])
    if qd is None:
        return {"error": "put-down IK failed at the inspection spot"}, True  # still held
    ga.cmd({"op": "grip", "value": 25.0, "speed": max(25.0, CFG["grip"])})
    time.sleep(0.3)
    prior = [float(v) for v in kin.fk(qd)[0][:2]]
    yaw0 = kin.psi_of(qd)
    pa.move_rel(qd, 25.0, [0, 0, 40.0], speed=CFG["desc"])
    ga.move(list(spot["q_hover"]) + [25.0], speed=CFG["transit"], comp=True)
    meas = measure_here(prior, yaw0, sp)
    out = {"where": "spot", "spot_xy": spot["xy"], "measure": {k_: v for k_, v in meas.items() if k_ != "img"}, "faces": {}}
    if meas.get("ok"):
        out["shape"] = meas["shape"]
        res, plan, summ = survey_views(spot["q_hover"], meas, sp, model, f"{ga.OUT}/survey", "spot")
        out.update(faces=res, missing=plan["missing"], n_views=len(summ["views"]), inspection=summ)
    else:
        out["why"] = f"measurement at the spot: {meas.get('why')}"
    out["survey_s"] = round(time.time() - t0, 1)
    log(f"ep {k}: survey at the spot in {out['survey_s']} s, {out.get('shape', ('?', meas.get('why')))[1]}: " +
        " ".join(f"{f}={out['faces'].get(f, {}).get('verdict', '-')}" for f in srv.FACES))
    if not repick:
        out["left_at_spot"] = True
        return out, True
    if meas.get("ok"):  # the brick's pose was just measured: straight to the hover above it, jaws along its faces
        qd = hover_over(meas["xy"], meas["yaw"], spot["q_hover"])
        if qd is not None:
            ga.move(list(qd) + [25.0], speed=CFG["hover"], comp=True)
            ok, info = grasp(qd, {}, tag="re_")
            out["regrasp"] = {"how": "measured pose", "ok": ok, "info": info}
            if ok:
                return out, ok
            qn = ga.state()["present"][:5]
            low = kin.MAT_Z + pa.HOVER_DZ - float(kin.fk(qn)[0][2])
            if low > 0:  # straight up before any joint-space move
                pa.move_rel(qn, 25.0, [0, 0, low], speed=CFG["hover"])
            ga.move(list(spot["q_hover"]) + [25.0], speed=CFG["transit"], comp=True)
    want = EP.get("target_colour")
    bh = pa.find(pa.blobs(pa.frame(), min_area=pa.SERVO_MIN_AREA), pa.target(), 260, want)
    if bh is None:
        return out, False
    q, bh, okc = pa.servo(list(spot["q_hover"]), 25.0, bh["centre"], log=log, want=want)
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


def underside_record():
    """Show the held brick's bottom to the close camera (underside.inspect), keep the frame, record the result."""
    try:
        rec = us.inspect(ga, srv.spec(), log)
    except Exception as e:  # noqa: BLE001
        rec = {"verdict": "UNCERTAIN", "reason": f"underside inspection failed: {e!r}"}
    img = rec.pop("_img", None)
    if img is not None:
        cv2.imwrite(f"{ga.OUT}/underside.jpg", img)
    EP["underside"] = rec
    return rec


def decide_now(shape):
    """OD-10 on this episode's evidence: the surveyed faces, and the bottom when the underside inspection is on (it
    runs only when nothing has decided green already)."""
    faces, required = merged_faces(), tuple(srv.FACES)
    if us.enabled():
        required += ("bottom",)
        if shape[0] == "2x2" and not any(r.get("verdict") == "DEFECT" for r in faces.values()):
            faces["bottom"] = underside_record()
    return sl.decide_faces({}, shape, faces, required)


def hover_over(xy, yaw, q_ref):
    """Hover pose (top-down IK) above a brick at xy with the jaws along its faces: of the four yaws that do that, the
    one nearest the reference pose's."""
    psi0 = kin.psi_of(q_ref)
    psi = min((yaw + k * math.pi / 2 for k in range(-2, 3)), key=lambda a: abs(math.remainder(a - psi0, 2 * math.pi)))
    return kin.ik(float(xy[0]), float(xy[1]), kin.MAT_Z + pa.HOVER_DZ, psi, q_ref)


def merged_faces():
    """Face verdicts of this episode for the decision. Each survey judged the brick as it lay then, and face names of
    two surveys need not name the same physical face (the brick may be gripped turned), so one survey's faces are
    used together: a survey that found a DEFECT stands; otherwise the survey with the most CLEAN faces counts."""
    surveys = [(EP.get(k) or {}).get("faces") or {} for k in ("survey_spot", "survey")]
    for f in surveys:
        if any(r.get("verdict") == "DEFECT" for r in f.values()):
            return dict(f)
    return dict(max(surveys, key=lambda f: sum(r.get("verdict") == "CLEAN" for r in f.values())))


def survey_shape():
    """The brick's shape from the most reliable measurement of this episode (the spot, then in place)."""
    for key in ("survey_spot", "survey"):
        s = (EP.get(key) or {}).get("shape")
        if s:
            return tuple(s)
    return ("unknown", "no valid footprint measurement")


def tmove(q, speed, comp=False, accel=None):
    accel = max(120.0, ga.ACCEL) if accel is None else accel
    st = ga.cmd({"op": "move", "q": [float(x) for x in q], "speed": speed, "comp": comp, "accel": accel})
    if st["status"] != "ok":
        raise ga.Stop(f"tmove -> {st['status']} {st['msg']}")
    return st


WP = json.load(open(os.path.join(APP, "motion", "gm", "spot_fast.yaml")))["poses"]
PLATE_IN = {"blue": [4, 3, 2, 1, 0], "green": [4, 3, 2, 1, 0]}
PLATE_OUT = {"blue": [1, 2, 3], "green": [1, 2, 3, 4]}


def drop_points(plate):
    """Release points over a known plate, spread like sunflower seeds within the radius that keeps a brick on it, in
    the order the next bricks use them (bricks placed earlier this run are not dropped on)."""
    rec = plates_seen()
    pl = rec["plates"].get(plate)
    if not pl:
        return []
    sp = srv.spec()
    rho = max(0.0, pl["r_mm"] - 0.75 * sp["footprint_mm"])
    k0 = int(rec.get("placed", {}).get(plate, 0))
    n = 12
    out = []
    for k in range(k0, k0 + n):
        a = math.radians(137.508 * k)
        r = rho * math.sqrt(((k % n) + 0.5) / n)
        out.append((pl["xy"][0] + r * math.cos(a), pl["xy"][1] + r * math.sin(a)))
    return out


def count_placed(plate):
    rec = plates_seen()
    rec.setdefault("placed", {})[plate] = int(rec["placed"].get(plate, 0)) + 1
    with open(PLATES_F + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(PLATES_F + ".tmp", PLATES_F)


def place(plate):
    """Carry the held brick to the plate where the scans last saw it: up to the carry height, across, down to the
    release height over the next free drop point (top-down IK), open, back up. Without a plate position or a
    reachable drop point, the taught path to that plate is used."""
    st = ga.state()
    q = st["present"][:5]
    carry = kin.MAT_Z + kin.cell.get("carry_dz_mm")
    rel = kin.MAT_Z + pa.TIP_FINAL + kin.cell.get("release_clearance_mm")
    psi = kin.psi_of(q)
    for xy in drop_points(plate):
        qa = kin.ik(xy[0], xy[1], carry, psi, q)
        qr = kin.ik(xy[0], xy[1], rel, psi, qa) if qa is not None else None
        if qa is None or qr is None:
            continue
        p, _ = kin.fk(q)
        if p[2] < carry:
            pa.move_rel(q, 0.0, [0, 0, carry - p[2]], speed=CFG["desc"] * 2)
        tmove(list(qa) + [0.0], CFG["transit"])
        tmove(list(qr) + [0.0], CFG["desc"] * 2)
        ga.cmd({"op": "grip", "value": 25.0, "speed": max(30.0, CFG["grip"])})
        time.sleep(0.2)
        ga.snap_async("released")
        tmove(list(qa) + [25.0], CFG["transit"])
        count_placed(plate)
        EP["place"] = {"how": "ik", "xy": [round(xy[0], 1), round(xy[1], 1)]}
        return
    place_taught(plate)


def place_taught(plate):
    """The taught plate path: straight up, then only the plate end of the validated teleop path (wp4..wp0), release,
    back up wp1..wp3/4."""
    st = ga.state()
    q = st["present"][:5]
    p, _ = kin.fk(q)
    if p[2] < kin.MAT_Z + 85 and q[1] > -60:
        pa.move_rel(q, 0.0, [0, 0, kin.MAT_Z + 90 - p[2]], speed=CFG["desc"] * 2)
    for i in PLATE_IN[plate]:
        tmove(WP[f"{plate}_wp{i}_closed"], CFG["transit"])
    ga.cmd({"op": "grip", "value": 25.0, "speed": max(30.0, CFG["grip"])})
    time.sleep(0.2)
    ga.snap_async("released")
    for i in PLATE_OUT[plate]:
        tmove(WP[f"{plate}_wp{i}"], CFG["transit"])
    count_placed(plate)
    EP["place"] = {"how": "taught path"}


def run_jaw_check(move=True):
    """Jaw_check.py at the current pose (the scan pose, gripper open). Returns (exit code, result dict)."""
    args = [f"{R}/envs/data-env/bin/python", "-B", "jaw_check.py"] + ([] if move else ["--no-move"]) + \
        (["--accept-bent"] if CFG.get("accept_bent_jaw") else [])
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


JAW_BASE = {}  # red fractions in and around the jaw mask on the scan frame right after a passed jaw check
GRASP_MISSED = {"last": False}  # the previous cycle's grasp closed on nothing


def jaw_red_fractions(img):
    """Fraction of the static jaw mask that is jaw-red in this wrist frame, and of a ring just outside it."""
    import jaw_check as jc
    jm = ga.JAW_MASK
    if img is None or jm is None or jm.shape != img.shape[:2]:
        return None
    red = jc.redmask(img) > 0
    m8 = jm.astype(np.uint8)
    ring = (cv2.dilate(m8, np.ones((31, 31), np.uint8)) > 0) & ~(cv2.dilate(m8, np.ones((11, 11), np.uint8)) > 0)
    return float(red[jm].mean()), float(red[ring].mean()) if ring.any() else 0.0


def jaw_check_due(k, img, every=20):
    """Run the two-roll jaw check when the scan frame suggests the jaws moved (less of the mask is jaw-red, or red
    appeared just outside it, compared with the frame after the last passed check), after a grasp missed, and as a
    backstop every `every` cycles."""
    if k % every == 0 or GRASP_MISSED["last"]:
        GRASP_MISSED["last"] = False
        return True
    f = jaw_red_fractions(img)
    if f is None or not JAW_BASE:
        return False
    if CFG.get("accept_bent_jaw"):  # a loose jaw moves a little at every grasp: re-measure only on a large change
        return f[0] < 0.6 * JAW_BASE["inside"] or f[1] > JAW_BASE["ring"] + 0.25
    return f[0] < 0.8 * JAW_BASE["inside"] or f[1] > JAW_BASE["ring"] + 0.10


def periodic_jaw_check(k, img):
    """The start-up jaw check again when it is due (jaw_check_due). A loose or bent fixed jaw pauses the loop with a
    clear message (the fixed jaw works loose when it touches the ground)."""
    if not jaw_check_due(k, img):
        return
    ok, d = jaw_check_now(f"cycle {k}")
    EP["jaw_check"] = d
    if ok:
        remember_jaws(pa.frame())
    elif CFG.get("accept_bent_jaw"):  # the operator cannot fix the jaw: keep the last accepted target, never pause
        log(f"jaw check (cycle {k}) not accepted ({d.get('error')}); keeping servo target {ga.TARGET_PX}")
    else:
        open(f"{ga.RUN_ROOT}/sort_pause", "w").write(
            f"{time.strftime('%H:%M:%SZ', time.gmtime())} jaw check failed: {d.get('error')}\n")
        log("pausing: tighten the fixed jaw, then delete run/sort_pause")


def remember_jaws(img):
    f = jaw_red_fractions(img)
    if f is not None:
        JAW_BASE.update(inside=f[0], ring=f[1])


def cycle(k):
    global EP
    EP = {"episode": k, "run": os.path.basename(RUN), "mode": "calibration" if CFG.get("calibrate") else "pick_anywhere"}
    ep_dir = f"{RUN}/ep_{k:04d}"
    os.makedirs(f"{ep_dir}/crops", exist_ok=True)
    ga.OUT = ep_dir
    ga.SUM.clear()
    ga.SUM["phases"] = []
    T = {}
    fit_scan_corr()
    periodic_jaw_check(k, pa.frame())
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
        GRASP_MISSED["last"] = True
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
    t1 = time.time()
    if CFG.get("calibrate"):
        calibration_rest(k, T, t0)
        return
    have = merged_faces()
    if survey_shape()[0] != "not_2x2" and not any(r.get("verdict") == "DEFECT" for r in have.values()) and \
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
    colour = EP.get("target_colour") or "unknown"
    shape = survey_shape()
    EP["shape_survey"] = list(shape)
    plate, label, conf, reasons = decide_now(shape)
    EP.update({"colour": colour, "plate": plate, "label": label, "decision_conf": conf, "shape": shape[0],
               "decision_reasons": reasons, "label_source": "survey", "needs_review": conf != "confident",
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


LABELS_F = os.path.join(ga.LOG_ROOT, "sort", "labels.json")  # ground truth for calibrate_defects.py (operator's labels)


def calibration_rest(k, T, t0):
    """Calibration run, after the grasp: survey the brick again at a free spot, set down turned by a random wrist
    roll, and leave it there (no decision, no plate). The next cycle picks it up again: every cycle adds two surveys of
    the same physical brick at new poses."""
    lo_b, hi_b = kin.CAL.norm_bounds(3)
    q = ga.state()["present"][:5]
    span = min(hi_b[4] - q[4], q[4] - lo_b[4], 30.0)
    roll = float(np.random.default_rng().uniform(-span, span)) if span > 0 else 0.0
    t1 = time.time()
    if cc.load() is not None:  # measure the bottom too (its model is fitted from these records)
        underside_record()
    surv, _ = survey_inspect(k, roll=roll, repick=False)
    EP["survey_spot"] = surv
    T["inspect_s"] = round(time.time() - t1, 1)
    if not surv.get("left_at_spot"):  # still held (no free spot or no pose): set it down where it is
        set_down_here()
        log(f"ep {k}: calibration: no spot survey ({surv.get('error')}); brick set down where it was held")
    go_scan(False)
    EP["t_end"] = time.time()
    ga.cmd({"op": "rec_stop"})
    T["cycle_s"] = round(EP["t_end"] - t0, 1)
    EP.update({"success": True, "failure": None, "timings": T, "calibration": CFG["calibrate"]})
    log(f"ep {k}: calibration cycle of brick {CFG['calibrate']['brick']} done in {T['cycle_s']} s")


def set_down_here():
    """Lower the held brick straight down onto the mat (put-down height as at the inspection spot), open, lift."""
    q = ga.state()["present"][:5]
    p, _ = kin.fk(q)
    qd = pa.move_rel(q, 0.0, [0, 0, (kin.MAT_Z + pa.TIP_FINAL + 1.0) - p[2]], speed=CFG["desc"])
    ga.cmd({"op": "grip", "value": 25.0, "speed": max(25.0, CFG["grip"])})
    time.sleep(0.3)
    if qd is not None:
        pa.move_rel(qd, 25.0, [0, 0, 40.0], speed=CFG["desc"])


def label_episode(k):
    """Add this calibration episode to the labels file (atomic write), with the operator's truth for the brick."""
    cal = CFG.get("calibrate")
    if not cal or not (EP.get("survey") or EP.get("survey_spot")):
        return
    try:
        lab = json.load(open(LABELS_F))
    except (OSError, ValueError):
        lab = {"bricks": {}, "episodes": {}, "source": "operator, sort_any.py --calibrate"}
    lab.setdefault("bricks", {})[cal["brick"]] = cal["truth"]
    lab.setdefault("episodes", {})[f"{os.path.basename(RUN)}/ep_{k:04d}"] = cal["brick"]
    os.makedirs(os.path.dirname(LABELS_F), exist_ok=True)
    with open(LABELS_F + ".tmp", "w") as f:
        json.dump(lab, f, indent=1)
    os.replace(LABELS_F + ".tmp", LABELS_F)


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
    label_episode(k)
    with open(f"{RUN}/cycles.jsonl", "a") as f:
        f.write(json.dumps({x: EP.get(x) for x in ("episode", "t_start", "t_end", "task", "plate", "label", "decision_conf",
                                                  "decision_reasons", "colour", "shape", "success", "failure",
                                                  "grip_closed", "timings", "brick_xy_est", "mode")}, default=str) + "\n")


def finish_held():
    """A brick is held: inspect (if no decision yet), decide, place."""
    plate = EP.get("plate")
    if not plate:
        have = merged_faces()
        if survey_shape()[0] != "not_2x2" and not any(r.get("verdict") == "DEFECT" for r in have.values()) and \
                any(have.get(f, {}).get("verdict") != "CLEAN" for f in srv.FACES):
            surv, ok2 = survey_inspect(EP.get("episode", 0))
            EP["survey_spot"] = surv
            if not ok2:
                log("held brick put down at the inspection spot; re-pick failed - the next cycle picks it up")
                return None
        shape = survey_shape()
        EP["shape_survey"] = list(shape)
        plate, label, conf, reasons = decide_now(shape)
        colour = EP.get("target_colour") or "unknown"
        EP.update({"plate": plate, "label": label, "decision_conf": conf, "colour": colour, "shape": shape[0],
                   "decision_reasons": reasons, "label_source": "survey", "needs_review": conf != "confident",
                   "task": f"move the {colour} brick to the {plate} plate"})
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
    ap.add_argument("--calibrate", metavar="BRICK_ID", default=None,
                    help="calibration run with one brick on the mat (any id the operator gives it): every cycle surveys it "
                         "twice and sets it down again, never on a plate; episodes go to logs/sort/labels.json")
    ap.add_argument("--accept-bent-jaw", action="store_true",
                    help="run with a bent but rigid fixed jaw: the jaw check accepts the jaws as they are")
    ap.add_argument("--truth", choices=("GOOD", "DEFECT"), default=None,
                    help="the calibration brick's ground truth (written to the labels file only)")
    ap.add_argument("--profile", choices=tuple(k for k in PROFILES if not k.startswith("_") and k != "limits"), default="safe",
                    help="motion profile from config/motion_profiles.json (safe, fast, demo)")
    a = ap.parse_args()
    RUN = f"{ga.LOG_ROOT}/sort/{a.run}"
    sl.RUN = RUN
    os.makedirs(RUN, exist_ok=True)
    prof = PROFILES[a.profile]
    CFG.update({k: float(prof[k]) for k in ("transit", "hover", "servo", "desc", "final")})
    CFG["settle_s"] = float(prof.get("settle_s", 0.0))
    CFG["grip"] = float(prof.get("grip", 20.0))
    hard = float(PROFILES["limits"]["hard_speed"])
    for key in ("transit", "hover", "servo", "desc"):
        CFG[key] = min(hard, CFG[key] * a.speed)
    pa.SERVO_SPEED = CFG["servo"]
    ga.ACCEL = float(prof["accel"])
    CFG["profile"] = a.profile
    CFG["accept_bent_jaw"] = a.accept_bent_jaw
    if a.calibrate:
        if not a.truth:
            sys.exit("--calibrate needs --truth GOOD or DEFECT")
        try:
            known = (json.load(open(LABELS_F)).get("bricks") or {}).get(a.calibrate)
        except (OSError, ValueError):
            known = None
        if known and known != a.truth:
            sys.exit(f"labels file already gives brick {a.calibrate} as {known}")
        CFG["calibrate"] = {"brick": a.calibrate, "truth": a.truth}
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
    if not ok and CFG.get("accept_bent_jaw") and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jaw_calib.json")):
        log(f"start-up jaw check not accepted ({d.get('error')}); bent-jaw mode keeps the last accepted target {ga.TARGET_PX}")
        ok = True
    if not ok:
        write_needs_operator(f"jaw check: {d.get('error')}")
        return
    remember_jaws(pa.frame())
    k = a.start
    while k is not None and k < a.start + a.n:
        k = run_one(k)
    log("loop end")


if __name__ == "__main__":
    main()
