#!/usr/bin/env python3
"""Autonomous sort loop, spot cell (verified procedure: docs/SPOT-GRASP-DEPTH.md).

Park = open jaws hovering over the spot. The operator puts a brick under the jaws (any brick, any yaw, within ~3 cm).
Cycle: wait (brick stable, no hands) -> pre-inspect (close tape rule + wrist shape) -> record episode ->
yaw align + wrist-cam centring -> constant-pitch descent with depth checks -> close -> lift check ->
multi-view (held brick rolled in front of the close camera) -> decide (I-52: any confident defect -> GREEN)
-> plate -> back to park. Every cycle: R/logs/sort/<RUN>/ep_<NNNN>/ (episode.json, joints.csv, frames, crops).
usage: sort_loop.py --run NAME [--n 150] [--speed 1.0] [--no-multiview]
Stop: touch R/run/sort_stop (finishes the current cycle). Pause: touch R/run/sort_pause.
"""
import argparse
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "anomaly", "det2_f2"))
import grasp_auto as ga  # noqa: E402
import handdet  # noqa: E402  (shared with hand_watch.py: the heartbeat and the hand rule, so they never disagree)
import inspector as DET  # noqa: E402

R = ga.R
HOVER = list(ga.HOVER)
RUNTIME_DIR = ga.RUN_ROOT  # live frames, stop/pause/estop flags, motion state, hand_watch_alive
HAND_LOG = f"{ga.LOG_ROOT}/monitor/hand_watch.log"
FRAME_MAX_AGE = 1.0  # s: an older live frame is not evidence of a clear workspace
WATCHDOG_DOWN = False  # True while the heartbeat is stale: keeps 'hand watchdog not running' to one line per period
CLOSE_SPOT_PX = (380, 262)          # brick centre at the spot in the close colour image
ROLL_PER_90 = 90.0 / ga.YAW_PER_ROLL
SPOT_FAST = f"{R}/app/motion/gm/spot_fast.yaml"
CFG = {"transit": 25.0, "approach": 12.0, "fine": 6.0, "desc": 4.0, "accel": 100.0, "multiview": True}
EP = {}


def log(msg):
    line = time.strftime("%H:%M:%SZ ", time.gmtime()) + msg
    print(line, flush=True)
    with open(f"{RUN}/loop.log", "a") as f:
        f.write(line + "\n")


def skin_area(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((ycc[..., 1] >= 138) & (ycc[..., 1] <= 170) & (ycc[..., 2] >= 90) & (ycc[..., 2] <= 125)
         & (h >= 5) & (h <= 20) & (s >= 45) & (s <= 150) & (v >= 90)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return int(st[1:, 4].max()) if n > 1 else 0


def watchdog_up():
    """False while hand_watch.py's heartbeat is missing or older than 3 s: without a live watchdog nothing is clear.
    Logs 'hand watchdog not running' once per stale period."""
    global WATCHDOG_DOWN
    if handdet.heartbeat_fresh():
        WATCHDOG_DOWN = False
        return True
    if not WATCHDOG_DOWN:
        WATCHDOG_DOWN = True
        log("hand watchdog not running")
    return False


def hands_seen(sec):
    """True if hand_watch logged any hand in the last `sec` seconds or a hand is visible now (top/close).
    Also True while the watchdog heartbeat is missing or stale: the loop does not move without a live watchdog."""
    if not watchdog_up():
        return True
    try:
        lines = open(HAND_LOG).read().splitlines()[-30:]
    except OSError:
        lines = []
    now = time.gmtime()
    nows = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    for ln in lines:
        try:
            hh, mm, ss = ln[:8].split(":")
            t = int(hh) * 3600 + int(mm) * 60 + int(ss.rstrip("Z"))
        except ValueError:
            continue
        if 0 <= nows - t <= sec and "HAND?" in ln:  # far-band notes do not count; wrist lines count too (the watchdog logs the wrist only when it may stop on it)
            return True
    try:
        hold = json.load(open(f"{RUNTIME_DIR}/mserver/state.json"))["target"][5] < 20
    except Exception:  # noqa: BLE001
        hold = False
    for view in ("top", "close"):
        p = f"{RUNTIME_DIR}/live_{view}.jpg"
        try:
            if time.time() - os.path.getmtime(p) > FRAME_MAX_AGE:
                return True  # a dead recorder shows an old frame forever: not clear
        except OSError:
            return True
        img = cv2.imread(p)
        if img is None:
            continue
        hand, f = handdet.is_hand(img, view, hold)
        if hand and view == "close" and handdet.far_band_only(f, img.shape[0]):
            continue  # beyond the far edge of the table and the top view (checked first) shows no hand
        if hand:
            return True
    return False


def wait_clear(reason):
    log(f"waiting for a clear workspace ({reason})")
    while True:
        if os.path.exists(f"{RUNTIME_DIR}/sort_stop"):
            raise SystemExit("stop requested")
        if not hands_seen(5):
            if os.path.exists(f"{RUNTIME_DIR}/estop"):
                os.remove(f"{RUNTIME_DIR}/estop")
                log("estop cleared automatically: no hands for 3 s")
            return
        time.sleep(0.5)


def wait_for_brick():
    stable, last = 0, None
    t0 = time.time()
    while True:
        if os.path.exists(f"{ga.RUN_ROOT}/sort_stop"):
            raise SystemExit("stop requested")
        if os.path.exists(f"{ga.RUN_ROOT}/sort_pause"):
            time.sleep(0.5)
            continue
        img = ga.fresh("wrist", time.time())
        wb = ga.wrist_brick(img) if img is not None else None
        ok = (wb is not None and wb["dist_to_target"] < 120 and 3500 < wb["area"] < 45000
              and skin_area(img) < 20000 and not hands_seen(2))
        if ok:
            if last is not None and np.hypot(wb["centre"][0] - last[0], wb["centre"][1] - last[1]) < 6:
                stable += 1
            else:
                stable = 0
            last = wb["centre"]
            if stable >= 4:
                return wb, round(time.time() - t0, 1)
        else:
            stable, last = 0, None
        time.sleep(0.05)


def mask_jaws(img):
    """Black out red components that touch the top border (the jaws/gripper hang into the close view)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((H <= 8) | (H >= 170)) & (S >= 100) & (V >= 50)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    out = img.copy()
    for i in range(1, n):
        if st[i, 1] <= 3 and st[i, 4] > 800:
            out[cv2.dilate((lab == i).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = 0
    return out


_cm = cv2.imread(f"{R}/app/tools/close_jaw_mask_hover.png", cv2.IMREAD_GRAYSCALE)
CLOSE_MASK_HOVER = (_cm > 0) if _cm is not None else None


def _seg_near_spot(img):
    seg = DET.segment_close(img)
    best = None
    for rec in seg["bricks"]:
        cx, cy = rec["centre_px"]
        d = float(np.hypot(cx - CLOSE_SPOT_PX[0], cy - CLOSE_SPOT_PX[1]))
        if best is None or d < best[0]:
            best = (d, rec)
    return best


def _jaw_anchored_verdict(img, thr):
    """When no brick is near the spot, locate the brick between the jaw tips (same tip rule as
    sort_any.close_pre_verdict) and run the same DET2 triage on it."""
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
        return {"verdict": "UNCERTAIN", "reasons": ["no_brick_near_spot", "no_jaw_tips"]}
    tips.sort(key=lambda t: -t[1])
    mx = float(np.mean([t[0] for t in tips[:2]]))
    my = float(max(t[1] for t in tips[:2])) + 40.0
    best = None
    for rec in DET.segment_close(mask_jaws(img))["bricks"]:
        cx, cy = rec["centre_px"]
        d = float(np.hypot(cx - mx, cy - my))
        if best is None or d < best[0]:
            best = (d, rec)
    if best is None or best[0] > 130:
        return {"verdict": "UNCERTAIN", "reasons": ["no_brick_near_spot", "no_brick_at_jaws"]}
    rec = best[1]
    verdict, reasons = DET.triage(rec, thr)
    x, y, w, h = rec["bbox"]
    return {"verdict": verdict, "reasons": list(reasons) + ["jaw_anchored"], "colour": rec.get("colour"),
            "tape_score": rec.get("tape_score"), "bbox": [int(x), int(y), int(w), int(h)], "dist": round(best[0], 1)}


def close_verdict(img, at_hover=False):
    """Tape verdict of the brick nearest the spot in a close-camera frame (DET2 close rule, jaws masked:
    colour mask of top-touching red parts; at the hover also the static jaw mask, so red/brown bricks survive)."""
    try:
        best = _seg_near_spot(mask_jaws(img))
        if (best is None or best[0] > 170) and at_hover and CLOSE_MASK_HOVER is not None:
            im2 = img.copy()
            im2[CLOSE_MASK_HOVER] = 0
            best = _seg_near_spot(im2)
    except Exception as e:  # noqa: BLE001
        return {"verdict": "UNCERTAIN", "reasons": [f"det2_error {e!r}"]}
    thr = DET.load_thresholds()
    if best is None or best[0] > 170:
        return _jaw_anchored_verdict(img, thr)  # held brick at the frame edge: find it between the jaw tips
    rec = best[1]
    verdict, reasons = DET.triage(rec, thr)
    x, y, w, h = rec["bbox"]
    return {"verdict": verdict, "reasons": reasons, "colour": rec.get("colour"), "tape_score": rec.get("tape_score"),
            "bbox": [int(x), int(y), int(w), int(h)], "dist": round(best[0], 1)}


def save_crop(img, bbox, path, pad=12):
    if img is None or bbox is None:
        return
    x, y, w, h = bbox
    H, W = img.shape[:2]
    cv2.imwrite(path, img[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)])


def shape_from_wrist(wb):
    short = min(wb["size"])
    if wb["aspect"] > 1.4:
        return "not_2x2", f"aspect {wb['aspect']}"
    if short < 80:
        return "not_2x2", f"narrow {short}px"
    return "2x2", f"aspect {wb['aspect']} short {short}px"


def run_fast(seq, speed):
    d = json.load(open(SPOT_FAST))
    for s in d["sequences"][seq]:
        st = ga.cmd({"op": "move", "q": [float(x) for x in d["poses"][s["pose"]]], "speed": speed, "comp": False,
                     "accel": CFG["accel"]})
        if st["status"] != "ok":
            raise ga.Stop(f"{seq}/{s['pose']} -> {st['status']}")


def go_park(holding=False):
    st = ga.state()
    g = 0.0 if holding else 25.0
    ga.move([st["present"][0], -40.0, 80.0, 66.0, -45.0, g], speed=CFG["transit"], comp=False)
    ga.move(HOVER[:5] + [g], speed=CFG["approach"], comp=True)


def pick(wb0):
    """Align, centre, descend, close, lift. Returns (ok, info)."""
    q = list(HOVER)
    wb = wb0
    for it in range(3):
        yaw = wb["yaw"]
        if wb["aspect"] > 1.4 and not wb["long_vertical"]:
            yaw = yaw + 90.0 if yaw < 0 else yaw - 90.0
        if abs(yaw) <= 2.5:
            break
        nr = q[4] - yaw / ga.YAW_PER_ROLL
        if abs(nr) > 99:
            nr = q[4] - (yaw + (90 if yaw < 0 else -90)) / ga.YAW_PER_ROLL if wb["aspect"] <= 1.4 else nr
        q[4] = round(float(np.clip(nr, -99, 99)), 2)
        ga.move(q, speed=CFG["fine"] * 2, comp=True)
        wb = ga.wrist_brick(ga.snap(f"yaw{it}")["wrist"])
        ga.note("yaw", iter=it, roll=q[4], wrist=wb)
        if wb is None:
            return False, "brick lost in wrist cam during yaw"
    J = np.zeros((2, 2))
    for col, (j, dq) in enumerate(((0, 2.0), (3, 3.0))):
        qq = list(q)
        qq[j] += dq
        ga.move(qq, speed=CFG["fine"], comp=True)
        wp = ga.wrist_brick(ga.snap(f"probe{j}")["wrist"])
        ga.move(q, speed=CFG["fine"], comp=True)
        if wp is None:
            return False, "brick lost during probe"
        J[:, col] = [(wp["centre"][0] - wb["centre"][0]) / dq, (wp["centre"][1] - wb["centre"][1]) / dq]
    ga.note("jacobian_px_per_unit", pan=J[:, 0].round(2).tolist(), wf=J[:, 1].round(2).tolist())
    for it in range(4):
        err = np.array(ga.TARGET_PX) - np.array(wb["centre"])
        if np.linalg.norm(err) <= ga.TOL_PX:
            break
        dq = np.clip(np.linalg.lstsq(J, err, rcond=None)[0], [-5.0, -10.0], [5.0, 10.0])
        q[0], q[3] = round(q[0] + dq[0], 2), round(q[3] + dq[1], 2)
        ga.move(q, speed=CFG["fine"], comp=True)
        wb = ga.wrist_brick(ga.snap(f"centre{it}")["wrist"])
        ga.note("centre", iter=it, d_pan=round(float(dq[0]), 2), d_wf=round(float(dq[1]), 2), q=q, wrist=wb)
        if wb is None:
            return False, "brick lost while centring"
    if np.linalg.norm(np.array(ga.TARGET_PX) - np.array(wb["centre"])) > 2 * ga.TOL_PX:
        return False, f"could not centre (wrist {wb['centre']})"
    hover_c = list(q)
    EP["hover_aligned"] = hover_c
    cv2.imwrite(f"{ga.OUT}/crops/wrist_hover.png", ga.fresh("wrist", time.time()))
    Hm, P, n, _ = ga.depth("hover")
    bb = ga.brick_blob(Hm)
    if bb is None or not (15 <= bb["top_h"] <= 26):
        return False, f"no brick in close depth at the spot ({None if bb is None else bb['top_h']})"
    q[2], q[3] = round(hover_c[2] + ga.STAGES[0], 2), round(hover_c[3] + ga.WF_PER_EL * ga.STAGES[0], 2)
    ga.move(q, speed=CFG["desc"], comp=True)
    Hm, P, n, _ = ga.depth("stage1")
    bb1 = ga.brick_blob(Hm, near=tuple(bb["centre"]), max_d=25) or bb
    clear = ga.jaws_clear_of_brick(Hm, P, n, bb1)
    ga.note("stage1", q=list(q), jaw_brick_clear_mm=clear, jaw_low=ga.jaw_tip_low(Hm, bb1))
    if clear is not None and clear < 2.0:
        ga.move(hover_c, speed=CFG["desc"], comp=True)
        return False, f"jaw over the brick (clearance {clear} mm)"
    q[2], q[3] = round(hover_c[2] + ga.STAGES[1], 2), round(hover_c[3] + ga.WF_PER_EL * ga.STAGES[1], 2)
    ga.move(q, speed=CFG["desc"], comp=True)
    Hm, P, n, _ = ga.depth("stage2")
    low = ga.jaw_tip_low(Hm, bb1)
    h2 = min(low.values()) if low else None
    extra = (h2 - ga.TIP_TARGET) / ga.EL_MM if h2 is not None and 10 <= h2 <= 30 else 6.8
    extra = float(np.clip(extra, 3.0, 10.0))
    q[2], q[3] = round(hover_c[2] + ga.STAGES[1] + extra, 2), round(hover_c[3] + ga.WF_PER_EL * (ga.STAGES[1] + extra), 2)
    st = ga.move(q, speed=CFG["desc"], comp=True)
    ga.note("grasp_open", q=list(q), jaw_low_stage2=low, extra_el=round(extra, 2), err=st["err"])
    ga.cmd({"op": "grip", "value": 0.0, "speed": 20})
    time.sleep(0.5)
    g = ga.state()["present"][5]
    EP["grip_closed"] = g
    ga.snap("closed")
    q[5] = 0.0
    ql = list(q)
    ql[2], ql[3] = round(q[2] + ga.LIFT_DEL, 2), round(q[3] + ga.WF_PER_EL * ga.LIFT_DEL, 2)
    ga.move(ql, speed=CFG["desc"], comp=True)
    imgs = ga.snap("lifted")
    Hm, P, n, _ = ga.depth("lifted")
    left = bb1["mask"] & (np.nan_to_num(Hm, nan=0) > 15) & (np.nan_to_num(Hm, nan=99) < 26)
    g2 = ga.state()["present"][5]
    wbl = ga.wrist_brick(imgs["wrist"])
    ga.note("lifted", gripper_closed=g, gripper_lifted=g2, spot_px_left=int(left.sum()), spot_px_before=int(bb1["mask"].sum()), wrist=wbl)
    EP["lift_pose"] = ql
    if left.sum() > 0.35 * bb1["mask"].sum() or g2 < 0.6:
        ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
        ga.move(hover_c[:5] + [25.0], speed=CFG["desc"], comp=True)
        return False, f"brick not lifted (left {int(left.sum())} px, gripper {g2})"
    return True, "held"


def multiview(ql):
    """Roll the held brick in front of the close camera; DET2 tape verdict per view; crops saved."""
    views = []
    base = list(ql)
    up = list(base)
    up[2], up[3] = round(base[2] - 8.0, 2), round(base[3] + ga.WF_PER_EL * -8.0, 2)
    ga.move(up, speed=CFG["desc"] * 2, comp=True)
    offs = [0.0]
    for k in (1, 2, -1):
        r = up[4] + k * ROLL_PER_90
        if -99 <= r <= 99:
            offs.append(k * ROLL_PER_90)
        if len(offs) >= 3:
            break
    for i, off in enumerate(offs):
        q = list(up)
        q[4] = round(up[4] + off, 2)
        ga.move(q, speed=CFG["fine"] * 4, comp=False)
        time.sleep(0.25)
        img = ga.fresh("close", time.time() + 0.15)
        cv = close_verdict(img)
        cv2.imwrite(f"{ga.OUT}/crops/held_close_r{i}_full.jpg", img)
        save_crop(img, cv.get("bbox"), f"{ga.OUT}/crops/held_close_r{i}.png")
        cv["roll_off_deg"] = round(off * ga.YAW_PER_ROLL)
        views.append(cv)
    ga.move(up, speed=CFG["fine"] * 4, comp=False)
    return views, up


def decide(pre, shape, views):
    """OD-10 (binding): GOOD (2x2, no tape) -> blue; DEFECT (tape or not 2x2) -> green; UNCERTAIN -> green.
    Returns (plate, label, confidence, reasons). Blue needs a measured 2x2 shape and a CONFIDENT_GOOD view."""
    reasons = []
    shape_known = shape[0] in ("2x2", "not_2x2")
    if shape_known and shape[0] != "2x2":
        reasons.append("not_2x2:" + shape[1])
    if pre.get("verdict") == "CONFIDENT_DEFECT":
        reasons.append("tape_pre:" + ",".join(pre.get("reasons", [])))
    for v in views:
        if v.get("verdict") == "CONFIDENT_DEFECT":
            reasons.append(f"tape_held_r{v.get('roll_off_deg')}")
    if reasons:
        return "green", "DEFECT", "confident", reasons
    if not shape_known:  # e.g. a held brick after a restart whose re-pick failed: not known to be 2x2
        return "green", "UNCERTAIN", "shape_unknown", ["shape not measured: UNCERTAIN -> GREEN (OD-10)"]
    goods = [pre.get("verdict") == "CONFIDENT_GOOD"] + [v.get("verdict") == "CONFIDENT_GOOD" for v in views]
    if any(goods):
        return "blue", "GOOD", "confident" if goods[0] else "held_only", ["no_defect_evidence"]
    scores = [x.get("tape_score") for x in [pre] + views if x.get("tape_score") is not None]
    if not scores:  # until 22:10Z this went to blue as 'GOOD by default'
        return "green", "UNCERTAIN", "no_evidence", ["no usable tape view: UNCERTAIN -> GREEN (OD-10)"]
    thr = DET.load_thresholds() or {}
    mid = (thr.get("good", 0) + thr.get("defect", 1)) / 2 if thr else None
    if mid is None:
        return "green", "UNCERTAIN", "no_threshold", [f"max_tape {max(scores):.3f}, thresholds not loaded (OD-10)"]
    if max(scores) >= mid:
        return "green", "DEFECT", "uncertain_high_score", [f"max_tape {max(scores):.3f} >= mid {mid:.3f}"]
    return "green", "UNCERTAIN", "uncertain_low_score", [f"max_tape {max(scores):.3f} < mid {mid:.3f}, no "
                                                         "CONFIDENT_GOOD view: UNCERTAIN -> GREEN (OD-10)"]


def decide_faces(pre, shape, faces, required):
    """OD-10 with full-surface evidence. DEFECT (green): the shape is not 2x2, a surveyed face shows a defect, or the
    pre-grasp view is a confident defect. GOOD (blue): a measured 2x2 brick whose required faces were all surveyed
    CLEAN. Everything else is UNCERTAIN (green). Returns (plate, label, confidence, reasons)."""
    if shape[0] == "not_2x2":
        return "green", "DEFECT", "confident", ["not_2x2:" + shape[1]]
    bad = [f for f in required if faces.get(f, {}).get("verdict") == "DEFECT"]
    if pre.get("verdict") == "CONFIDENT_DEFECT":
        bad.append("pre_close:" + ",".join(pre.get("reasons", [])))
    if bad:
        return "green", "DEFECT", "confident", ["defect: " + ", ".join(bad)]
    if shape[0] != "2x2":
        return "green", "UNCERTAIN", "shape_unknown", ["shape not measured: UNCERTAIN -> GREEN (OD-10)"]
    open_faces = [f for f in required if faces.get(f, {}).get("verdict") != "CLEAN"]
    if open_faces:
        why = {f: faces.get(f, {}).get("reason", faces.get(f, {}).get("verdict", "not surveyed")) for f in open_faces}
        return "green", "UNCERTAIN", "incomplete", [f"not cleared: {why} -> GREEN (OD-10)"]
    return "blue", "GOOD", "confident", ["2x2, every surveyed face clean"]


def place(plate):
    st = ga.state()
    ga.move([st["present"][0], -40.0, 80.0, 66.0, -45.0, 0.0], speed=CFG["transit"], comp=False)
    ga.move([-6.0, -98.0, 96.6, 66.0, -45.0, 0.0], speed=CFG["transit"], comp=False)
    run_fast(f"home_to_{plate}_carry", CFG["transit"])
    run_fast(f"release_{plate}", CFG["transit"])
    time.sleep(0.3)
    ga.snap("released")
    run_fast(f"{plate}_to_home", CFG["transit"])


def cycle(k):
    global EP
    EP = {"episode": k, "run": os.path.basename(RUN)}
    ep_dir = f"{RUN}/ep_{k:04d}"
    os.makedirs(f"{ep_dir}/crops", exist_ok=True)
    ga.OUT = ep_dir
    ga.SUM.clear()
    ga.SUM["phases"] = []
    T = {}
    wb0, waited = wait_for_brick()
    T["wait_s"] = waited
    t0 = time.time()
    imgs = ga.snap("pre")
    pre = close_verdict(imgs["close"], at_hover=True)
    save_crop(imgs["close"], pre.get("bbox"), f"{ep_dir}/crops/pre_close.png")
    shape = shape_from_wrist(wb0)
    colour = pre.get("colour") or "unknown"
    EP.update({"pre": pre, "shape": shape[0], "shape_why": shape[1], "colour": colour, "wrist_pre": wb0})
    log(f"ep {k}: brick {colour} {shape[0]} ({shape[1]}), close {pre.get('verdict')} {pre.get('reasons')}")
    ga.cmd({"op": "rec_start", "path": f"{ep_dir}/joints.csv"})
    EP["t_start"] = time.time()
    ok, why = pick(wb0)
    T["pick_s"] = round(time.time() - t0, 1)
    if not ok:
        EP.update({"success": False, "failure": why})
        log(f"ep {k}: PICK FAILED: {why}")
        ga.cmd({"op": "rec_stop"})
        ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
        go_park(False)
        return
    t1 = time.time()
    fast = pre.get("verdict") == "CONFIDENT_DEFECT" or shape[0] != "2x2"  # I-52: a sure defect skips inspection
    EP["fast_path"] = fast
    views, up = multiview(EP["lift_pose"]) if (CFG["multiview"] and not fast) else ([], EP["lift_pose"])
    T["multiview_s"] = round(time.time() - t1, 1)
    plate, label, conf, reasons = decide(pre, shape, views)
    EP.update({"views": views, "plate": plate, "label": label, "decision_conf": conf, "decision_reasons": reasons,
               "label_source": "det2", "task": f"move the {colour} brick to the {plate} plate"})
    log(f"ep {k}: decision {label} -> {plate} ({conf}; {reasons})")
    t2 = time.time()
    place(plate)
    T["place_s"] = round(time.time() - t2, 1)
    t3 = time.time()
    go_park(False)
    T["return_s"] = round(time.time() - t3, 1)
    EP["t_end"] = time.time()
    ga.cmd({"op": "rec_stop"})
    T["cycle_s"] = round(EP["t_end"] - t0, 1)
    EP.update({"success": True, "failure": None, "timings": T})
    log(f"ep {k}: DONE {label} -> {plate}, cycle {T['cycle_s']} s, timings {T}")


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-multiview", action="store_true")
    a = ap.parse_args()
    RUN = f"{ga.LOG_ROOT}/sort/{a.run}"
    os.makedirs(RUN, exist_ok=True)
    for key in ("transit", "approach", "fine", "desc"):
        CFG[key] = min(45.0, CFG[key] * a.speed)
    CFG["multiview"] = not a.no_multiview
    log(f"start run {a.run} n={a.n} cfg={CFG}")
    for f in ("sort_stop",):
        if os.path.exists(f"{ga.RUN_ROOT}/{f}"):
            os.remove(f"{ga.RUN_ROOT}/{f}")
    wait_clear("start-up")  # no move before a live hand watchdog and a clear workspace
    go_park(False)
    k = a.start
    while k < a.start + a.n:
        try:
            cycle(k)
        except SystemExit as e:
            log(f"stopping: {e}")
            break
        except ga.Stop as e:
            EP.update({"success": False, "failure": f"stop: {e}"})
            log(f"ep {k}: STOP {e}; recovering")
            try:
                ga.cmd({"op": "rec_stop"})
                wait_clear("estop")
                g = ga.state()["present"][5]
                holding = 1.0 <= g <= 6.0 and EP.get("lift_pose") is not None
                if holding and EP.get("plate"):
                    place(EP["plate"])
                    go_park(False)
                    EP["failure"] += " (recovered: placed after estop)"
                else:
                    ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
                    go_park(False)
            except Exception as e2:  # noqa: BLE001
                log(f"recovery failed: {e2!r}; waiting for the operator")
                EP["failure"] += f" recovery failed {e2!r}"
                json.dump(EP, open(f"{RUN}/ep_{k:04d}/episode.json", "w"), indent=1, default=str)
                break
        except Exception as e:  # noqa: BLE001
            EP.update({"success": False, "failure": f"exception {e!r}"})
            log(f"ep {k}: EXCEPTION {traceback.format_exc()[-800:]}")
            try:
                ga.cmd({"op": "rec_stop"})
                ga.cmd({"op": "grip", "value": 25.0, "speed": 20})
                go_park(False)
            except Exception:  # noqa: BLE001
                break
        finally:
            if EP:
                EP["t_end"] = EP.get("t_end") or time.time()
                cams = {"top": sorted(os.listdir(f"{ga.LOG_ROOT}/monitor")), "wrist_dir": "data/wrist/raw"}
                EP["cams"] = {"top": "logs/monitor/" + [x for x in cams["top"] if x.startswith("top_")][-1],
                              "close": "logs/monitor/" + [x for x in cams["top"] if x.startswith("close_")][-1],
                              "wrist_dir": cams["wrist_dir"]}
                json.dump(EP, open(f"{RUN}/ep_{k:04d}/episode.json", "w"), indent=1, default=str)
                with open(f"{RUN}/cycles.jsonl", "a") as f:
                    f.write(json.dumps({x: EP.get(x) for x in ("episode", "t_start", "t_end", "task", "plate", "label",
                                                              "decision_conf", "decision_reasons", "colour", "shape",
                                                              "success", "failure", "grip_closed", "timings")},
                                       default=str) + "\n")
        k += 1
    log("loop end")


RUN = ""
if __name__ == "__main__":
    main()
