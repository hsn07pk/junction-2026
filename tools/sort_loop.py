#!/usr/bin/env python3
"""Shared helpers of the sort loop (sort_any.py): the workspace-clear rule (hand watchdog heartbeat and live frames,
the same rule hand_watch.py uses), the loop log, crops, and the OD-10 decision from full-surface evidence."""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grasp_auto as ga  # noqa: E402
import handdet  # noqa: E402  (shared with hand_watch.py: the heartbeat and the hand rule, so they never disagree)

RUNTIME_DIR = ga.RUN_ROOT  # live frames, stop/pause/estop flags, motion state, hand_watch_alive
HAND_LOG = f"{ga.LOG_ROOT}/monitor/hand_watch.log"
FRAME_MAX_AGE = 1.0  # s: an older live frame is not evidence of a clear workspace
WATCHDOG_DOWN = False  # True while the heartbeat is stale: keeps 'hand watchdog not running' to one line per period
RUN = ""  # the run's log folder, set by the loop; without one the log goes to the monitor folder


def log(msg):
    line = time.strftime("%H:%M:%SZ ", time.gmtime()) + msg
    print(line, flush=True)
    d = RUN or os.path.join(ga.LOG_ROOT, "monitor")
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "loop.log"), "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


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


def save_crop(img, bbox, path, pad=12):
    if img is None or bbox is None:
        return
    x, y, w, h = bbox
    H, W = img.shape[:2]
    cv2.imwrite(path, img[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)])


def _min_clean_faces(n_required: int) -> int:
    """How many surveyed-clean faces make a defect-free 2x2 brick GOOD (config/inspection.json min_clean_faces)."""
    try:
        cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                                          "inspection.json")))
        return max(1, min(n_required, int(cfg.get("min_clean_faces", n_required))))
    except (OSError, ValueError):
        return n_required


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
    n_clean = len(required) - len(open_faces)
    need = _min_clean_faces(len(required))
    if open_faces and n_clean >= need:
        return "blue", "GOOD", "partial", [f"2x2, {n_clean} of {len(required)} faces surveyed clean, none defective "
                                           f"(at least {need} clean required)"]
    if open_faces:
        why = {f: faces.get(f, {}).get("reason", faces.get(f, {}).get("verdict", "not surveyed")) for f in open_faces}
        return "green", "UNCERTAIN", "incomplete", [f"not cleared: {why} -> GREEN (OD-10)"]
    return "blue", "GOOD", "confident", ["2x2, every surveyed face clean"]
