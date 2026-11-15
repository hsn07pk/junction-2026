#!/usr/bin/env python3
"""Client of the motion server and the live cameras, shared by the sort loop: commands (motion_server.py's command
file and state), the newest live frame of each camera, snapshots into the episode folder, and the wrist camera's
static jaw mask and servo target (from the jaw calibration). Needs motion_server.py, the camera recorders
(monitor_cams.sh, wrist_rec.sh) and hand_watch.py running."""
import json
import os
import sys
import threading
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell  # noqa: E402
from paths import APP, LOG_ROOT, R, RUN_ROOT, SIM_ROOT  # noqa: E402,F401

T = f"{APP}/tools"
SD = f"{RUN_ROOT}/mserver"
TARGET_PX = tuple(json.load(open(os.path.join(T, "jaw_calib.json")))["target_px"])  # wrist pixel under the jaw centre
YAW_PER_ROLL = cell.get("yaw_per_roll_deg")  # deg of wrist-cam brick yaw per roll unit (+)
OUT = ""
SUM = {"phases": []}
_jm = cv2.imread(f"{T}/wrist_jaw_mask.png", cv2.IMREAD_GRAYSCALE)
JAW_MASK = (_jm > 0) if _jm is not None else None  # static: the wrist camera rides on the gripper (open = 25)


class Stop(Exception):
    pass


def cmd(c, timeout=120):
    c = dict(c)
    c["id"] = int(time.time() * 1000) % 1_000_000_000
    with open(f"{SD}/cmd.jsonl", "a") as f:
        f.write(json.dumps(c) + "\n")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st = json.load(open(f"{SD}/state.json"))
            if st.get("last_id") == c["id"]:
                return st
        except Exception:
            pass
        time.sleep(0.05)
    raise Stop("motion server timeout")


ACCEL = 60.0  # joint acceleration of move() and pick_any.move_rel(); the sort loop sets it from its motion profile


def move(q, speed=5.0, comp=True):
    st = cmd({"op": "move", "q": [float(x) for x in q], "speed": speed, "comp": comp, "accel": ACCEL})
    if st["status"] != "ok":
        raise Stop(f"move -> {st['status']} {st['msg']}")
    return st


def fresh(view, after):
    p = f"{RUN_ROOT}/live_{view}.jpg"
    for _ in range(60):
        try:
            if os.path.getmtime(p) > after:
                img = cv2.imread(p)
                if img is not None:
                    return img
        except OSError:
            pass
        time.sleep(0.05)
    return cv2.imread(p)


def snap(tag):
    t = time.time() + 0.3
    imgs = {v: fresh(v, t) for v in ("top", "close", "wrist")}
    for v, im in imgs.items():
        if im is not None:
            cv2.imwrite(f"{OUT}/{tag}_{v}.jpg", im)
    return imgs


def state():
    return cmd({"op": "state"})


def snap_async(tag):
    """snap() in the background: the episode keeps its phase frames without the arm waiting for them."""
    out = OUT
    t = time.time() + 0.3

    def work():
        for v in ("top", "close", "wrist"):
            im = fresh(v, t)
            if im is not None and out:
                cv2.imwrite(f"{out}/{tag}_{v}.jpg", im)

    threading.Thread(target=work, daemon=True).start()


def wait_grip(timeout=0.8, tol=0.02):
    """Gripper reading once the jaws have stopped (two readings 50 ms apart within tol), at most timeout s."""
    t0 = time.time()
    last = state()["present"][5]
    while time.time() - t0 < timeout:
        time.sleep(0.05)
        now = state()["present"][5]
        if abs(now - last) <= tol and time.time() - t0 >= 0.15:
            return now
        last = now
    return last

