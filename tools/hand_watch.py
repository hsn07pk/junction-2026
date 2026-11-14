"""Hand watchdog (rules shared with the sort loop through handdet.py).

Every 0.25 s: read the newest run/live_{top,close,wrist}.jpg. While run/motion_active exists, a hand seen on enough
consecutive frames writes run/estop (the motion server holds the arm and keeps squeezing a held brick) and SIGINTs
any run_sequence.py. Every hand frame is logged to logs/monitor/hand_watch.log; the frame that stops the arm is saved
as logs/monitor/estop_<view>_<epoch>.jpg.

The detector lives in handdet.py and is the SAME code the loop's clear check (sort_loop.hands_seen) uses, so the
watchdog and the loop can never disagree about whether the workspace is clear (cause of the 18:03Z tight loop).
The wrist view is used only while the gripper is open and the jaws are > 90 mm above the mat: a white or cream
brick in or under the gripper looks skin-toned to the wrist camera.
Every pass writes the heartbeat run/hand_watch_alive (handdet.write_heartbeat); the sort loop treats a missing or
stale heartbeat (over 3 s) as not clear.

Run:  nohup envs/data-env/bin/python -u -B app/tools/hand_watch.py >> logs/monitor/hand_watch.out 2>&1 &
      echo $! > run/hand_watch.pid      (stop: kill $(cat run/hand_watch.pid))
"""
import json
import os
import subprocess
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP, R  # noqa: E402  (roots from this file's location)
sys.path.insert(0, f"{APP}/tools")
import handdet
from paths import LOG_ROOT, RUN_ROOT  # noqa: E402

try:
    import kin  # noqa: E402
except Exception:  # noqa: BLE001  (no kinematics -> the wrist view is never used)
    kin = None

STATE = f"{RUN_ROOT}/mserver/state.json"
FRAME_MAX_AGE = 1.0  # s: an older top or close frame means a dead recorder, so no heartbeat
WRIST_MIN_HEIGHT_MM = 90.0


def _state():
    try:
        return json.load(open(STATE))
    except Exception:  # noqa: BLE001
        return None


def holding(st) -> bool:
    """Gripper commanded closed: a brick may be between the jaws."""
    return bool(st) and st["target"][5] < 20


def wrist_usable(st) -> bool:
    """Wrist view only with the gripper open and the jaws high above the mat (transit / scan)."""
    if not st or kin is None or holding(st):
        return False
    try:
        p, _ = kin.fk(st["present"][:5])
    except Exception:  # noqa: BLE001
        return False
    return p[2] > kin.MAT_Z + WRIST_MIN_HEIGHT_MM


def frames_live(now=None):
    """True when the top and close live frames are at most FRAME_MAX_AGE s old."""
    now = time.time() if now is None else now
    for view in ("top", "close"):
        try:
            if now - os.path.getmtime(f"{RUN_ROOT}/live_{view}.jpg") > FRAME_MAX_AGE:
                return False
        except OSError:
            return False
    return True


def main():
    log = open(f"{LOG_ROOT}/monitor/hand_watch.log", "a")
    last, hits = {}, {}
    while True:
        try:
            watch_once(log, last, hits)
            if frames_live():
                handdet.write_heartbeat()
        except Exception as e:  # noqa: BLE001  (keep watching; no heartbeat on a failed pass)
            log.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} WATCHDOG ERROR {e!r}\n")
            log.flush()
        time.sleep(0.1)


def watch_once(log, last, hits):
    """One pass: read the live frames that changed, run the hand detector, write HAND? lines and the e-stop."""
    active = os.path.exists(f"{RUN_ROOT}/motion_active")
    st = _state()
    hold = holding(st)
    views = ("top", "close", "wrist") if wrist_usable(st) else ("top", "close")
    for view in views:
        p = f"{RUN_ROOT}/live_{view}.jpg"
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        if last.get(view) == mt:
            continue
        last[view] = mt
        img = cv2.imread(p)
        if img is None:  # frame still being written
            continue
        hand, f = handdet.is_hand(img, view, hold)
        if hand and view == "close" and handdet.far_band_only(f, img.shape[0]) and hits.get("top", 0) == 0:
            hits[view] = 0  # beyond the far edge and the top camera sees no hand: someone behind the table
            log.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} far-band close blob={f['area']}px edge={f['edge']} "
                      f"bbox={f['bbox']} (top view clear: no stop)\n")
            log.flush()
            continue
        hits[view] = hits.get(view, 0) + 1 if hand else 0
        if not hand:
            continue
        msg = (f"{time.strftime('%H:%M:%SZ', time.gmtime())} HAND? view={view} blob={f['area']}px "
               f"edge={f['edge'] or '-'} hsv={f['hue']}/{f['sat']}/{f['val']} active={active} hits={hits[view]}")
        log.write(msg + "\n")
        log.flush()
        if active and hits[view] >= handdet.PERSIST[view]:
            open(f"{RUN_ROOT}/estop", "w").write(msg + "\n")
            subprocess.run(["pkill", "-INT", "-f", "run_sequence.py"], check=False)
            cv2.imwrite(f"{LOG_ROOT}/monitor/estop_{view}_{int(time.time())}.jpg", img)


if __name__ == "__main__":
    main()
