#!/usr/bin/env python3
"""Precise motion server for the SO-101 follower (one persistent connection).

Why: every run_sequence call connects with goal=present and disconnects with goal=present
(physicalai_so101_plugin so101.py disconnect -> _hold_position), so gravity-loaded joints sag by the servo's
steady-state error at each connect and disconnect. Measured 2026-11-14 16:37Z: the same commanded pose gave jaw
tips 11 mm lower and 5 mm shifted (max_error 4-5 units). This server keeps one connection, ramps from the last
commanded goal, removes the steady-state error with a clamped outer integral loop (goal offset), and on exit
closes the port WITHOUT re-holding at the sagged present pose.

Commands: one JSON object per line appended to run/mserver/cmd.jsonl
  {"id": 1, "op": "move", "q": [pan, sl, el, wf, roll, grip], "speed": 5, "comp": true}
  {"id": 2, "op": "rel", "dq": [0, 1.5, 0, -0.5, 0, 0], "speed": 4, "comp": true}   relative to the last target
  {"id": 3, "op": "grip", "value": 0.0, "speed": 20}    gripper only, body goal kept, no compensation
  {"id": 4, "op": "state"}
  {"id": 5, "op": "exit"}      torque stays on at the current goal, port closed
  {"id": 6, "op": "release"}   torque off, port closed
State after each command: run/mserver/state.json
Safety: run/estop -> stop sending at once and hold at present; moves are refused until it is removed.
run/motion_active exists only while the arm moves (the hand watchdog acts only then).
"""
import json
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "motion"))
from so101_motion import (  # noqa: E402
    MotionPlayer,
    claim_devices,
    device_ids_for,
    open_follower_held,
    plan_ramp,
    resolve_arm,
    validate_target,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP, R  # noqa: E402  (roots from this file's location)
D = f"{R}/run/mserver"
ESTOP = f"{R}/run/estop"
ACTIVE = f"{R}/run/motion_active"
HEARTBEAT = f"{R}/run/hand_watch_alive"  # written by hand_watch.py on every pass with live frames
HEARTBEAT_MAX_AGE = 3.0
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00"
GRIP = 5
def _limits():
    try:
        lim = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                                          "motion_profiles.json")))["limits"]
        return float(lim["hard_speed"]), float(lim["hard_accel"])
    except (OSError, ValueError, KeyError):
        return 45.0, 200.0


HARD_SPEED, HARD_ACCEL = _limits()
OFFSET_CLAMP = 8.0
STOP = {"flag": False}


def log(msg):
    with open(f"{D}/server.log", "a") as f:
        f.write(time.strftime("%H:%M:%SZ ", time.gmtime()) + msg + "\n")


def watchdog_alive():
    """True while hand_watch.py's heartbeat is at most HEARTBEAT_MAX_AGE s old: no move without a live watchdog."""
    try:
        with open(HEARTBEAT) as f:
            t = float(f.read())
    except (OSError, ValueError):
        return False
    return 0.0 <= time.time() - t <= HEARTBEAT_MAX_AGE


class Server:
    def __init__(self, pl):
        self.pl = pl
        self.goal = pl.current()
        self.target = self.goal.copy()
        self.offset = np.zeros(6)
        self.status = "ready"
        self.rec = None
        self.rec_last = 0.0

    def rec_tick(self, goal, force=False):
        if self.rec is None:
            return
        now = time.time()
        if not force and now - self.rec_last < 1.0 / 30 - 0.002:
            return
        try:
            cur = self.pl.current()
        except Exception:
            return
        self.rec_last = now
        self.rec.write("%.4f," % now + ",".join("%.2f" % x for x in goal) + "," + ",".join("%.2f" % x for x in cur) + "\n")

    def estop(self):
        return os.path.exists(ESTOP) or STOP["flag"] or not watchdog_alive()

    def send_for(self, goal, seconds):
        n = max(1, int(round(seconds / self.pl.dt)))
        return self.paced(np.repeat(goal.reshape(1, -1), n, axis=0))

    def paced(self, setpoints):
        nxt = time.monotonic()
        for sp in setpoints:
            if self.estop():
                try:
                    self.pl.hold()
                    cur = self.pl.current()
                    if self.target[GRIP] < 5:  # keep squeezing a held brick through an e-stop (17:47Z drop)
                        cur[GRIP] = self.target[GRIP]
                        self.pl.driver.send_action(cur.astype(np.float32))
                except Exception:
                    pass
                self.goal = self.pl.current()
                if self.target[GRIP] < 5:
                    self.goal[GRIP] = self.target[GRIP]
                self.status = "estop"
                log("ESTOP during motion - held at present")
                return False
            self.pl.driver.send_action(sp.astype(np.float32))
            self.goal = sp.copy()
            self.rec_tick(self.goal)
            nxt = max(nxt, time.monotonic()) + self.pl.dt
            time.sleep(max(0.0, nxt - time.monotonic()))
        return True

    def move(self, q, speed, comp, keep_grip=False, accel=60.0, keep_offset=False):
        if self.estop():
            self.status = "estop"
            return "refused: estop present" if os.path.exists(ESTOP) else "refused: hand watchdog not running"
        q = np.asarray(q, dtype=np.float64)
        if keep_grip:
            q[GRIP] = self.target[GRIP]
        validate_target(q)
        speed = min(float(speed), HARD_SPEED)
        if not comp and not keep_offset:
            self.offset[:] = 0  # offsets are pose dependent: never carry them into an uncompensated (transit) move
        g = q.copy()
        g[:GRIP] += self.offset[:GRIP]
        try:
            validate_target(g)
        except Exception:
            self.offset[:] = 0
            g = q.copy()
        open(ACTIVE, "w").close()
        try:
            ok = self.paced(plan_ramp(self.goal, g, speed, self.pl.dt, max_accel=min(float(accel), HARD_ACCEL)))
            ok = ok and self.send_for(g, 0.5)
            its = 0
            while ok and comp and its < 10:
                cur = self.pl.current()
                err = q - cur
                err[GRIP] = 0.0
                if np.max(np.abs(err[:GRIP])) <= 0.5:
                    break
                self.offset[:GRIP] = np.clip(self.offset[:GRIP] + 0.7 * err[:GRIP], -OFFSET_CLAMP, OFFSET_CLAMP)
                g = q.copy()
                g[:GRIP] += self.offset[:GRIP]
                try:
                    validate_target(g)
                except Exception:
                    break
                ok = self.send_for(g, 0.35)
                its += 1
        finally:
            if os.path.exists(ACTIVE):
                os.remove(ACTIVE)
        if ok:
            self.target = q
            self.status = "ok"
        else:  # interrupted: re-anchor at the present pose so the next rel or grip starts from where the arm is
            held = self.pl.current()
            held[GRIP] = self.goal[GRIP]
            self.target = np.asarray(held, dtype=np.float64)
        resid = float(np.max(np.abs((q - self.pl.current())[:GRIP]))) if ok else float("nan")
        return f"comp_iters={its if comp else 0} resid={resid:.2f}"

    def state(self, cid, msg):
        cur = self.pl.current()
        st = {"last_id": cid, "status": self.status, "msg": msg, "t": time.strftime("%H:%M:%SZ", time.gmtime()),
              "target": [round(float(x), 2) for x in self.target], "goal": [round(float(x), 2) for x in self.goal],
              "present": [round(float(x), 2) for x in cur], "err": [round(float(x), 2) for x in (self.target - cur)],
              "offset": [round(float(x), 2) for x in self.offset]}
        tmp = f"{D}/state.tmp"
        json.dump(st, open(tmp, "w"))
        os.replace(tmp, f"{D}/state.json")
        log(json.dumps(st))


def main():
    os.makedirs(D, exist_ok=True)
    cmdf = f"{D}/cmd.jsonl"
    open(cmdf, "a").close()
    pos = os.path.getsize(cmdf)
    signal.signal(signal.SIGTERM, lambda *_: STOP.__setitem__("flag", True))
    signal.signal(signal.SIGINT, lambda *_: STOP.__setitem__("flag", True))
    port, cal = resolve_arm(studio_export=f"{APP}/motion/project_robots.json", name="SO101 Follower", port=PORT,
                            follower=True)
    with claim_devices(device_ids_for(port), ports=(port,)):
        drv = open_follower_held(port, cal)
        pl = MotionPlayer(drv, max_speed=HARD_SPEED, tolerance=3.0)
        srv = Server(pl)
        open(f"{D}/pid", "w").write(str(os.getpid()))
        srv.state(0, "started")
        release = False
        try:
            while not STOP["flag"]:
                with open(cmdf) as f:
                    f.seek(pos)
                    lines = f.readlines()
                    pos = f.tell()
                if not lines:
                    srv.rec_tick(srv.goal)
                    time.sleep(0.01 if srv.rec is not None else 0.05)
                    continue
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        c = json.loads(line)
                        cid, op = c.get("id", -1), c.get("op")
                        if op == "move":
                            msg = srv.move(c["q"], c.get("speed", 5), c.get("comp", True), accel=c.get("accel", 60.0))
                        elif op == "rec_start":
                            if srv.rec is not None:
                                srv.rec.close()
                            os.makedirs(os.path.dirname(c["path"]), exist_ok=True)
                            srv.rec = open(c["path"], "w", buffering=1)
                            srv.rec.write("t," + ",".join(f"goal{i}" for i in range(6)) + "," + ",".join(f"pos{i}" for i in range(6)) + "\n")
                            srv.rec_tick(srv.goal, force=True)
                            msg = "rec_start " + c["path"]
                        elif op == "rec_stop":
                            if srv.rec is not None:
                                srv.rec_tick(srv.goal, force=True)
                                srv.rec.close()
                            srv.rec = None
                            msg = "rec_stop"
                        elif op == "rel":
                            msg = srv.move(srv.target + np.asarray(c["dq"], dtype=np.float64), c.get("speed", 5),
                                           c.get("comp", True), accel=c.get("accel", 60.0))
                        elif op == "grip":
                            q = srv.target.copy()
                            q[GRIP] = float(c["value"])
                            msg = srv.move(q, c.get("speed", 20), False, accel=c.get("accel", 120.0), keep_offset=True)
                        elif op == "state":
                            msg = "state"
                        elif op in ("exit", "release"):
                            release = op == "release"
                            srv.state(cid, op)
                            STOP["flag"] = True
                            break
                        else:
                            msg = f"unknown op {op}"
                    except Exception as e:  # report and keep serving; the arm holds its last goal
                        srv.status = "error"
                        msg = f"error: {e!r}"
                    srv.state(cid, msg)
        finally:
            try:
                if release:
                    drv.set_torque(enabled=False)
                    log("released: torque off")
                conn = drv._connection
                drv._connection = None
                if conn is not None:
                    conn.port_handler.closePort()
                log("port closed (no re-hold)")
            except Exception as e:
                log(f"close error {e!r}")
            for p in (f"{D}/pid",):
                if os.path.exists(p):
                    os.remove(p)


if __name__ == "__main__":
    main()
