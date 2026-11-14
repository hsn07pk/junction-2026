#!/usr/bin/env python3
"""Run a named sequence from a gm pose file through motion_server.py (no re-hold sag between steps).
usage: mseq.py POSEFILE SEQUENCE [--speed 15] [--comp] [--grip G]   (--grip overrides the gripper value of every step)
Stops at the first step that is not 'ok' (estop/error) and exits non-zero."""
import argparse
import json
import subprocess
import sys

import os  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP, R  # noqa: E402  (roots from this file's location)
ap = argparse.ArgumentParser()
ap.add_argument("posefile")
ap.add_argument("sequence")
ap.add_argument("--speed", type=float, default=15)
ap.add_argument("--comp", action="store_true")
ap.add_argument("--grip", type=float, default=None)
a = ap.parse_args()
d = json.load(open(a.posefile))
steps = d["sequences"][a.sequence]
for i, s in enumerate(steps, 1):
    q = list(d["poses"][s["pose"]])
    if a.grip is not None:
        q[5] = a.grip
    sp = min(a.speed, s.get("speed", a.speed))
    r = subprocess.run([f"{R}/envs/data-env/bin/python", "-B", "mcmd.py",
                        json.dumps({"op": "move", "q": q, "speed": sp, "comp": a.comp})],
                       cwd=f"{APP}/tools", capture_output=True, text=True, timeout=120)
    try:
        st = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        print(f"step {i} {s['pose']}: no state ({r.stdout[-200:]})")
        sys.exit(3)
    print(f"step {i} {s['pose']}: {st['status']} present={st['present']} maxerr="
          f"{max(abs(x) for x in st['err'][:5]):.2f}")
    if st["status"] != "ok":
        sys.exit(2)
