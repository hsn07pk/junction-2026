"""Review aid for one sort cycle: prints the episode's key fields and the loop-log tail, and tiles its phase
snapshots (pre, stage1, grasp_open, closed, lifted, released x top/close/wrist) into one labelled JPEG.
usage: cycle_peek.py RUN [EPISODE]   (default: the newest episode folder)"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP, R  # noqa: E402  (roots from this file's location)
run = sys.argv[1]
base = f"{R}/logs/sort/{run}"
eps = sorted(glob.glob(f"{base}/ep_*"))
ep = f"{base}/ep_{int(sys.argv[2]):04d}" if len(sys.argv) > 2 else eps[-1]
print("episode folder:", ep)
try:
    d = json.load(open(f"{ep}/episode.json"))
    keys = ("episode", "success", "failure", "colour", "shape", "shape_why", "label", "plate", "decision_conf",
            "decision_reasons", "grip_closed", "grip_lifted", "grasp_use_mm", "hover_tip_size_mm", "brick_xy_est",
            "timings", "needs_review")
    for k in keys:
        if k in d:
            print(f"  {k}: {d[k]}")
    pre = d.get("pre_close") or {}
    print("  pre_close:", {k: pre.get(k) for k in ("verdict", "reasons", "colour", "tape_score")})
    for v in d.get("views") or []:
        print("  view:", {k: v.get(k) for k in ("verdict", "reasons", "colour", "tape_score")})
except Exception as e:  # noqa: BLE001
    print("  episode.json:", e)
print("loop log tail:")
try:
    for line in open(f"{base}/loop.log").read().splitlines()[-14:]:
        print("  " + line[:220])
except OSError as e:
    print("  ", e)
rows = []
for phase in ("pre", "stage1", "grasp_open", "closed", "lifted", "station", "released"):
    tiles = []
    for cam in ("top", "close", "wrist"):
        im = cv2.imread(f"{ep}/{phase}_{cam}.jpg")
        im = np.zeros((240, 320, 3), np.uint8) if im is None else cv2.resize(im, (320, 240))
        cv2.putText(im, f"{phase} {cam}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(im)
    rows.append(np.hstack(tiles))
out = f"{R}/logs/depth/servo/cycle_peek.jpg"
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 75])
print("sheet:", out)
