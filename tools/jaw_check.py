#!/usr/bin/env python3
"""Jaw check and static jaw mask for the wrist camera (robust to bricks under the jaws).
At the current pose with the gripper open (25): frames at two wrist rolls; the jaws stay fixed in the wrist image
while the background rotates, so red pixels present in BOTH frames are the jaws. The vector between the two tips is
compared with the reference geometry: a loose or bent fixed jaw changes it (exit 3), a knocked wrist camera only
shifts both tips together. When the jaws are rigid, writes wrist_jaw_mask.png and jaw_calib.json (servo target =
gap centre + 3 px, y 268 + the common vertical shift).
usage: jaw_check.py [--tol 15] [--no-move]"""
import argparse
import os
import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grasp_auto as ga  # noqa: E402

T = os.path.dirname(os.path.abspath(__file__))
REF = [[235.7, 320], [544.6, 329]]  # tips with the fixed jaw tight (wrist camera, gripper 25)


def redmask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r = (((H >= 165) | (H <= 4)) & (S >= 85) & (V >= 55)).astype(np.uint8)
    r[:180] = 0
    return r


def jaws(m, h):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    tips = []
    for i in range(1, n):
        if st[i, 4] > 1500 and st[i, 1] + st[i, 3] >= h - 3:
            keep[lab == i] = 1
            ys, xs = np.nonzero(lab == i)
            top = ys.min()
            tips.append([round(float(xs[ys <= top + 8].mean()), 1), int(top)])
    return keep, sorted(tips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=15.0, help="px change of the tip-to-tip vector still fully ok")
    ap.add_argument("--fail", type=float, default=25.0, help="px change that means a loose or bent jaw")
    ap.add_argument("--no-move", action="store_true")
    ap.add_argument("--accept-bent", action="store_true",
                    help="a bent or loose fixed jaw that cannot be fixed is accepted as it is now (two tips seen in at least two "
                         "roll pairs): the mask covers the jaws at every roll and the servo target follows the gap centre")
    a = ap.parse_args()
    st = ga.state()
    q = list(st["target"])
    q[5] = 25.0
    frames = []
    if not a.no_move:
        ga.move(q, speed=8, comp=True)
        for dr in (0.0, 25.0, -25.0):
            qq = list(q)
            qq[4] = round(max(-95.0, min(95.0, q[4] + dr)), 2)
            ga.move(qq, speed=15, comp=False if dr else True)
            time.sleep(0.4)
            frames.append(ga.fresh("wrist", time.time() + 0.2))
        ga.move(q, speed=15, comp=True)
    else:
        frames.append(ga.fresh("wrist", time.time()))
    grip = round(float(ga.state()["present"][5]), 2)
    pairs = [(0, 1), (0, 2), (1, 2)] if len(frames) == 3 else [(0, 0)]
    meas = []
    for i, j in pairs:  # every pair of rolls gives one measurement; the median ignores one contaminated frame
        r = cv2.morphologyEx(redmask(frames[i]) & redmask(frames[j]), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        k_, t_ = jaws(r, frames[0].shape[0])
        if len(t_) == 2:
            meas.append((k_, t_))
    if meas:
        med = [[float(np.median([m[1][i][0] for m in meas])), float(np.median([m[1][i][1] for m in meas]))] for i in range(2)]
        keep = meas[int(np.argmin([abs(m[1][0][0] - med[0][0]) + abs(m[1][1][0] - med[1][0]) for m in meas]))][0]
        tips = [[round(med[i][0], 1), int(round(med[i][1]))] for i in range(2)]
    else:
        keep, tips = None, []
    out = {"t": time.strftime("%H:%M:%SZ", time.gmtime()), "jaw_tips": tips, "ref": REF, "gripper": grip,
           "n_measurements": len(meas), "spread_px": round(float(max(abs(m[1][0][0] - tips[0][0]) + abs(m[1][1][0] - tips[1][0])
                                                                   for m in meas)), 1) if meas else None}
    if len(tips) != 2:
        out["error"] = f"found {len(tips)} jaws"
        print(json.dumps(out))
        return 4
    d = [float(np.hypot(tips[i][0] - REF[i][0], tips[i][1] - REF[i][1])) for i in range(2)]
    gx = (tips[0][0] + tips[1][0]) / 2
    gap = (tips[1][0] - tips[0][0], tips[1][1] - tips[0][1])
    gap0 = (REF[1][0] - REF[0][0], REF[1][1] - REF[0][1])
    dg = float(np.hypot(gap[0] - gap0[0], gap[1] - gap0[1]))
    common = [round((tips[0][0] - REF[0][0] + tips[1][0] - REF[1][0]) / 2, 1),
              round((tips[0][1] - REF[0][1] + tips[1][1] - REF[1][1]) / 2, 1)]
    out.update({"shift_px": [round(x, 1) for x in d], "gap_change_px": round(dg, 1), "common_shift_px": common,
                "gap_centre_x": round(gx, 1), "target_px": [round(gx + 3, 1), round(268.0 + common[1], 1)]})
    if a.accept_bent and len(meas) >= 2:  # the jaws as they are over all rolls: union of the measured masks, widened by the spread
        keep = np.max([m[0] for m in meas], axis=0).astype(np.uint8)
        w = 17 + 2 * int(np.ceil(min(out["spread_px"] or 0.0, a.fail)))
        mask = cv2.dilate(keep, np.ones((w, w), np.uint8))
    else:
        mask = cv2.dilate(keep, np.ones((17, 17), np.uint8))
    vis = frames[0].copy()
    vis[mask > 0] = (0, 255, 0)
    cv2.circle(vis, (int(gx + 3), 268), 8, (255, 0, 255), 2)
    os.makedirs(os.path.join(ga.LOG_ROOT, "depth", "servo"), exist_ok=True)
    cv2.imwrite(os.path.join(ga.LOG_ROOT, "depth", "servo", "jaw_check.jpg"), vis)
    if a.no_move:  # check only: never rewrite the mask/target from a single frame
        out["check_only"] = True
        out["ok"] = dg <= a.tol
        print(json.dumps(out))
        return 0 if out["ok"] else 3
    if a.accept_bent and out["spread_px"] is not None and out["spread_px"] > 2 * a.fail:
        out["ok"] = False  # a red brick next to the jaws joined a jaw in some roll: keep the last accepted jaws
        out["error"] = f"jaw check contaminated: tips spread {out['spread_px']}px over the rolls (a red object next to the jaws)"
        print(json.dumps(out))
        return 3
    if dg > a.fail and a.accept_bent and len(meas) >= 2:  # loose jaw that cannot be fixed: the jaws as they are now
        out["bent_accepted"] = f"jaw gap {dg:.0f}px from the tight reference, {out['spread_px']}px spread over the rolls"
    elif dg > a.fail:
        out["ok"] = False
        out["error"] = f"the jaws moved {dg:.0f}px relative to each other (fixed jaw loose or bent): tighten the fixed jaw"
        print(json.dumps(out))
        return 3
    if dg > a.tol:
        out["warning"] = f"jaw gap drifting ({dg:.0f}px from the tight reference): tighten the fixed jaw at the next stop"
    out["ok"] = True
    cv2.imwrite(f"{T}/wrist_jaw_mask.png", mask * 255)
    json.dump(out, open(f"{T}/jaw_calib.json", "w"), indent=1)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
