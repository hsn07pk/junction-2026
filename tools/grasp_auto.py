#!/usr/bin/env python3
"""Automatic spot-cell grasp: the verified 16:46-16:54Z procedure (docs/SPOT-GRASP-DEPTH.md) as one routine.

Needs: motion_server.py running, live JPEGs (monitor_cams.sh, wrist_rec.sh), hand_watch.py.
usage: grasp_auto.py --plate green|blue|none [--from-home] [--dry]
  --dry       perception at the hover only (moves only to the hover and back up nothing)
Exit: 0 placed (or held with --plate none), 10 no brick, 11 align failed, 12 grasp missed, 13 lift check failed,
      2 estop / motion error (arm held where it stopped).
Every phase saves frames, depth and numbers to logs/grasp_auto/<HHMMSS>/ and summary.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rs_height as rh  # noqa: E402
from paths import LOG_ROOT, RUN_ROOT, SIM_ROOT  # noqa: E402,F401

from paths import APP, R  # noqa: E402  (roots from this file's location)
T = f"{APP}/tools"
PY = f"{R}/envs/data-env/bin/python"
SD = f"{RUN_ROOT}/mserver"
HOVER = [-12.62, 30.0, -19.33, 69.73, -45.65, 25.0]
TARGET_PX = (392.0, 268.0)   # wrist-cam brick centre at the hover when the jaws are centred (16:46Z run)
try:  # re-measured target (jaw_calib.py): gap centre + 3 px, y 268
    TARGET_PX = tuple(json.load(open(os.path.join(APP, "tools", "jaw_calib.json")))["target_px"])
except Exception:  # noqa: BLE001
    pass
TOL_PX = 9.0
YAW_PER_ROLL = 1.37          # deg of wrist-cam brick yaw per roll unit (+)
WF_PER_EL = -0.973           # constant-pitch descent: d_wf = WF_PER_EL * d_el
STAGES = [11.5, 18.0]        # cumulative d_el from the centred hover: tips ~31 mm, ~19.5 mm
EL_MM = 1.75                 # mm of descent per el unit (measured 1.70-1.77)
TIP_TARGET = 6.0             # mm above the mat for the closed jaws
LIFT_DEL = -8.5
SPOT_DEPTH_PX = (222, 122)   # brick centre in the close 424x240 depth image at the spot
OUT = ""
SUM = {"phases": []}
_jm = cv2.imread(f"{T}/wrist_jaw_mask.png", cv2.IMREAD_GRAYSCALE)
JAW_MASK = (_jm > 0) if _jm is not None else None  # static: the wrist camera rides on the gripper (open = 25)


class Stop(Exception):
    pass


def note(phase, **kw):
    kw["phase"] = phase
    kw["t"] = time.strftime("%H:%M:%SZ", time.gmtime())
    SUM["phases"].append(kw)
    print(json.dumps(kw), flush=True)
    json.dump(SUM, open(f"{OUT}/summary.json", "w"), indent=1)


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


def move(q, speed=5.0, comp=True):
    st = cmd({"op": "move", "q": [float(x) for x in q], "speed": speed, "comp": comp})
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


def wrist_brick(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h, w = H.shape
    jaw = JAW_MASK if JAW_MASK is not None and JAW_MASK.shape == H.shape else None
    if jaw is None:  # fallback: colour mask of the lower image (fails for red/brown bricks next to the jaws)
        red = ((H <= 8) | (H >= 170)) & (S >= 110) & (V >= 60)
        jaw = np.zeros_like(red)
        jaw[int(h * 0.6):, :] = red[int(h * 0.6):, :]
        jaw = cv2.dilate(jaw.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    fg = (((S >= 90) & (V >= 70)) | (V >= 150)) & ~jaw
    fg = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(fg)
    best = None
    for i in range(1, n):
        if st[i, 4] < 2500:
            continue
        d = float(np.hypot(cen[i][0] - TARGET_PX[0], cen[i][1] - TARGET_PX[1]))
        if best is None or d < best[0]:
            best = (d, i)
    if best is None:
        return None
    ys, xs = np.nonzero(lab == best[1])
    rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    (cx, cy), (sw, sh), ang = rect
    yaw = ang % 90.0
    if yaw > 45:
        yaw -= 90.0
    box = cv2.boxPoints(rect)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    lng = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    long_vertical = abs(lng[1]) >= abs(lng[0])
    return {"centre": [round(float(cx), 1), round(float(cy), 1)], "size": [round(float(sw), 1), round(float(sh), 1)],
            "area": int(len(xs)), "yaw": round(float(yaw), 1), "aspect": round(float(max(sw, sh) / max(1.0, min(sw, sh))), 2),
            "long_vertical": bool(long_vertical), "dist_to_target": round(best[0], 1)}


def depth(tag):
    d = f"{OUT}/{tag}_depth"
    os.makedirs(d, exist_ok=True)
    subprocess.run([PY, "-B", "rs_grab.py", "--cam", "close", "--out", d, "--mode", "424x240@15", "--frames", "20",
                    "--skip", "10", "--no-ir"], cwd=T, capture_output=True, text=True, timeout=60)
    D = np.load(f"{d}/close_depth.npy")
    h, w = D.shape
    fx, fy, cx, cy, _ = rh.intrinsics(w, h, 65.0, "")
    P = rh.cloud(D, fx, fy, cx, cy)
    valid = np.isfinite(D)
    pts = P[valid]
    pts = pts[np.random.default_rng(1).choice(len(pts), min(40000, len(pts)), replace=False)]
    n, dd, _, rms = rh.fit_plane(pts)
    Hm = np.where(valid, np.einsum("ijk,k->ij", np.nan_to_num(P), n) + dd, np.nan)
    np.save(f"{d}/height.npy", Hm)
    return Hm, P, n, rms


def brick_blob(Hm, near=SPOT_DEPTH_PX, max_d=60):
    m = (np.nan_to_num(Hm, nan=-9) > 12) & (np.nan_to_num(Hm, nan=99) < 27)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k, lab, st, cen = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, k):
        if st[i, 4] < 250:
            continue
        dd = float(np.hypot(cen[i][0] - near[0], cen[i][1] - near[1]))
        if dd <= max_d and (best is None or dd < best[0]):
            best = (dd, i)
    if best is None:
        return None
    i = best[1]
    hv = Hm[lab == i]
    return {"bbox": [int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])], "area": int(st[i, 4]),
            "centre": [round(float(cen[i][0]), 1), round(float(cen[i][1]), 1)],
            "top_h": round(float(np.percentile(hv[hv > 15], 75)) if (hv > 15).any() else float("nan"), 1),
            "mask": lab == i}


def jaw_tip_low(Hm, bb):
    """Lowest jaw height beside the brick (left and right boxes, brick pixels excluded)."""
    x0, y0, w, h = bb["bbox"]
    res = {}
    for side, (a, b) in (("left", (max(0, x0 - 45), x0 - 2)), ("right", (x0 + w + 2, min(Hm.shape[1], x0 + w + 45)))):
        B = Hm[max(0, y0 - 70):y0 + h + 25, a:b]
        m = np.isfinite(B) & (B > 2.5)
        if m.sum() >= 15:
            res[side] = round(float(np.percentile(B[m], 2)), 1)
    return res


def jaws_clear_of_brick(Hm, P, n, bb, thr=24.0):
    """Min horizontal distance (mm) between jaw points above thr and the brick top-face points."""
    ex = np.array([1.0, 0, 0]) - n * n[0]
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    top = bb["mask"] & (np.nan_to_num(Hm, nan=0) > 15)
    jaw = np.nan_to_num(Hm, nan=0) > thr
    jaw[int(Hm.shape[0] * 0.85):, :] = False
    if top.sum() < 20 or jaw.sum() < 20:
        return None
    bt = np.stack([P[top] @ ex, P[top] @ ey], 1)
    jp = np.stack([P[jaw] @ ex, P[jaw] @ ey], 1)
    jp = jp[np.random.default_rng(0).choice(len(jp), min(3000, len(jp)), replace=False)]
    bt = bt[np.random.default_rng(1).choice(len(bt), min(800, len(bt)), replace=False)]
    dmin = np.min(np.linalg.norm(jp[:, None, :] - bt[None, :, :], axis=2))
    return round(float(dmin), 1)


def state():
    return cmd({"op": "state"})


def run_named(posefile, seq, speed=15.0):
    d = json.load(open(posefile))
    for s in d["sequences"][seq]:
        move(d["poses"][s["pose"]], speed=min(speed, s.get("speed", speed)), comp=False)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", choices=["green", "blue", "none"], required=True)
    ap.add_argument("--from-home", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    OUT = f"{LOG_ROOT}/grasp_auto/{time.strftime('%H%M%S', time.gmtime())}"
    os.makedirs(OUT, exist_ok=True)
    SUM.update({"plate": a.plate, "dry": a.dry, "start": time.strftime("%H:%M:%SZ", time.gmtime())})
    if not os.path.exists(f"{SD}/pid"):
        print("motion server not running")
        return 2
    try:
        if a.from_home:
            move([HOVER[0], -40.0, 80.0, 66.0, -45.0, 25.0], speed=12, comp=False)
        st = move(HOVER, speed=8, comp=True)
        q = list(HOVER)
        imgs = snap("hover")
        Hm, P, n, rms = depth("hover")
        bb = brick_blob(Hm)
        wb = wrist_brick(imgs["wrist"]) if imgs["wrist"] is not None else None
        note("hover", present=st["present"], plane_rms=round(rms, 2),
             brick_depth={k: v for k, v in (bb or {}).items() if k != "mask"}, wrist=wb)
        if bb is None or not (15 <= bb["top_h"] <= 26) or wb is None:
            note("result", code=10, reason="no brick at the spot (depth or wrist cam)")
            return 10
        # yaw: square -> nearest face; elongated -> long side vertical in the wrist image (jaws close across the short side)
        for it in range(3):
            yaw = wb["yaw"]
            if wb["aspect"] > 1.4 and not wb["long_vertical"]:
                yaw = yaw + 90.0 if yaw < 0 else yaw - 90.0
            if abs(yaw) <= 2.5 and (wb["aspect"] <= 1.4 or wb["long_vertical"]):
                break
            q[4] = round(q[4] - yaw / YAW_PER_ROLL, 2)
            move(q, speed=8, comp=True)
            wb = wrist_brick(snap(f"yaw{it}")["wrist"])
            note("yaw", iter=it, roll=q[4], wrist=wb)
            if wb is None:
                note("result", code=11, reason="brick lost in the wrist cam during yaw alignment")
                return 11
        # image Jacobian at this roll: probe pan and wf
        J = np.zeros((2, 2))
        for col, (j, dq) in enumerate(((0, 2.0), (3, 3.0))):
            qq = list(q)
            qq[j] += dq
            move(qq, speed=6, comp=True)
            wp = wrist_brick(snap(f"probe{j}")["wrist"])
            move(q, speed=6, comp=True)
            if wp is None:
                note("result", code=11, reason="brick lost during probe")
                return 11
            J[:, col] = [(wp["centre"][0] - wb["centre"][0]) / dq, (wp["centre"][1] - wb["centre"][1]) / dq]
        note("jacobian_px_per_unit", pan=J[:, 0].round(2).tolist(), wf=J[:, 1].round(2).tolist())
        for it in range(4):
            err = np.array(TARGET_PX) - np.array(wb["centre"])
            if np.linalg.norm(err) <= TOL_PX:
                break
            dq = np.linalg.lstsq(J, err, rcond=None)[0]
            dq = np.clip(dq, [-4.0, -8.0], [4.0, 8.0])
            q[0] = round(q[0] + dq[0], 2)
            q[3] = round(q[3] + dq[1], 2)
            move(q, speed=6, comp=True)
            wb = wrist_brick(snap(f"centre{it}")["wrist"])
            note("centre", iter=it, d_pan=round(float(dq[0]), 2), d_wf=round(float(dq[1]), 2), q=q, wrist=wb)
            if wb is None:
                note("result", code=11, reason="brick lost while centring")
                return 11
        if np.linalg.norm(np.array(TARGET_PX) - np.array(wb["centre"])) > 2 * TOL_PX or abs(wb["yaw"]) > 4 and wb["aspect"] <= 1.4:
            note("result", code=11, reason="could not centre/align", wrist=wb)
            return 11
        hover_c = list(q)
        note("aligned", q=hover_c, wrist=wb)
        if a.dry:
            note("result", code=0, reason="dry run: aligned at hover")
            return 0
        # stage 1 (tips ~31 mm, above the 19 mm brick top): jaws must be beside the brick
        q = list(hover_c)
        q[2], q[3] = round(hover_c[2] + STAGES[0], 2), round(hover_c[3] + WF_PER_EL * STAGES[0], 2)
        move(q, speed=4, comp=True)
        imgs = snap("stage1")
        Hm, P, n, _ = depth("stage1")
        bb1 = brick_blob(Hm, near=tuple(bb["centre"]), max_d=25) or bb
        clear = jaws_clear_of_brick(Hm, P, n, bb1)
        wb1 = wrist_brick(imgs["wrist"])
        note("stage1", q=q, jaw_low=jaw_tip_low(Hm, bb1), jaw_brick_clear_mm=clear, wrist=wb1)
        if clear is not None and clear < 2.0:
            move(hover_c, speed=4, comp=True)
            note("result", code=11, reason=f"jaw over the brick (clearance {clear} mm), backed off to hover")
            return 11
        # stage 2 (tips ~19.5 mm)
        q[2], q[3] = round(hover_c[2] + STAGES[1], 2), round(hover_c[3] + WF_PER_EL * STAGES[1], 2)
        move(q, speed=3, comp=True)
        snap("stage2")
        Hm, P, n, _ = depth("stage2")
        low = jaw_tip_low(Hm, bb1)
        h2 = min(low.values()) if low else None
        extra = (h2 - TIP_TARGET) / EL_MM if h2 is not None and 10 <= h2 <= 30 else (24.8 - STAGES[1])
        extra = float(np.clip(extra, 3.0, 10.0))
        note("stage2", q=q, jaw_low=low, final_extra_el=round(extra, 2))
        q[2], q[3] = round(hover_c[2] + STAGES[1] + extra, 2), round(hover_c[3] + WF_PER_EL * (STAGES[1] + extra), 2)
        st = move(q, speed=3, comp=True)
        snap("grasp_open")
        Hm, P, n, _ = depth("grasp_open")
        note("grasp_open", q=q, present=st["present"], err=st["err"], jaw_low=jaw_tip_low(Hm, bb1))
        st = cmd({"op": "grip", "value": 0.0, "speed": 15})
        time.sleep(0.6)
        st = state()
        g = st["present"][5]
        imgs = snap("closed")
        note("closed", gripper=g, present=st["present"])
        if g < 1.2:
            cmd({"op": "grip", "value": 25.0, "speed": 15})
            move(hover_c, speed=4, comp=True)
            note("result", code=12, reason=f"gripper closed to {g}: nothing held")
            return 12
        q[5] = 0.0
        ql = list(q)
        ql[2], ql[3] = round(q[2] + LIFT_DEL, 2), round(q[3] + WF_PER_EL * LIFT_DEL, 2)
        move(ql, speed=3, comp=True)
        imgs = snap("lifted")
        Hm, P, n, _ = depth("lifted")
        left = bb1["mask"] & (np.nan_to_num(Hm, nan=0) > 15) & (np.nan_to_num(Hm, nan=99) < 26)
        top0 = int((bb1["mask"]).sum())
        wbl = wrist_brick(imgs["wrist"])
        g = state()["present"][5]
        note("lifted", q=ql, gripper=g, spot_pixels_left=int(left.sum()), spot_pixels_before=top0, wrist=wbl)
        if left.sum() > 0.35 * top0 or g < 1.0:
            cmd({"op": "grip", "value": 25.0, "speed": 15})
            move(hover_c, speed=4, comp=True)
            note("result", code=13, reason="brick still on the mat after lift (or slipped)")
            return 13
        if a.plate == "none":
            note("result", code=0, reason="held and lifted (no plate requested)")
            return 0
        move([ql[0], -40.0, 80.0, 66.0, -45.0, 0.0], speed=12, comp=False)
        move([-6.0, -98.0, 96.6, 66.0, -45.0, 0.0], speed=12, comp=False)
        pf = f"{APP}/motion/gm/spot_fast.yaml"
        run_named(pf, f"home_to_{a.plate}_carry")
        run_named(pf, f"release_{a.plate}")
        time.sleep(0.4)
        snap("released")
        run_named(pf, f"{a.plate}_to_home")
        note("result", code=0, reason=f"placed on {a.plate}")
        return 0
    except Stop as e:
        note("result", code=2, reason=str(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
