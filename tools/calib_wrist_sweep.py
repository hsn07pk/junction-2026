#!/usr/bin/env python3
"""Re-calibrate the wrist camera on the robot after it moved on the gripper.

One brick on the mat. The arm finds it from the scan pose and centres it at the hover exactly as the sort loop does,
then photographs it from a grid of poses: three heights, lateral steps at each height, two wrist rolls and two wrist
tilts. Each frame gives the brick's blob centre and the measured joints. A bundle fit (Levenberg-Marquardt, Huber
weights) refines the camera's pose on the gripper and its focal length, with the brick position as an unknown; the
image centre and the radial term are kept close to the current model by priors and the arm's joint offsets stay as
they are (the arm did not change, the camera did). The new model is written only when the held-out error
(5-fold, by view) is small; the previous model is kept as wrist_cam_model.prev.json.

usage: calib_wrist_sweep.py [--write] [--obs FILE (refit saved observations, no robot)]"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import cv2
import numpy as np

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, T)
import kin  # noqa: E402

APP = os.path.dirname(T)
sys.path.insert(0, os.path.join(APP, "eyeinhand"))
import wristcam as wc  # noqa: E402

BRICK_MM = 31.8          # 2x2 footprint, only used for the scale check of the report
PRIOR = {"cx": 10.0, "cy": 10.0, "k1": 0.03}  # sd of the intrinsic priors around the current model
MIN_OBS = 12
MAX_MEDIAN_PX = 5.0       # in-sample (the model fitted on a rigid camera mount scored 4.0 px)
MAX_HELDOUT_PX = 6.0      # 5-fold, by view
OUTLIER_PX = 8.0          # views with a larger residual after the first fit are checked against 3 x the median


# ---------------------------------------------------------------- model

def rod(r) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]


def project(th: np.ndarray, q, P, dq) -> np.ndarray:
    """th = [f, cx, cy, k1, r(3), t(3)]: pixel of base-frame point P seen from joints q (normalized)."""
    f, cx, cy, k1 = th[:4]
    p, Rm = kin.fk(list(np.asarray(list(q)[:5], float) + dq))
    p, Rm = np.asarray(p, float), np.asarray(Rm, float)
    Rbc = Rm @ rod(th[4:7])
    cb = p + Rm @ th[7:10]
    X = Rbc.T @ (np.asarray(P, float) - cb)
    if X[2] <= 1e-3:
        return np.array([1e4, 1e4]), -1.0
    x, y = X[0] / X[2], X[1] / X[2]
    d = 1.0 + k1 * (x * x + y * y)
    return np.array([f * x * d + cx, f * y * d + cy]), float(X[2])


def theta_of(M: dict) -> np.ndarray:
    return np.array([M["f"], M["cx"], M["cy"], M["k1"], *M["rvec_tcp_cam"], *M["t_tcp_cam_mm"]], float)


def residuals(x: np.ndarray, th0: np.ndarray, obs: list, dq, z: float) -> np.ndarray:
    th, bxy = x[:10], x[10:12]
    P = np.array([bxy[0], bxy[1], z])
    r = [project(th, o["q"], P, dq)[0] - np.asarray(o["u"], float) for o in obs]
    r.append(np.array([(th[1] - th0[1]) / PRIOR["cx"], (th[2] - th0[2]) / PRIOR["cy"], (th[3] - th0[3]) / PRIOR["k1"]]))
    return np.concatenate(r)


def lm(x0: np.ndarray, th0, obs, dq, z, iters=60, huber=3.0):
    x = np.asarray(x0, float).copy()
    n_px = 2 * len(obs)

    def cost(r):
        a = np.abs(r[:n_px])
        rho = np.where(a <= huber, 0.5 * a * a, huber * (a - 0.5 * huber))
        return float(rho.sum() + 0.5 * np.sum(r[n_px:] ** 2))

    r = residuals(x, th0, obs, dq, z)
    c, lam = cost(r), 1e-3
    for _ in range(iters):
        a = np.abs(r[:n_px])
        w = np.ones_like(r)
        w[:n_px] = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-9))
        J = np.empty((r.size, x.size))
        for j in range(x.size):
            h = 1e-6 * max(1.0, abs(x[j]))
            xp = x.copy()
            xp[j] += h
            J[:, j] = (residuals(xp, th0, obs, dq, z) - r) / h
        A = J.T @ (w[:, None] * J)
        g = J.T @ (w * r)
        improved = False
        for _ in range(10):
            dx = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            xn = x + dx
            rn = residuals(xn, th0, obs, dq, z)
            cn = cost(rn)
            if cn < c:
                x, r, lam, improved = xn, rn, max(lam / 3.0, 1e-7), True
                if c - cn < 1e-9 * max(c, 1.0):
                    return x, cn
                c = cn
                break
            lam *= 4.0
        if not improved:
            break
    return x, c


def fit(obs: list, M: dict, z: float):
    """Multi-start fit from the current model (focal length and camera depth on the gripper varied)."""
    dq = np.asarray(M.get("joint_offsets_norm", [0, 0, 0, 0, 0]), float)
    th0 = theta_of(M)
    o0 = min(obs, key=lambda o: np.hypot(o["u"][0] - th0[1], o["u"][1] - th0[2]))
    best = None
    for fs in (0.8, 1.0, 1.25, 1.5):
        for dt in (-15.0, 0.0, 15.0):
            th = th0.copy()
            th[0] *= fs
            th[9] += dt
            xy = _pixel_to_plane(th, o0["q"], o0["u"], dq, z)
            x, c = lm(np.concatenate([th, xy]), th0, obs, dq, z)
            if best is None or c < best[1]:
                best = (x, c)
    return best[0], dq, th0


def _pixel_to_plane(th, q, u, dq, z):
    f, cx, cy, k1 = th[:4]
    xd, yd = (u[0] - cx) / f, (u[1] - cy) / f
    x, y = xd, yd
    for _ in range(8):
        d = 1.0 + k1 * (x * x + y * y)
        x, y = xd / d, yd / d
    p, Rm = kin.fk(list(np.asarray(list(q)[:5], float) + dq))
    p, Rm = np.asarray(p, float), np.asarray(Rm, float)
    Rbc = Rm @ rod(th[4:7])
    cb = p + Rm @ th[7:10]
    ray = Rbc @ np.array([x, y, 1.0])
    s = (z - cb[2]) / ray[2] if abs(ray[2]) > 1e-6 else 0.0
    return (cb + s * ray)[:2]


def pixel_errors(x, obs, dq, z):
    P = np.array([x[10], x[11], z])
    return np.array([np.linalg.norm(project(x[:10], o["q"], P, dq)[0] - np.asarray(o["u"], float)) for o in obs])


def heldout_errors(obs, M, z, k=5):
    idx = np.arange(len(obs))
    errs = []
    for i in range(k):
        test = idx[i::k]
        train = [obs[j] for j in idx if j not in set(test.tolist())]
        if len(train) < MIN_OBS - 2 or not len(test):
            continue
        x, dq, _ = fit(train, M, z)
        errs.extend(pixel_errors(x, [obs[j] for j in test], dq, z).tolist())
    return np.asarray(errs)


def evaluate(obs: list, M: dict):
    """Fit, drop views whose blob was contaminated (residual above max(OUTLIER_PX, 3 x median): the brick blob merged
    with a jaw or was cut by the image border), refit on the rest, then score in-sample and held-out."""
    z = float(M["z_top_mm"])
    x, dq, th0 = fit(obs, M, z)
    e_all = pixel_errors(x, obs, dq, z)
    lim = max(OUTLIER_PX, 3.0 * float(np.median(e_all)))
    keep = [o for o, e in zip(obs, e_all) if e <= lim]
    dropped = [o.get("tag") for o, e in zip(obs, e_all) if e > lim]
    if len(keep) >= MIN_OBS and dropped:
        obs = keep
        x, dq, th0 = fit(obs, M, z)
    else:
        dropped = []
    e_in = pixel_errors(x, obs, dq, z)
    e_old = pixel_errors(np.concatenate([th0, _fit_brick_only(th0, obs, dq, z)]), obs, dq, z)
    e_ho = heldout_errors(obs, M, z)
    scale = []
    for o in obs:
        if o.get("tag", "").endswith("_c") and o.get("short"):
            _, depth = project(x[:10], o["q"], np.array([x[10], x[11], z]), dq)
            if depth > 0:
                scale.append(float(o["short"]) / (x[0] * BRICK_MM / depth))
    return x, dq, {"n_obs": len(obs), "dropped_views": dropped, "median_px": float(np.median(e_in)),
                   "p90_px": float(np.percentile(e_in, 90)),
                   "heldout_median_px": float(np.median(e_ho)) if e_ho.size else None,
                   "heldout_p90_px": float(np.percentile(e_ho, 90)) if e_ho.size else None,
                   "old_model_median_px": float(np.median(e_old)), "brick_xy_mm": [round(float(v), 1) for v in x[10:12]],
                   "blob_short_over_predicted": [round(s, 3) for s in scale]}


def _fit_brick_only(th, obs, dq, z):
    """Best brick position under a fixed camera (to report how far off the current model is)."""
    xy = _pixel_to_plane(th, obs[0]["q"], obs[0]["u"], dq, z)
    for _ in range(20):
        r0 = np.concatenate([project(th, o["q"], np.array([xy[0], xy[1], z]), dq)[0] - o["u"] for o in obs])
        J = np.empty((r0.size, 2))
        for j in range(2):
            xp = xy.copy()
            xp[j] += 1e-3
            J[:, j] = (np.concatenate([project(th, o["q"], np.array([xp[0], xp[1], z]), dq)[0] - o["u"] for o in obs]) - r0) / 1e-3
        xy = xy - np.linalg.lstsq(J, r0, rcond=None)[0]
    return xy


def write_model(M: dict, x: np.ndarray, rep: dict):
    new = copy.deepcopy(M)
    new.update({"f": float(x[0]), "cx": float(x[1]), "cy": float(x[2]), "k1": float(x[3]),
                "rvec_tcp_cam": [float(v) for v in x[4:7]], "t_tcp_cam_mm": [float(v) for v in x[7:10]],
                "reproj_median_px": max(2.0, rep["median_px"]), "n_pairs": rep["n_obs"],
                "fitted": f"tools/calib_wrist_sweep.py {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
                "sweep_report": rep})
    prev = wc.MODEL_F.replace(".json", ".prev.json")
    with open(prev, "w") as f:
        json.dump(M, f, indent=1)
    with open(wc.MODEL_F + ".tmp", "w") as f:
        json.dump(new, f, indent=1)
    os.replace(wc.MODEL_F + ".tmp", wc.MODEL_F)
    print("wrote", wc.MODEL_F, "(previous:", prev + ")")


# ---------------------------------------------------------------- robot sweep

def collect(out_dir: str) -> list:
    import grasp_auto as ga
    import pick_any as pa
    import sort_any as sa
    import sort_loop as sl

    name = os.path.basename(out_dir)
    sa.RUN = out_dir
    sl.RUN = out_dir
    ga.OUT = out_dir
    os.makedirs(out_dir, exist_ok=True)
    sa.EP = {"episode": 0, "run": name, "mode": "camera_calibration"}
    sl.wait_clear("camera calibration")
    sa.go_scan(False)
    b, _, _ = sa.wait_for_bricks()
    c = sa.scan_cal()
    q, bh = sa.approach(b, c)
    if q is None:
        raise SystemExit(f"approach failed: {bh}")
    want, g, obs = bh.get("colour"), 25.0, []
    ref = [list(bh["centre"])]

    def shoot(tag):
        time.sleep(0.35)
        img = ga.fresh("wrist", time.time() + 0.1)
        qm = [float(v) for v in ga.state()["present"][:5]]
        bl = pa.blobs(img, min_area=800, max_area=150000) if img is not None else []
        bb = pa.find(bl, ref[0], 230, want)
        if bb is None or bb.get("border") or bb.get("plate_frac", 0.0) >= 0.3:
            sa.log(f"camera sweep {tag}: brick not found")
            return None
        cv2.imwrite(f"{out_dir}/{tag}.jpg", img)
        obs.append({"tag": tag, "q": qm, "u": [float(bb["centre"][0]), float(bb["centre"][1])],
                    "short": float(bb.get("short", 0.0)), "area": float(bb.get("area", 0.0))})
        return bb

    for dz in (0.0, 30.0, 60.0):
        qz = q if dz == 0.0 else pa.move_rel(q, g, [0.0, 0.0, dz], speed=10.0)
        if qz is None:
            continue
        bz = shoot(f"z{dz:.0f}_c")
        if bz is not None:
            ref[0] = list(bz["centre"])
        s = 15.0 + 0.25 * dz
        for dx, dy in ((s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s), (s, s), (-s, -s), (s, -s)):
            if pa.move_rel(qz, g, [dx, dy, 0.0], speed=10.0) is not None:
                shoot(f"z{dz:.0f}_{dx:+.0f}_{dy:+.0f}")
        if dz == 30.0:
            for j, dv in ((4, -30.0), (4, 30.0), (3, -6.0), (3, 6.0)):
                qq = list(qz)
                qq[j] = qz[j] + dv
                st = ga.cmd({"op": "move", "q": qq + [g], "speed": 10.0, "comp": True})
                if st.get("status") == "ok":
                    shoot(f"{'roll' if j == 4 else 'tilt'}{dv:+.0f}")
        ga.cmd({"op": "move", "q": list(qz) + [g], "speed": 10.0, "comp": True})
    sa.go_scan(False)
    with open(f"{out_dir}/observations.json", "w") as f:
        json.dump(obs, f, indent=1)
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--obs", default=None, help="refit observations saved by an earlier sweep (no robot motion)")
    a = ap.parse_args()
    M = wc.M
    if M is None:
        raise SystemExit("no current wrist camera model to start from")
    if a.obs:
        obs = json.load(open(a.obs))
    else:
        import grasp_auto as ga
        obs = collect(f"{ga.LOG_ROOT}/camera_sweep/sweep_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    if len(obs) < MIN_OBS:
        raise SystemExit(f"only {len(obs)} usable views (need {MIN_OBS})")
    x, dq, rep = evaluate(obs, M)
    rep["f"], rep["t_tcp_cam_mm"] = round(float(x[0]), 1), [round(float(v), 1) for v in x[7:10]]
    rep["camera_moved_deg"] = round(float(np.degrees(np.linalg.norm(cv2.Rodrigues(
        rod(x[4:7]) @ rod(M["rvec_tcp_cam"]).T)[0]))), 2)
    rep["camera_moved_mm"] = round(float(np.linalg.norm(x[7:10] - np.asarray(M["t_tcp_cam_mm"], float))), 1)
    print(json.dumps(rep))
    ok = rep["median_px"] <= MAX_MEDIAN_PX and rep["heldout_median_px"] is not None and \
        rep["heldout_median_px"] <= MAX_HELDOUT_PX
    if a.write and ok:
        write_model(M, x, rep)
    elif a.write:
        print("not written: errors above the limits", MAX_MEDIAN_PX, MAX_HELDOUT_PX)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
