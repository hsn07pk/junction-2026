"""Refine the wrist-camera model (eyeinhand/wrist_cam_model.json) from the sort loop's own surveys.

Every survey photographs one brick of known size from several arm poses. Each view is registered (defects.register_view:
the projected brick aligned to the image's edges and silhouette), which gives where the brick's eight corners really
are in that image. A bundle adjustment then fits the camera model - focal length, principal point, radial term, the
camera's pose on the gripper and the arm's joint offsets - together with each surveyed brick's pose on the mat, so
that the model projects the corners where the images show them. Priors keep every parameter near the current model,
so poorly observed ones stay put. The registration is redone with the refined model and the fit repeated.

The new model is written only when it predicts held-out surveys better (k-fold by survey: fit without them, then
place their bricks with the fitted model alone and compare the corners' reprojection error with the current model's).
The previous model is kept as wrist_cam_model.prev.json. After a new model, rerun plan_survey.py (the survey's view
library is computed with the camera model).
Usage: python -I -B calib_wristcam.py [--runs RUN ...] [--write]"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP, LOG_ROOT  # noqa: E402
import defects as dfx  # noqa: E402
import kin  # noqa: E402
import survey as sv  # noqa: E402

wc = sv.wc
NAMES = ["f", "cx", "cy", "k1", "r0", "r1", "r2", "t0", "t1", "t2", "dq0", "dq1", "dq2", "dq3", "dq4"]
PRIOR_SD = np.array([25.0, 25.0, 25.0, 0.08, 0.05, 0.05, 0.05, 6.0, 6.0, 6.0, 2.0, 2.0, 2.0, 2.0, 2.0])
POSE_SD = np.array([4.0, 4.0, math.radians(4.0)])  # the survey's own measurement of the brick
FACE_CORNERS = {"top": (4, 5, 6, 7), "side0": (0, 1, 5, 4), "side1": (1, 2, 6, 5), "side2": (2, 3, 7, 6),
                "side3": (3, 0, 4, 7)}  # corner ids as survey.brick_faces orders them (bottom 0-3, top 4-7)


def params_of(M: dict) -> np.ndarray:
    return np.array([M["f"], M["cx"], M["cy"], M["k1"], *M["rvec_tcp_cam"], *M["t_tcp_cam_mm"],
                     *M.get("joint_offsets_norm", [0.0] * 5)], float)


def model_of(th: np.ndarray, base: dict) -> dict:
    M = dict(base)
    M.update(f=float(th[0]), cx=float(th[1]), cy=float(th[2]), k1=float(th[3]), rvec_tcp_cam=[float(v) for v in th[4:7]],
             t_tcp_cam_mm=[float(v) for v in th[7:10]], joint_offsets_norm=[float(v) for v in th[10:15]])
    return M


@contextlib.contextmanager
def installed(M: dict):
    """Use model M for everything that projects through wristcam (registration included) inside the block."""
    old = (wc.M, wc._RC, wc._DQ)
    wc.M = M
    wc._RC = cv2.Rodrigues(np.asarray(M["rvec_tcp_cam"], float).reshape(3, 1))[0]
    wc._DQ = np.asarray(M.get("joint_offsets_norm", [0] * 5), float)
    try:
        yield
    finally:
        wc.M, wc._RC, wc._DQ = old


def corners(pose, sp) -> np.ndarray:
    fp = sp["footprint_mm"]
    f = sv.brick_faces(pose[:2], pose[2], (fp, fp), sp["height_mm"], kin.MAT_Z)
    return np.array([f["side0"]["pts"][0], f["side1"]["pts"][0], f["side2"]["pts"][0], f["side3"]["pts"][0],
                     f["top"]["pts"][0], f["top"]["pts"][1], f["top"]["pts"][2], f["top"]["pts"][3]], float)


def project(th: np.ndarray, q, P: np.ndarray) -> np.ndarray:
    p, Rm = kin.fk(list(np.asarray(list(q)[:5], float) + th[10:15]))
    Rm = np.asarray(Rm, float)
    Rbc = Rm @ cv2.Rodrigues(th[4:7].reshape(3, 1))[0]
    cb = np.asarray(p, float) + Rm @ th[7:10]
    X = (P - cb) @ Rbc
    x, y = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    d = 1.0 + th[3] * (x * x + y * y)
    return np.stack([th[0] * x * d + th[1], th[0] * y * d + th[2]], 1)


# ---------------------------------------------------------------- data

def surveys_in(runs: list | None) -> list:
    """Every replayable survey under logs/sort (or only the given runs): views with joints and frames, measured pose."""
    root = os.path.join(LOG_ROOT, "sort")
    out = []
    for run in sorted(runs or (os.listdir(root) if os.path.isdir(root) else [])):
        rdir = os.path.join(root, run)
        if not os.path.isdir(rdir):
            continue
        for ep in sorted(os.listdir(rdir)):
            try:
                d = json.load(open(os.path.join(rdir, ep, "episode.json")))
            except (OSError, ValueError):
                continue
            for key in ("survey", "survey_spot"):
                rec = d.get(key) or {}
                m = rec.get("measure") or {}
                views = (rec.get("inspection") or {}).get("views") or []
                if not m.get("ok") or (rec.get("shape") or [None])[0] != "2x2" or len(views) < 2:
                    continue
                vs = []
                for v in views:
                    img = cv2.imread(os.path.join(rdir, ep, "survey", f"{v['name']}.jpg"))
                    if img is not None:
                        vs.append({"img": img, "q": v["q"], "name": v["name"]})
                if len(vs) >= 2:
                    out.append({"id": f"{run}/{ep}/{key}", "views": vs, "pose0": [m["xy"][0], m["xy"][1], m["yaw"]]})
    return out


def observe(survey: dict, M: dict, sp: dict, cfg: dict, jaw) -> list:
    """Corner observations of one survey under model M: (q, corner id, observed pixel) from each well-registered view."""
    obs = []
    with installed(M):
        fp = sp["footprint_mm"]
        f3 = sv.brick_faces(survey["pose0"][:2], survey["pose0"][2], (fp, fp), sp["height_mm"], kin.MAT_Z)
        search = int(round(6 * float(M["reproj_median_px"])))
        for v in survey["views"]:
            reg = dfx.register_view(v["img"], v["q"], f3, search, jaw)
            if not reg.get("ok") or reg["iou"] < cfg["min_iou"] or reg["at_limit"] or \
                    reg.get("edge_inliers", 1.0) < cfg["min_edge_inliers"]:
                continue
            P = corners(survey["pose0"], sp)
            uv0 = dfx.apply_T(sv.project_batch(v["q"], P), reg["c"], reg["T"])
            acc = {}
            for f, ids in FACE_CORNERS.items():
                if reg["proj"][f]["cos"] <= 0:
                    continue
                sh = np.asarray(reg["face_shift"].get(f, (0.0, 0.0)), float)
                for j in ids:
                    acc.setdefault(j, []).append(uv0[j] + sh)
            for j, pts in acc.items():
                obs.append((list(v["q"]), j, np.mean(pts, axis=0)))
    return obs


# ---------------------------------------------------------------- bundle adjustment

def _residuals(x: np.ndarray, th0: np.ndarray, data: list, sp: dict, fixed_cam: bool = False) -> np.ndarray:
    th = th0 if fixed_cam else x[:15]
    off = 0 if fixed_cam else 15
    r = []
    for i, s in enumerate(data):
        pose = x[off + 3 * i: off + 3 * i + 3]
        P = corners(pose, sp)
        for q, j, uv in s["obs"]:
            r.append(project(th, q, P[j:j + 1])[0] - uv)
        r.append((pose - np.asarray(s["pose0"])) / POSE_SD)
    if not fixed_cam:
        r.append((th - th0) / PRIOR_SD)
    return np.concatenate([np.ravel(a) for a in r])


def solve(th0: np.ndarray, data: list, sp: dict, fixed_cam: bool = False, iters: int = 30) -> tuple[np.ndarray, list]:
    """Levenberg-Marquardt with a Huber weight on pixel residuals (a bad registration does not drag the model).
    Returns the camera parameters and the brick poses."""
    x = np.concatenate(([] if fixed_cam else [th0]) + [np.asarray(s["pose0"], float) for s in data])
    lam = 1e-2
    steps = np.concatenate(([] if fixed_cam else [np.abs(th0) * 1e-6 + np.array([1e-3] * 3 + [1e-6] + [1e-6] * 3 +
                                                                                   [1e-3] * 3 + [1e-3] * 5)]) +
                           [np.array([1e-3, 1e-3, 1e-5])] * len(data))

    def weighted(xx):
        r = _residuals(xx, th0, data, sp, fixed_cam)
        w = np.ones_like(r)
        a = np.abs(r)
        w[a > 3.0] = np.sqrt(3.0 / a[a > 3.0])  # Huber at 3 px (priors are in units of their sd: same cut)
        return r * w

    r = weighted(x)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.empty((r.size, x.size))
        for k in range(x.size):
            xk = x.copy()
            xk[k] += steps[k]
            J[:, k] = (weighted(xk) - r) / steps[k]
        A, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(8):
            dx = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            xn = x + dx
            rn = weighted(xn)
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost, lam, improved = xn, rn, cn, max(lam / 3.0, 1e-7), True
                break
            lam *= 4.0
        if not improved or float(np.max(np.abs(dx) / steps)) < 1e-2:
            break
    th = th0 if fixed_cam else x[:15]
    off = 0 if fixed_cam else 15
    return th, [x[off + 3 * i: off + 3 * i + 3] for i in range(len(data))]


def pixel_errors(th: np.ndarray, data: list, sp: dict) -> np.ndarray:
    """Corner reprojection errors (px) of these surveys under camera th, each brick placed by the model alone."""
    _, poses = solve(th, data, sp, fixed_cam=True)
    e = []
    for s, pose in zip(data, poses):
        P = corners(pose, sp)
        for q, j, uv in s["obs"]:
            e.append(float(np.linalg.norm(project(th, q, P[j:j + 1])[0] - uv)))
    return np.array(e)


def fit_model(M0: dict, surveys: list, sp: dict, cfg: dict, jaw, rounds: int = 2) -> tuple[np.ndarray, list]:
    """Register with the current guess, adjust, repeat (the registration's search is centred by the model)."""
    th = params_of(M0)
    data = []
    for _ in range(rounds):
        M = model_of(th, M0)
        data = [dict(s, obs=observe(s, M, sp, cfg, jaw)) for s in surveys]
        data = [d for d in data if len(d["obs"]) >= 8]
        if not data:
            break
        th, _ = solve(params_of(M0), data, sp)
    return th, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="runs under logs/sort (default: all)")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    sp, cfg = sv.spec(), dfx.config()
    jm = cv2.imread(os.path.join(APP, "tools", "wrist_jaw_mask.png"), 0)
    jaw = (jm > 0) if jm is not None else None
    M0 = dict(wc.M)
    surveys = surveys_in(a.runs)
    print(f"{len(surveys)} replayable surveys")
    if len(surveys) < 2 * a.folds:
        sys.exit(f"need at least {2 * a.folds} surveys")
    th, data = fit_model(M0, surveys, sp, cfg, jaw)
    ids = [d["id"] for d in data]
    order = np.random.default_rng(0).permutation(len(data))
    err_new, err_old = [], []
    for k in range(a.folds):
        test = [data[i] for i in order[k::a.folds]]
        train = [data[i] for i in order if i not in order[k::a.folds]]
        th_k, _ = solve(params_of(M0), train, sp)
        err_new.append(pixel_errors(th_k, test, sp))
        err_old.append(pixel_errors(params_of(M0), test, sp))
    en, eo = np.concatenate(err_new), np.concatenate(err_old)
    rep = {"surveys": len(data), "observations": int(sum(len(d["obs"]) for d in data)),
           "heldout_median_px": {"current": round(float(np.median(eo)), 2), "refined": round(float(np.median(en)), 2)},
           "heldout_p90_px": {"current": round(float(np.percentile(eo, 90)), 2),
                              "refined": round(float(np.percentile(en, 90)), 2)},
           "change": {n: round(float(v), 4) for n, v in zip(NAMES, th - params_of(M0))}}
    print(json.dumps(rep, indent=1))
    print(f"calibrated in {time.time() - t0:.0f} s")
    if a.write:
        if not (np.median(en) < 0.9 * np.median(eo) and np.percentile(en, 90) <= np.percentile(eo, 90)):
            sys.exit("not written: the refined model does not predict held-out surveys better")
        M = model_of(th, M0)
        M["reproj_median_px"] = float(np.median(en))
        M["self_calibration"] = dict(rep, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runs=a.runs,
                                     survey_ids=ids)
        prev = wc.MODEL_F.replace(".json", ".prev.json")
        with open(prev, "w") as f:
            json.dump(M0, f, indent=1)
        with open(wc.MODEL_F + ".tmp", "w") as f:
            json.dump(M, f, indent=1)
        os.replace(wc.MODEL_F + ".tmp", wc.MODEL_F)
        print("wrote", wc.MODEL_F, "(previous:", prev + "); rerun plan_survey.py")


if __name__ == "__main__":
    main()
