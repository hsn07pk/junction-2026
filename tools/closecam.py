"""Close-camera model (config/close_cam_model.json): where the fixed close camera is and how it projects, fitted from
the arm itself. The red jaws are moved through a grid of poses; at each, the close frame minus the frame with the arm
out of view gives the jaws' pixel, paired with the TCP's position from the arm's forward kinematics. A pinhole camera
with one radial term and its pose are fitted to the pairs (Levenberg-Marquardt from a direct linear transform start).
The underside inspection (underside.py) uses the model to show a held brick's bottom to the camera.
Usage on the cell (watchdog, recorders and motion server running; nothing red on the mat):
  python -I -B closecam.py --collect      moves the arm through the grid, writes run/close_pairs.jsonl
  python -I -B closecam.py --fit [--write]  fits and checks the model (k-fold), writes it when it holds"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP, RUN_ROOT  # noqa: E402
import kin  # noqa: E402

MODEL_F = os.path.join(APP, "config", "close_cam_model.json")
PAIRS_F = os.path.join(RUN_ROOT, "close_pairs.jsonl")


def load() -> dict | None:
    try:
        return json.load(open(MODEL_F))
    except (OSError, ValueError):
        return None


def project(M: dict, P: np.ndarray) -> np.ndarray:
    """Pixels of base-frame points P (N x 3, mm)."""
    R = cv2.Rodrigues(np.asarray(M["rvec"], float).reshape(3, 1))[0]
    X = (np.asarray(P, float).reshape(-1, 3) - np.asarray(M["centre_mm"], float)) @ R.T
    x, y = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    d = 1.0 + M["k1"] * (x * x + y * y)
    return np.stack([M["f"] * x * d + M["cx"], M["f"] * y * d + M["cy"]], 1)


def in_front(M: dict, P: np.ndarray) -> np.ndarray:
    R = cv2.Rodrigues(np.asarray(M["rvec"], float).reshape(3, 1))[0]
    return ((np.asarray(P, float).reshape(-1, 3) - np.asarray(M["centre_mm"], float)) @ R.T)[:, 2] > 0


def jaw_pixel(img: np.ndarray, empty: np.ndarray) -> list | None:
    """Pixel of the jaws: the largest red region present now but not in the frame with the arm out of view."""
    import jaw_check as jc
    diff = cv2.absdiff(cv2.GaussianBlur(img, (5, 5), 0), cv2.GaussianBlur(empty, (5, 5), 0)).max(axis=2) > 30
    red = (jc.redmask(img) > 0) | _red_full(img)
    m = (diff & red).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    return [float(cen[i][0]), float(cen[i][1])] if st[i, 4] >= 150 else None


def _red_full(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((H >= 165) | (H <= 4)) & (S >= 85) & (V >= 55)


def _params(M):
    return np.array([M["f"], M["cx"], M["cy"], M["k1"], *M["rvec"], *M["centre_mm"]], float)


def _model(p, size):
    return {"f": float(p[0]), "cx": float(p[1]), "cy": float(p[2]), "k1": float(p[3]), "rvec": [float(v) for v in p[4:7]],
            "centre_mm": [float(v) for v in p[7:10]], "image_size": list(size)}


def fit(P: np.ndarray, uv: np.ndarray, size) -> dict:
    """Pinhole + radial camera from 3D-2D pairs: DLT start (non-coplanar points), then Levenberg-Marquardt with a Huber
    weight (a jaw pixel off the TCP by a few millimetres does not drag the model)."""
    A = []
    for (X, Y, Z), (u, v) in zip(P, uv):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])
    _, _, Vt = np.linalg.svd(np.asarray(A, float))
    Pm = Vt[-1].reshape(3, 4)
    K, R, C = cv2.decomposeProjectionMatrix(Pm)[:3]
    K = K / K[2, 2]
    C = (C[:3] / C[3]).ravel()
    if np.linalg.det(R) < 0:
        R = -R
    p = np.array([0.5 * (abs(K[0, 0]) + abs(K[1, 1])), K[0, 2], K[1, 2], 0.0,
                  *cv2.Rodrigues(R)[0].ravel(), *C], float)

    def res(pp):
        r = (project(_model(pp, size), P) - uv).ravel()
        a = np.abs(r)
        w = np.where(a > 3.0, np.sqrt(3.0 / np.maximum(a, 1e-9)), 1.0)
        return r * w

    lam, r = 1e-3, res(p)
    for _ in range(100):
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            dp = np.zeros(p.size)
            dp[k] = 1e-6 * max(1.0, abs(p[k]))
            J[:, k] = (res(p + dp) - r) / dp[k]
        A2, g = J.T @ J, J.T @ r
        done = True
        for _ in range(8):
            step = -np.linalg.solve(A2 + lam * np.diag(np.diag(A2) + 1e-12), g)
            rn = res(p + step)
            if rn @ rn < r @ r:
                p, r, lam, done = p + step, rn, max(lam / 3, 1e-9), False
                break
            lam *= 4
        if done:
            break
    return _model(p, size)


def grid_poses(q_ref):
    """Top-down TCP positions over the work area at three heights (where the jaws are likely in the close view)."""
    out = []
    for r in (170.0, 220.0, 270.0, 320.0):
        for ang in (-40.0, -20.0, 0.0, 20.0, 40.0):
            for dz in (40.0, 90.0, 140.0):
                x = kin.PAN_XY_MM[0] + r * math.cos(math.radians(ang))
                y = kin.PAN_XY_MM[1] + r * math.sin(math.radians(ang))
                q = kin.ik(x, y, kin.MAT_Z + dz, kin.psi_of(q_ref), q_ref)
                if q is not None:
                    out.append(q)
    return out


def collect():
    import grasp_auto as ga
    import sort_any as sa
    q_ref = list(kin.cell.get("spot_hover_q"))
    sa.go_scan(False)
    empty = ga.fresh("close", time.time() + 0.3)
    n = 0
    with open(PAIRS_F, "a") as f:
        for q in grid_poses(q_ref):
            ga.move(list(q) + [0.0], speed=30.0, comp=True)
            img = ga.fresh("close", time.time() + 0.3)
            uv = jaw_pixel(img, empty) if img is not None and empty is not None else None
            if uv is None:
                continue
            p, _ = kin.fk(ga.state()["present"][:5])
            f.write(json.dumps({"x_mm": float(p[0]), "y_mm": float(p[1]), "z_mm": float(p[2]), "u": uv[0], "v": uv[1],
                                "kind": "jaws"}) + "\n")
            n += 1
    sa.go_scan(False)
    print(f"{n} pairs added to {PAIRS_F}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect()
    if a.fit:
        rows = [json.loads(ln) for ln in open(PAIRS_F) if '"z_mm"' in ln]
        if len(rows) < 12:
            sys.exit(f"{len(rows)} 3-D pairs: collect at least 12 (closecam.py --collect)")
        P = np.array([[r["x_mm"], r["y_mm"], r["z_mm"]] for r in rows])
        uv = np.array([[r["u"], r["v"]] for r in rows])
        import grasp_auto as ga
        img = ga.fresh("close", 0)
        size = (img.shape[1], img.shape[0]) if img is not None else (640, 480)
        idx = np.random.default_rng(0).permutation(len(rows))
        held = []
        for k in range(4):
            te = idx[k::4]
            tr = np.setdiff1d(idx, te)
            Mk = fit(P[tr], uv[tr], size)
            held.append(np.linalg.norm(project(Mk, P[te]) - uv[te], axis=1))
        held = np.concatenate(held)
        M = fit(P, uv, size)
        M.update(pairs=len(rows), heldout_median_px=round(float(np.median(held)), 2),
                 heldout_p90_px=round(float(np.percentile(held, 90)), 2),
                 created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        print(json.dumps(M, indent=1))
        if a.write:
            if np.median(held) > 0.03 * size[0]:
                sys.exit("not written: held-out pairs are reprojected too far (median above 3 % of the image width)")
            with open(MODEL_F + ".tmp", "w") as f:
                json.dump(M, f, indent=1)
            os.replace(MODEL_F + ".tmp", MODEL_F)
            print("wrote", MODEL_F)


if __name__ == "__main__":
    main()
