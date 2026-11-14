#!/usr/bin/env python3
"""Height above the mat from a rs_grab.py depth map.

Pinhole intrinsics: factory values from --intr JSON when available, else a nominal D415 depth HFOV
(default 65 deg, principal point at image centre). The dominant plane (the mat) is found with RANSAC and
refined by SVD; height = signed distance to that plane in mm, positive toward the camera.

Prints: camera-to-mat distance, tilt, object components above --min-h, and stats for each --roi.
Writes <out>/<cam>_height.npy and <cam>_height_viz.png (height colour map | IR, 50 px grid, labelled).
"""
import argparse
import json
import os

import cv2
import numpy as np


def intrinsics(w, h, hfov, intr_json):
    if intr_json and os.path.exists(intr_json):
        d = json.load(open(intr_json))
        k = d.get(f"{w}x{h}")
        if k:
            return k["fx"], k["fy"], k["ppx"], k["ppy"], "factory"
    fx = 0.5 * w / np.tan(np.radians(hfov) / 2)
    return fx, fx, (w - 1) / 2.0, (h - 1) / 2.0, f"nominal_hfov{hfov}"


def cloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return np.dstack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth])


def fit_plane(P, iters=600, thr=2.5, seed=0):
    rng = np.random.default_rng(seed)
    best, bestc = None, -1
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        nv = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(nv)
        if nn < 1e-6:
            continue
        nv = nv / nn
        d = -float(nv @ s[0])
        c = int((np.abs(P @ nv + d) < thr).sum())
        if c > bestc:
            best, bestc = (nv, d), c
    nv, d = best
    for _ in range(2):
        inl = np.abs(P @ nv + d) < thr
        Q = P[inl]
        c0 = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c0, full_matrices=False)
        nv = vt[2]
        d = -float(nv @ c0)
    if d < 0:
        nv, d = -nv, -d
    inl = np.abs(P @ nv + d) < thr
    return nv, d, float(inl.mean()), float(np.std((P[inl] @ nv) + d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--hfov", type=float, default=65.0)
    ap.add_argument("--intr", default="")
    ap.add_argument("--plane-roi", default="", help="x0,y0,x1,y1 region that is mostly mat (default: whole image)")
    ap.add_argument("--min-h", type=float, default=5.0)
    ap.add_argument("--roi", action="append", default=[], help="name:x0,y0,x1,y1 (depth-image pixels)")
    ap.add_argument("--vmax", type=float, default=45.0)
    a = ap.parse_args()
    depth = np.load(os.path.join(a.dir, f"{a.cam}_depth.npy"))
    h, w = depth.shape
    fx, fy, cx, cy, src = intrinsics(w, h, a.hfov, a.intr)
    P = cloud(depth, fx, fy, cx, cy)
    valid = np.isfinite(depth)
    sel = valid.copy()
    if a.plane_roi:
        x0, y0, x1, y1 = map(int, a.plane_roi.split(","))
        m = np.zeros_like(sel)
        m[y0:y1, x0:x1] = True
        sel &= m
    pts = P[sel]
    if len(pts) > 40000:
        pts = pts[np.random.default_rng(1).choice(len(pts), 40000, replace=False)]
    nv, d, inl, rms = fit_plane(pts)
    H = np.full((h, w), np.nan, np.float32)
    H[valid] = P[valid] @ nv + d
    np.save(os.path.join(a.dir, f"{a.cam}_height.npy"), H)
    tilt = float(np.degrees(np.arccos(abs(nv[2]))))
    out = {"cam": a.cam, "size": [w, h], "intr": [round(fx, 1), round(fy, 1), round(cx, 1), round(cy, 1), src],
           "cam_to_mat_mm": round(d, 1), "tilt_deg": round(tilt, 1), "plane_inlier_frac": round(inl, 3),
           "plane_rms_mm": round(rms, 2), "valid_frac": round(float(valid.mean()), 3), "normal": [round(float(x), 4) for x in nv]}
    print(json.dumps(out))
    # components above the mat
    obj = (np.nan_to_num(H, nan=-99) > a.min_h).astype(np.uint8)
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(obj)
    comps = []
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        if area < 150:
            continue
        hv = H[lab == i]
        comps.append((int(area), int(x), int(y), int(ww), int(hh), float(np.percentile(hv, 50)),
                      float(np.percentile(hv, 95)), float(hv.max())))
    comps.sort(reverse=True)
    for c in comps[:15]:
        print("comp area=%d bbox=(%d,%d,%d,%d) h_p50=%.1f h_p95=%.1f h_max=%.1f" % c)
    for r in a.roi:
        name, box = r.split(":")
        x0, y0, x1, y1 = map(int, box.split(","))
        hv = H[y0:y1, x0:x1]
        pv = P[y0:y1, x0:x1][np.isfinite(hv)]
        hv = hv[np.isfinite(hv)]
        if hv.size == 0:
            print(f"roi {name}: no depth")
            continue
        q = np.percentile(hv, [5, 25, 50, 75, 95])
        c3 = pv.mean(0)
        print(f"roi {name}: n={hv.size} fill={hv.size/max(1,(y1-y0)*(x1-x0)):.2f} h p5={q[0]:.1f} p25={q[1]:.1f} "
              f"p50={q[2]:.1f} p75={q[3]:.1f} p95={q[4]:.1f} min={hv.min():.1f} max={hv.max():.1f} "
              f"xyz_cam=({c3[0]:.0f},{c3[1]:.0f},{c3[2]:.0f})")
    # visualisation
    vis = np.clip(np.nan_to_num(H, nan=-1e3) / a.vmax, 0, 1)
    col = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col[~valid] = 0
    col[np.nan_to_num(H, nan=-1e3) < -3] = (60, 60, 60)
    irp = os.path.join(a.dir, f"{a.cam}_ir.png")
    ir = cv2.imread(irp) if os.path.exists(irp) else np.zeros_like(col)
    for im in (col, ir):
        for x in range(0, w, 50):
            cv2.line(im, (x, 0), (x, h - 1), (255, 255, 255), 1)
            cv2.putText(im, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        for y in range(0, h, 50):
            cv2.line(im, (0, y), (w - 1, y), (255, 255, 255), 1)
            cv2.putText(im, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(col, f"height 0..{a.vmax:.0f} mm (jet), grey<-3", (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(a.dir, f"{a.cam}_height_viz.png"), np.vstack([col, ir]))


if __name__ == "__main__":
    main()
