"""Fit the face model (config/face_model.json) from surveyed bricks with known truth.

Data: episodes whose episode.json carries "truth" (GOOD or DEFECT, set by a collection run with --collect) and whose
survey records hold, per face, the view image and the aligned face polygon. A GOOD brick has all faces clean; a DEFECT
brick has at least one defective face (which one is not needed: the brick's score is its worst face).

Search: erosion, the two per-pixel cutoffs (in units of each face's own robust spread) and the noise floors (low
quantiles of the clean faces' spreads). For every setting each face gets its largest anomalous region (mm^2); the
setting kept is the one with the widest gap, in log scale, between the worst clean face and the weakest defect brick.
The area limit is the geometric middle of that gap. Leave-one-brick-out repeats the whole fit without each brick and
classifies it. Usage: python -I -B fit_faces.py RUN_DIR [RUN_DIR ...] [--write]"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP  # noqa: E402
import faces as fc  # noqa: E402
import survey as sv  # noqa: E402

MODEL_F = os.path.join(APP, "config", "face_model.json")
ERODE = (0.06, 0.10, 0.14)
K = tuple(range(3, 13))
NOISE_Q = (0.10, 0.25, 0.50)


def load(run_dirs):
    sp = sv.spec()
    area = {"top": sp["footprint_mm"] ** 2, **{f"side{i}": sp["footprint_mm"] * sp["height_mm"] for i in range(4)}}
    bricks = []
    for rd in run_dirs:
        for epj in sorted(glob.glob(os.path.join(rd, "ep_*", "episode.json"))):
            d = json.load(open(epj))
            truth = d.get("truth")
            if truth not in ("GOOD", "DEFECT"):
                continue
            ep = os.path.dirname(epj)
            fs = {}
            for rec in (d.get("survey") or {}, d.get("survey_spot") or {}):
                for f, r in (rec.get("faces") or {}).items():
                    img_p = os.path.join(ep, "survey", f"{r.get('view')}.jpg")
                    if r.get("uv") and os.path.exists(img_p):
                        fs[f] = {"img": img_p, "uv": r["uv"], "area": area[f]}
            if fs:
                bricks.append({"ep": os.path.relpath(ep, os.path.dirname(rd)), "truth": truth, "faces": fs,
                               "colour": d.get("target_colour")})
    return bricks


def precompute(bricks):
    """Per brick, per face, per erosion: the deviation record (independent of the cutoffs)."""
    cache = {}
    for i, b in enumerate(bricks):
        for f, r in b["faces"].items():
            img = cv2.imread(r["img"])
            for e in ERODE:
                m = fc.face_mask(img.shape, r["uv"], e)
                d = fc.deviations(img, m)
                cache[(i, f, e)] = (img, d)
    return cache


def scores(bricks, cache, idx, e, kc, kl, nc, nl):
    """Worst-face region (mm^2) per brick in idx."""
    out = []
    for i in idx:
        worst = 0.0
        for f, r in bricks[i]["faces"].items():
            img, d = cache[(i, f, e)]
            if d is None:
                continue
            reg, _, _ = fc.region_mm2(img, r["uv"], r["area"], d, kc, kl, nc, nl)
            worst = max(worst, reg)
        out.append(worst)
    return out


def fit(bricks, cache, idx):
    good = [i for i in idx if bricks[i]["truth"] == "GOOD"]
    bad = [i for i in idx if bricks[i]["truth"] == "DEFECT"]
    if not good or not bad:
        return None
    best = None
    for e in ERODE:
        sC = [cache[(i, f, e)][1]["sC"] for i in good for f in bricks[i]["faces"] if cache[(i, f, e)][1]]
        sL = [cache[(i, f, e)][1]["sL"] for i in good for f in bricks[i]["faces"] if cache[(i, f, e)][1]]
        for qn in NOISE_Q:
            nc, nl = float(np.quantile(sC, qn)), float(np.quantile(sL, qn))
            for kc, kl in itertools.product(K, K):
                g = scores(bricks, cache, good, e, kc, kl, nc, nl)
                b = scores(bricks, cache, bad, e, kc, kl, nc, nl)
                lo, hi = max(g), min(b)
                gap = math.log((hi + 1e-3) / (lo + 1e-3))
                if best is None or gap > best["gap"]:
                    best = {"gap": gap, "erode_frac": e, "k_chroma": kc, "k_light": kl, "noise_chroma": nc,
                            "noise_light": nl, "noise_q": qn, "clean_max_mm2": lo, "defect_min_mm2": hi,
                            "max_region_mm2": math.sqrt((lo + 1e-3) * (hi + 1e-3))}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    bricks = load(a.runs)
    n_g = sum(b["truth"] == "GOOD" for b in bricks)
    n_d = len(bricks) - n_g
    print(f"bricks {len(bricks)} (GOOD {n_g}, DEFECT {n_d}), faces {sum(len(b['faces']) for b in bricks)}")
    if not n_g or not n_d:
        sys.exit("need surveyed GOOD and DEFECT bricks")
    cache = precompute(bricks)
    allidx = list(range(len(bricks)))
    m = fit(bricks, cache, allidx)
    print("fit:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    right, wrong = 0, []
    for i in allidx:
        mi = fit(bricks, cache, [j for j in allidx if j != i])
        if mi is None:
            continue
        s = scores(bricks, cache, [i], mi["erode_frac"], mi["k_chroma"], mi["k_light"], mi["noise_chroma"],
                   mi["noise_light"])[0]
        pred = "DEFECT" if s > mi["max_region_mm2"] else "GOOD"
        if pred == bricks[i]["truth"]:
            right += 1
        else:
            wrong.append((bricks[i]["ep"], bricks[i]["truth"], round(s, 1), round(mi["max_region_mm2"], 1)))
    print(f"leave-one-brick-out: {right}/{len(allidx)} correct; wrong {wrong}")
    vf = []
    for i in allidx:
        if bricks[i]["truth"] != "GOOD":
            continue
        for f in bricks[i]["faces"]:
            d = cache[(i, f, m["erode_frac"])][1]
            if d:
                vf.append(d["n_valid"] / max(1, d["n_mask"]))
    model = {k: m[k] for k in ("erode_frac", "k_chroma", "k_light", "noise_chroma", "noise_light", "max_region_mm2")}
    model["min_valid_frac"] = round(0.8 * float(np.quantile(vf, 0.05)), 3) if vf else 0.5
    model["provenance"] = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": a.runs,
                           "bricks_good": n_g, "bricks_defect": n_d,
                           "colours": sorted({str(b["colour"]) for b in bricks}),
                           "clean_max_mm2": round(m["clean_max_mm2"], 2), "defect_min_mm2": round(m["defect_min_mm2"], 2),
                           "log_gap": round(m["gap"], 3), "noise_quantile": m["noise_q"],
                           "lobo_correct": right, "lobo_total": len(allidx), "lobo_wrong": wrong}
    print(json.dumps(model, indent=1))
    print(f"fitted in {time.time() - t0:.0f} s")
    if a.write:
        if m["gap"] <= 0 or wrong:
            sys.exit("not written: the classes overlap or leave-one-brick-out has errors")
        with open(MODEL_F + ".tmp", "w") as f:
            json.dump(model, f, indent=1)
        os.replace(MODEL_F + ".tmp", MODEL_F)
        print("wrote", MODEL_F)


if __name__ == "__main__":
    main()
