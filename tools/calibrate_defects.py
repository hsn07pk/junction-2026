"""Fit the model misfits of clean plastic (config/defect_model.json) from surveyed bricks with ground truth, and check
the defect test (defects.py) brick by brick.

Data: the survey records of labelled episodes (episode.json: the views with their joints, the measured brick pose;
the frames in survey/) and a labels file:
  {"bricks": {"<brick id>": "GOOD" | "DEFECT", ...},
   "episodes": {"<run>/ep_0001": "<brick id>", ...},
   "source": "who labelled, when"}
Brick ids are any names the operator gives the physical bricks. The labels are ground truth for fitting and checking
only: nothing in the sorting loop reads them.

Fit, on clean bricks only: per misfit (colour; lightness on the top; lightness on the sides; in 8-bit levels, see
defects.fused), the smallest value at which none of a clean survey's windows reaches the a-contrario threshold; the
model takes the largest over the clean surveys, never less than the camera's quantisation floor, times the design
margin (quantisation_floor_levels, calibration_margin in config/inspection.json). Taped bricks are not used for the fit: they check it, so the test stays as sensitive as
clean plastic allows instead of being tuned to the tapes at hand.

Check: hold out one physical brick at a time (all its surveys), fit on the clean surveys of the others, judge the
held-out surveys with the loop's rule (defects.judge, then sort_loop.decide_faces). The model is written only when
every held-out survey is judged right and the final model finds every DEFECT survey.
Usage: python -I -B calibrate_defects.py [LABELS.json] [--write]   (default: logs/sort/labels.json)"""
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
from paths import APP, LOG_ROOT  # noqa: E402
import defects as dfx  # noqa: E402
import kin  # noqa: E402
import survey as sv  # noqa: E402

CHANNELS = {"kappa_chroma": ("zc", sv.FACES), "kappa_light_top": ("zl", ("top",)),
            "kappa_light_side": ("zl", ("side0", "side1", "side2", "side3"))}


def surveys_of(ep_dir: str, sp: dict) -> list:
    """The surveys of one episode that can be replayed: a valid 2x2 measurement and the views' frames on disk."""
    try:
        d = json.load(open(os.path.join(ep_dir, "episode.json")))
    except (OSError, ValueError):
        return []
    out = []
    for key in ("survey", "survey_spot"):
        rec = d.get(key) or {}
        m = rec.get("measure") or {}
        views = (rec.get("inspection") or {}).get("views") or []
        if not m.get("ok") or (rec.get("shape") or [None])[0] != "2x2" or not views:
            continue
        vs = []
        for v in views:
            img = cv2.imread(os.path.join(ep_dir, "survey", f"{v['name']}.jpg"))
            if img is not None:
                vs.append({"img": img, "q": v["q"], "name": v["name"]})
        if vs:
            fp = sp["footprint_mm"]
            out.append({"key": key, "views": vs,
                        "faces3d": sv.brick_faces(m["xy"], m["yaw"], (fp, fp), sp["height_mm"], kin.MAT_Z)})
    return out


def max_z(obs: dict, channel: str, misfit: float) -> float:
    kind, faces = CHANNELS[channel]
    model = {k: 0.0 for k in CHANNELS}
    model[channel] = misfit
    best = 0.0
    for f in faces:
        if f not in obs["faces"]:
            continue
        z = dfx.fused(obs, f, model)[kind]
        if z is not None and np.isfinite(z).any():
            best = max(best, float(np.nanmax(z)))
    return best


def required_misfit(obs: dict, channel: str) -> float:
    """Smallest misfit at which no window of this survey reaches the threshold (bisection: z falls as it grows)."""
    thr = obs["z_star"][0 if CHANNELS[channel][0] == "zc" else 1]
    if max_z(obs, channel, 0.0) < thr:
        return 0.0
    lo, hi = 0.0, 0.1
    while max_z(obs, channel, hi) >= thr:
        hi *= 2.0
        if hi > 1e5:
            return float("inf")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if max_z(obs, channel, mid) >= thr:
            lo = mid
        else:
            hi = mid
    return hi


def fit(surveys: list, cfg: dict) -> dict | None:
    """Model from the clean surveys among `surveys`: per channel, the design margin times the larger of the largest
    misfit a clean survey needed and the camera's quantisation floor (an 8-bit JPEG frame leaves a flat face about a
    level off, whether or not the few clean bricks at hand happened to show it). The hold-out check tests the margin
    on clean bricks the fit did not see."""
    clean = [s for s in surveys if s["truth"] == "GOOD"]
    if not clean:
        return None
    return {ch: float(cfg["calibration_margin"] * max(cfg["quantisation_floor_levels"],
                                                         max(s["req"][ch] for s in clean))) for ch in CHANNELS}


def classify(obs: dict, model: dict, cfg: dict) -> tuple[str, dict]:
    import sort_loop as sl
    res = dfx.judge(obs, model, cfg)
    _, label, _, _ = sl.decide_faces({}, ("2x2", ""), res, sv.FACES)
    return label, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="?", default=os.path.join(LOG_ROOT, "sort", "labels.json"),
                    help="labels file (default: the one sort_any.py --calibrate writes)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--clean-only", action="store_true",
                    help="fit from clean bricks alone (no defective brick surveyed yet): the margins come from the clean "
                         "surveys and only the clean hold-out is checked; defects are checked later on labelled runs")
    a = ap.parse_args()
    t0 = time.time()
    lab = json.load(open(a.labels))
    bricks, episodes = lab.get("bricks") or {}, lab.get("episodes") or {}
    bad = {b: t for b, t in bricks.items() if t not in ("GOOD", "DEFECT")}
    if bad:
        sys.exit(f"labels: truth must be GOOD or DEFECT, got {bad}")
    sp, cfg = sv.spec(), dfx.config()
    search_px = int(round(float(cfg.get("align_search_factor", 6.0)) * float(sv.wc.M["reproj_median_px"])))
    jm = cv2.imread(os.path.join(APP, "tools", "wrist_jaw_mask.png"), 0)
    jaw = (jm > 0) if jm is not None else None
    surveys, skipped = [], []
    for ep, bid in sorted(episodes.items()):
        if bid not in bricks:
            skipped.append((ep, f"brick {bid} has no truth"))
            continue
        for s in surveys_of(os.path.join(LOG_ROOT, "sort", ep), sp):
            obs = dfx.observe(s["views"], s["faces3d"], cfg, search_px, jaw, sp)
            if not obs.get("ok"):
                skipped.append((ep, s["key"], obs.get("why")))
                continue
            surveys.append({"ep": ep, "key": s["key"], "brick": bid, "truth": bricks[bid], "obs": obs,
                            "req": {ch: required_misfit(obs, ch) for ch in CHANNELS}})
    ids = sorted({s["brick"] for s in surveys})
    print(f"surveys {len(surveys)} (GOOD {sum(s['truth'] == 'GOOD' for s in surveys)}, DEFECT "
          f"{sum(s['truth'] == 'DEFECT' for s in surveys)}) of {len(ids)} physical bricks; skipped {skipped}")
    for s in surveys:
        print(f"  {s['ep']} {s['key']} {s['brick']} {s['truth']}: required " +
              " ".join(f"{ch}={v:.4g}" for ch, v in s["req"].items()))
    model = fit(surveys, cfg)
    if model is None or (not any(s["truth"] == "DEFECT" for s in surveys) and not a.clean_only):
        sys.exit("need surveys of clean and of defective bricks (or --clean-only)")
    right, wrong = 0, []
    for b in ids:
        mi = fit([s for s in surveys if s["brick"] != b], cfg)
        for s in (s for s in surveys if s["brick"] == b):
            pred = classify(s["obs"], mi, cfg)[0] if mi else "UNCERTAIN"
            if pred == s["truth"]:
                right += 1
            else:
                wrong.append((s["ep"], s["key"], b, s["truth"], pred))
    found = []
    for s in surveys:
        if s["truth"] == "DEFECT":
            label, res = classify(s["obs"], model, cfg)
            margin = {ch: (round(min(s["req"][ch] / model[ch], 1e6), 2) if model[ch] > 0 else None) for ch in CHANNELS}
            found.append((s["ep"], s["key"], s["brick"], label, [f for f, r in res.items() if r["verdict"] == "DEFECT"],
                          margin))
    missed = [f for f in found if f[3] != "DEFECT"]
    print(f"hold-out by physical brick: {right}/{len(surveys)} surveys right; wrong {wrong}")
    print("DEFECT surveys with the final model (label, faces, required/fitted misfit per channel):")
    for f in found:
        print("  ", f)
    out = dict(model)
    out["provenance"] = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "labels": a.labels,
                         "labels_source": lab.get("source"), "surveys": len(surveys), "physical_bricks": len(ids),
                         "clean_surveys": sum(s["truth"] == "GOOD" for s in surveys), "holdout_right": right,
                         "holdout_total": len(surveys), "holdout_wrong": wrong, "defects_missed": missed,
                         "clean_only": bool(a.clean_only and not any(s["truth"] == "DEFECT" for s in surveys)),
                         "inspection_config": cfg}
    print(json.dumps({k: v for k, v in out.items() if k != "provenance"}, indent=1))
    print(f"calibrated in {time.time() - t0:.0f} s")
    if a.write:
        if wrong or missed:
            sys.exit("not written: a held-out survey is wrong or a defect is missed")
        with open(dfx.MODEL_F + ".tmp", "w") as f:
            json.dump(out, f, indent=1, default=float)
        os.replace(dfx.MODEL_F + ".tmp", dfx.MODEL_F)
        print("wrote", dfx.MODEL_F)


if __name__ == "__main__":
    main()
