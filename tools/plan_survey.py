"""Choose the inspection spot and the survey poses from the arm and camera models; write config/survey_plan.json.

For every candidate spot on the mat (polar grid around the pan axis, clear of the plate keep-out), a brick standing
there with the jaws' yaw is surveyed in simulation: plan_views() finds the best reachable pose per face. The spot
score is the worst face's score, so the chosen spot is the one where every face is seen well. The chosen spot is then
re-planned on a finer grid. Inputs are recorded in the plan (file hashes), so a changed model or calibration shows.
Usage: python -I -B plan_survey.py [--dry]"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP  # noqa: E402
import kin  # noqa: E402
import survey as sv  # noqa: E402
import sort_any as sa  # noqa: E402  (plate positions and keep-out, the hover height and the put-down depth)
import pick_any as pa  # noqa: E402

COARSE = {"offsets_mm": list(range(-90, 91, 30)), "lifts_mm": [0, 30], "rolls": list(range(-90, 91, 20)),
          "min_cos": 0.35, "max_jaw": 0.03}
FINE = {"offsets_mm": list(range(-100, 101, 10)), "lifts_mm": [0, 15, 30, 45], "rolls": [r / 2 for r in range(-190, 191, 15)],
        "min_cos": 0.35, "max_jaw": 0.03}


def sha(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    except OSError:
        return None


def spot_pose(x, y, q_ref, psi):
    return kin.ik(x, y, kin.MAT_Z + pa.HOVER_DZ, psi, q_ref)


def evaluate(xy, q_h, spec, jaw, search):
    psi = kin.psi_of(q_h)
    faces = sv.brick_faces(xy, psi, (spec["footprint_mm"], spec["footprint_mm"]), spec["height_mm"], kin.MAT_Z)
    best = sv.plan_views(q_h, faces, jaw, search)
    worst = min((best.get(f, {}).get("score", 0.0) for f in sv.FACES), default=0.0)
    return worst, best, psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print, do not write the plan file")
    a = ap.parse_args()
    t0 = time.time()
    spec = sv.spec()
    jm = cv2.imread(os.path.join(APP, "tools", "wrist_jaw_mask.png"), 0)
    jaw = (jm > 0) if jm is not None else None
    q_ref = list(sa.SPOT_HOVER)
    psi_ref = kin.psi_of(q_ref)
    spots, keys = [], set()
    for r in spec["spot_search"]["radius_mm"]:
        for ang in spec["spot_search"]["angle_deg"]:
            x = kin.PAN_XY_MM[0] + r * math.cos(math.radians(ang))
            y = kin.PAN_XY_MM[1] + r * math.sin(math.radians(ang))
            if min(math.hypot(x - px, y - py) for px, py in sa.PLATES_XY.values()) < sa.PLATE_KEEPOUT + spec["footprint_mm"]:
                continue
            q_h = spot_pose(x, y, q_ref, psi_ref)
            if q_h is None:
                continue
            worst, best, psi = evaluate((x, y), q_h, spec, jaw, COARSE)
            spots.append((worst, x, y, q_h))
            keys |= {tuple(v["key"]) for v in best.values()}
            print(f"spot r {r} ang {ang} ({x:.0f}, {y:.0f}): worst face score {worst:.0f}",
                  {f: best.get(f, {}).get("score", 0) for f in sv.FACES}, flush=True)
    if not spots:
        sys.exit("no reachable spot")
    spots.sort(key=lambda s: -s[0])
    worst, x, y, q_h = spots[0]
    if worst <= 0:
        sys.exit("no spot shows every face")
    worst_f, best_f, psi = evaluate((x, y), q_h, spec, jaw, FINE)
    keys |= {tuple(v["key"]) for v in best_f.values()}
    lib = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "relative survey views (TCP offset mm from the hover above the brick, wrist roll): the best view of "
                   "each face at every candidate spot of the workspace search",
           "keys": sorted([list(k) for k in keys]), "from_spots": len(spots),
           "criteria": {"min_cos": FINE["min_cos"], "max_jaw": FINE["max_jaw"], "max_occ": 0.02}}
    q_put = kin.ik_rel(q_h, [0, 0, -(pa.HOVER_DZ - pa.TIP_FINAL - 1.0)])
    plan = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spot_xy": [round(x, 1), round(y, 1)],
            "q_hover": q_h, "q_put": q_put, "brick_yaw_rad": round(psi, 5), "worst_face_score": round(worst_f, 1),
            "views": best_f, "search": FINE,
            "inputs": {"wrist_cam_model": sha(os.path.join(APP, "eyeinhand", "wrist_cam_model.json")),
                       "robot_calibration": sha(os.path.join(APP, "motion", "project_robots.json")),
                       "jaw_mask": sha(os.path.join(APP, "tools", "wrist_jaw_mask.png")),
                       "brick_spec": sha(sv.SPEC_F)},
            "ranked_spots": [[round(s[0]), round(s[1], 1), round(s[2], 1)] for s in spots[:8]]}
    print(json.dumps({k: v for k, v in plan.items() if k != "views"}, indent=1))
    for f in sv.FACES:
        print(f, plan["views"].get(f))
    print(f"planned in {time.time() - t0:.0f} s")
    if q_put is None or len(best_f) < len(sv.FACES):
        sys.exit("plan incomplete: not written")
    print("library keys", len(lib["keys"]), "distinct offsets", len({tuple(k[:3]) for k in lib["keys"]}))
    if not a.dry:
        os.makedirs(os.path.dirname(sv.PLAN_F), exist_ok=True)
        lib["inputs"] = plan["inputs"]
        with open(sv.LIB_F + ".tmp", "w") as f:
            json.dump(lib, f, indent=1)
        os.replace(sv.LIB_F + ".tmp", sv.LIB_F)
        with open(sv.PLAN_F + ".tmp", "w") as f:
            json.dump(plan, f, indent=1)
        os.replace(sv.PLAN_F + ".tmp", sv.PLAN_F)
        print("wrote", sv.PLAN_F)


if __name__ == "__main__":
    main()
