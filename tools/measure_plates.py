#!/usr/bin/env python3
"""Measure where the two plates are, with the wrist camera looking down at each one.

For each plate the arm goes above the plate end of its taught path (the taught release pose), high enough for the plate
to fit in the wrist view, photographs it and fits the rim circle on the mat (scene.plates with the wrist camera model).
The measured centres and radii go to run/plates_seen.json, which the sort loop uses to keep plate pixels out of the
brick candidates and to compute IK drop points inside each plate. With --radius the mean measured radius is also
written to config/cell_calibration.json (plate_radius_mm), so short rim arcs seen in scan frames fix a plate too.

usage: measure_plates.py [--radius]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, T)
import grasp_auto as ga  # noqa: E402
import kin  # noqa: E402
import scene  # noqa: E402
import sort_any as sa  # noqa: E402
import sort_loop as sl  # noqa: E402
import survey as srv  # noqa: E402

HEIGHTS_MM = (150.0, 130.0, 115.0, 100.0, 85.0, 70.0, 60.0)  # look-down heights above the mat; the highest reachable is used


def look(plate: str, out_dir: str) -> dict | None:
    q_rel = list(sa.WP[f"{plate}_wp0_closed"][:5])
    p, _ = kin.fk(q_rel)
    best = None
    for h in HEIGHTS_MM:
        q = kin.ik(p[0], p[1], kin.MAT_Z + h, kin.psi_of(q_rel), q_rel)
        if q is None:
            sa.log(f"plates: {plate} look pose at {h:.0f} mm not reachable")
            continue
        ga.move(list(q) + [25.0], speed=sa.CFG["transit"], comp=True)
        time.sleep(0.3)
        img = ga.fresh("wrist", time.time() + 0.1)
        qm = [float(v) for v in ga.state()["present"][:5]]
        if img is None:
            continue
        cv2.imwrite(f"{out_dir}/{plate}_{h:.0f}.jpg", img)
        res = scene.plates(img, qm, srv.spec(), ignore=ga.JAW_MASK).get(plate)
        if res is None:
            sa.log(f"plates: {plate} not found from {h:.0f} mm")
            continue
        rec = {k: v for k, v in res.items() if k != "mask"}
        sa.log(f"plates: {plate} from {h:.0f} mm: {rec}")
        if rec.get("reliable"):
            best = dict(rec, t=time.time(), look_h_mm=h, taught_release_xy=[round(float(p[0]), 1), round(float(p[1]), 1)])
        break  # the highest reachable view shows the most rim
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", action="store_true", help="also write the mean radius to cell_calibration.json")
    a = ap.parse_args()
    out_dir = f"{ga.LOG_ROOT}/plates/measure_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    os.makedirs(out_dir, exist_ok=True)
    sa.RUN = out_dir
    sl.RUN = out_dir
    ga.OUT = out_dir
    sa.EP = {"episode": 0, "mode": "plate_measurement"}
    sl.wait_clear("plate measurement")
    sa.go_scan(False)
    found = {}
    for plate in ("blue", "green"):  # the blue plate is seen from higher (nearer the base): its radius helps the green fit
        rec = look(plate, out_dir)
        sa.go_scan(False)
        if rec is not None:
            found[plate] = rec
            if rec.get("radius") != "known" and kin.cell.get("plate_radius_mm") is None:
                kin.cell.C["plate_radius_mm"]["value"] = rec["r_mm"]  # same plates: a short rim arc of the other one now fixes it
    print(json.dumps(found))
    if not found:
        return 2
    old = sa.plates_seen()
    old["plates"].update(found)
    old["placed"] = {}
    with open(sa.PLATES_F + ".tmp", "w") as f:
        json.dump(old, f)
    os.replace(sa.PLATES_F + ".tmp", sa.PLATES_F)
    print("wrote", sa.PLATES_F)
    measured = [v["r_mm"] for v in found.values() if v.get("radius") != "known"]
    if a.radius and measured:
        r = measured if len(measured) == 2 else measured * 2
        if abs(r[0] - r[1]) <= 0.1 * max(r):
            f = os.path.join(os.path.dirname(T), "config", "cell_calibration.json")
            cfg = json.load(open(f))
            cfg["plate_radius_mm"] = {"value": round(sum(r) / 2.0, 1),
                                      "how": f"measure_plates.py {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: "
                                             f"mean rim radius of both plates seen from above ({r[0]} and {r[1]} mm)"}
            with open(f + ".tmp", "w") as fh:
                json.dump(cfg, fh, indent=1)
            os.replace(f + ".tmp", f)
            print("plate_radius_mm", cfg["plate_radius_mm"])
    return 0 if len(found) == 2 else 1


if __name__ == "__main__":
    sys.exit(main())
