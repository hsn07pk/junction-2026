"""Replay tests for app/tools/kin.py (URDF FK and differential IK) and the reach check in sort_any, using the
calibration copy in tests/fixtures.

Protects: the scan and hover heights used by the pick code; ik_rel's round trip over a 5x5 grid of +-60 mm at hover
height (error < 1.5 mm wherever it returns a pose); the reach limit of a vertical-jaw grasp at hover height (radius
from the pan axis about 296 mm, the reach map log); and sort_any.grasp_reachable, which keeps bricks
beyond that limit out of the candidate list (3 of 6 approach failures in runs any_1731/1740 were IK failures).
"""
import csv
import math
import os

import numpy as np

import conftest as C

REACH_IN = 290.0   # every grid point closer than this to the pan axis must be reachable
REACH_OUT = 300.0  # no grid point farther than this may be reachable (vertical jaws at hover height)


def _r(p):
    return math.hypot(p[0] - C.kin.PAN_XY_MM[0], p[1] - C.kin.PAN_XY_MM[1])


def test_fk_scan_and_hover_heights():
    """Fixture 7: fk(q_scan) is well above fk(q_hover); fk(q_hover) TCP z is within 5 mm of MAT_Z + HOVER_DZ."""
    c = C.load_json("scan_calib.json")
    z_scan = float(C.kin.fk(c["q_scan"])[0][2])
    z_hover = float(C.kin.fk(c["q_hover"])[0][2])
    assert z_scan > z_hover + 50, f"scan z {z_scan:.2f} mm is not 50 mm above hover z {z_hover:.2f} mm"
    nominal = C.kin.MAT_Z + C.kin.HOVER_DZ
    assert abs(z_hover - nominal) < 5.0, f"hover z {z_hover:.2f} mm vs nominal {nominal:.2f} mm"


def test_ik_rel_round_trip_5x5_grid_60mm():
    """Fixture 7: for d in a 5x5 grid of +-60 mm at hover height, ik_rel(q_hover, d) returns a pose whose fk is within
    1.5 mm of fk(q_hover) + d at every point closer than REACH_IN to the pan axis, and returns None beyond REACH_OUT.
    At q_hover (TCP r = 274 mm) the points +60/+30 and +60/+60 mm lie at r = 303 and 313 mm, outside the reach."""
    c = C.load_json("scan_calib.json")
    q = c["q_hover"]
    p0 = np.asarray(C.kin.fk(q)[0], float)
    fails = []
    for dx in (-60, -30, 0, 30, 60):
        for dy in (-60, -30, 0, 30, 60):
            d = np.array([dx, dy, 0.0], float)
            r = _r(p0 + d)
            qn = C.kin.ik_rel(q, d)
            if qn is None:
                if r < REACH_IN:
                    fails.append((dx, dy, round(r, 1), "ik_rel None inside the reach"))
                continue
            if r > REACH_OUT:
                fails.append((dx, dy, round(r, 1), "ik_rel returned a pose beyond the reach"))
            err = float(np.linalg.norm(np.asarray(C.kin.fk(qn)[0], float) - p0 - d))
            if err >= 1.5:
                fails.append((dx, dy, round(r, 1), round(err, 3)))
    assert not fails, f"ik_rel grid failures (dx, dy, r mm, problem): {fails}"


def test_grasp_reachable_matches_logged_outcomes():
    """Sort_any.grasp_reachable against fixtures/reach_cases.tsv (brick_xy_est of run any_1740, logs/sort/any_1740/
    cycles.jsonl): the four picked bricks (r 179 to 283 mm) and four bricks that failed for other reasons are
    reachable; the episode-9 brick (r 345 mm, logged 'ik fail moving toward brick') is not. Also: the scan point is
    reachable, and points 305 and 330 mm straight out from the pan axis are not."""
    c = C.load_json("scan_calib.json")
    p = np.asarray(C.kin.fk(c["q_scan"])[0], float)
    with open(os.path.join(C.FIX, "reach_cases.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    bad = []
    for row in rows:
        x, y = float(row["x_mm"]), float(row["y_mm"])
        got = C.sa.grasp_reachable(c["q_scan"], (x - p[0], y - p[1]))
        if got != (row["expect_reachable"] == "1"):
            bad.append((row["case"], row["expect_reachable"], round(_r((x, y)), 1), got))
    if not C.sa.grasp_reachable(c["q_scan"], (0.0, 0.0)):
        bad.append(("scan point", "1", round(_r(p), 1), False))
    a = math.atan2(p[1] - C.kin.PAN_XY_MM[1], p[0] - C.kin.PAN_XY_MM[0])
    for r in (305.0, 330.0):
        x, y = C.kin.PAN_XY_MM[0] + r * math.cos(a), C.kin.PAN_XY_MM[1] + r * math.sin(a)
        if C.sa.grasp_reachable(c["q_scan"], (x - p[0], y - p[1])):
            bad.append((f"radial {r:.0f} mm", "0", r, True))
    assert not bad, f"grasp_reachable disagrees (case, expected, r mm, got): {bad}"
    return f"{len(rows)} logged bricks and 3 geometric cases checked"
