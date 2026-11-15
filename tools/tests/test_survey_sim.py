"""The in-place survey end to end on a virtual arm: moves set the pose, frames are rendered from it (sim_scene), the
code under test plans the views, aligns, segments and judges, and decide_faces gives the plate. Thresholds below are
test values for the logic only; the loop uses the face model fitted from real bricks."""
import json
import math
import os
import sys
import tempfile
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import kin  # noqa: E402
import sim_scene as sim  # noqa: E402
import sort_any as sa  # noqa: E402
import survey as sv  # noqa: E402

MODEL = {"kappa_chroma": 2.0, "kappa_light_top": 2.0, "kappa_light_side": 2.0}  # test values (8-bit levels) for the logic only
XY = (232.0, 38.0)
ORANGE, BLUE, GREY = (40, 120, 230), (200, 110, 30), (175, 175, 175)


class Arm:
    def __init__(self, q, bricks, plates=()):
        self.q, self.bricks, self.plates, self.moves = list(q)[:5], bricks, plates, 0

    def move(self, q6, **_):
        self.q, self.moves = list(q6)[:5], self.moves + 1
        return {"status": "ok"}

    def state(self):
        return {"present": self.q + [25.0]}

    def frame(self):
        return sim.render(self.q, self.bricks, self.plates, seed=self.moves)


def _run(bricks, plates=(), out_dir=None, model=None, xy=XY):
    psi = kin.psi_of(sa.SPOT_HOVER)
    q_h = kin.ik(xy[0], xy[1], kin.MAT_Z + sa.pa.HOVER_DZ, psi, sa.SPOT_HOVER)
    assert q_h is not None
    arm = Arm(q_h, bricks, plates)
    z_top = kin.MAT_Z + sv.spec()["height_mm"]
    bh = {"centre": sv.wc.project(q_h, np.array([bricks[0]["xy"][0], bricks[0]["xy"][1], z_top])).tolist()}
    q_scan = sa.scan_cal()["q_scan"]
    with tempfile.TemporaryDirectory() as d:
        with mock.patch.object(sa.dfx, "load_model", lambda *a, **k: dict(model or MODEL)), \
                mock.patch.object(sa, "log", lambda *a, **k: None), \
                mock.patch.object(sa.ga, "move", arm.move), mock.patch.object(sa.ga, "state", arm.state), \
                mock.patch.object(sa.pa, "frame", arm.frame), mock.patch.object(sa.ga, "OUT", out_dir or d, create=True), \
                mock.patch.dict(sa.EPIMG, {"pre": {"wrist": sim.render(q_scan, bricks, plates, seed=99)}}, clear=True):
            out = sa.survey_here(1, q_h, bh, {"xy": list(xy)}, [])
    return out, arm


def test_survey_sim_clean_brick_alone_all_faces_clean_and_blue():
    out, arm = _run([sim.brick(XY, 0.3, ORANGE)])
    v = {f: r.get("verdict") for f, r in out["faces"].items()}
    assert out["shape"][0] == "2x2", out.get("measure")
    assert set(v) == set(sv.FACES) and all(x == "CLEAN" for x in v.values()), (v, out.get("missing"))
    plate, label, conf, _ = sa.sl.decide_faces({}, tuple(out["shape"]), out["faces"], sv.FACES)
    assert (plate, label) == ("blue", "GOOD")


def test_survey_sim_tape_on_each_hidden_side_is_found():
    """A grey tape patch on any one side face (hidden from the hover) makes that brick DEFECT."""
    for side in ("side0", "side1", "side2", "side3"):
        out, _ = _run([sim.brick(XY, 0.3, BLUE, tapes=[(side, 0.25, 0.75, 0.25, 0.75, GREY)])])
        bad = [f for f, r in out["faces"].items() if r.get("verdict") == "DEFECT"]
        assert bad, (side, {f: r.get("verdict") for f, r in out["faces"].items()}, out.get("missing"))
        assert sa.sl.decide_faces({}, tuple(out["shape"]), out["faces"], sv.FACES)[:2] == ("green", "DEFECT")


def test_survey_sim_neighbour_close_means_survey_at_the_spot():
    for gap in (10.0, 25.0):  # nearly touching, and a clear gap inside the isolation radius
        out, arm = _run([sim.brick(XY, 0.3, ORANGE), sim.brick((XY[0] + 31.8 + gap, XY[1] + 5.0), 0.1, BLUE)])
        assert not out["faces"] and set(out["missing"]) == set(sv.FACES), out
        assert out.get("isolation", {}).get("isolated") is False or "touching a neighbour" in out.get("why", ""), out
        assert arm.moves == 0, "the arm surveyed although a neighbour was within the radius"


def test_survey_sim_2x4_is_not_2x2():
    out, _ = _run([sim.brick(XY, 0.3, ORANGE, size=(31.8, 63.6))])
    assert out["shape"][0] == "not_2x2", out.get("measure")


def test_scene_plates_found_and_brick_clusters_not_taken_for_plates():
    """Both plates are found from the scan frame (centre and radius on the mat), a square of four touching blue bricks
    next to them is not taken for a plate, and the drop points stay on the plate."""
    import scene
    q_scan = sa.scan_cal()["q_scan"]
    truth = {"blue": ((187.1, -86.1), 75.0, (200, 90, 20)), "green": ((194.6, 218.3), 75.0, (60, 150, 40))}
    cluster = [sim.brick((262.0 + 31.8 * i, 40.0 + 31.8 * j), 0.0, (200, 90, 20)) for i in range(2) for j in range(2)]
    img = sim.render(q_scan, cluster, list(truth.values()), seed=5)
    found = scene.plates(img, q_scan, sv.spec())
    assert set(found) == set(truth) and all(v["reliable"] for v in found.values()), found
    for k, (xy, r, _) in truth.items():
        assert math.hypot(found[k]["xy"][0] - xy[0], found[k]["xy"][1] - xy[1]) < 3.0 and abs(found[k]["r_mm"] - r) < 3.0, found
    hidden = scene.plates(img, q_scan, sv.spec(), ignore=sa.ga.JAW_MASK)  # the jaws hide part of the view
    assert set(hidden) == set(truth), hidden  # still found by colour; the circle only where enough rim shows
    for k, v in hidden.items():
        if v["reliable"]:
            assert math.hypot(v["xy"][0] - truth[k][0][0], v["xy"][1] - truth[k][0][1]) < 3.0, (k, v)
    with tempfile.TemporaryDirectory() as d, mock.patch.object(sa, "PLATES_F", os.path.join(d, "plates.json")), \
            mock.patch.object(sa.ga, "JAW_MASK", None):  # the virtual camera shows no jaws
        sa.see_plates(img, q_scan)
        for k, (xy, r, _) in truth.items():
            pts = sa.drop_points(k)
            assert pts and all(math.hypot(p[0] - xy[0], p[1] - xy[1]) <= r - 0.75 * 31.8 + 1e-6 for p in pts), pts
        assert sa.near_plate((187.0, -30.0), sa.plates_seen()["plates"], sv.spec())
        assert not sa.near_plate((262.0, 40.0), sa.plates_seen()["plates"], sv.spec())


def test_scene_plate_centre_from_a_short_arc_with_a_known_radius():
    """With the jaws hiding most of a plate's rim, the plate's centre still comes out when its radius is known."""
    import scene
    q_scan = sa.scan_cal()["q_scan"]
    truth = {"blue": ((187.1, -86.1), 75.0, (200, 90, 20)), "green": ((194.6, 218.3), 75.0, (60, 150, 40))}
    img = sim.render(q_scan, [], list(truth.values()), seed=6)
    entry = dict(kin.cell.C["plate_radius_mm"], value=75.0)
    with mock.patch.dict(kin.cell.C, {"plate_radius_mm": entry}):
        found = scene.plates(img, q_scan, sv.spec(), ignore=sa.ga.JAW_MASK)
    for k, (xy, r, _) in truth.items():
        v = found[k]
        assert v["reliable"] and math.hypot(v["xy"][0] - xy[0], v["xy"][1] - xy[1]) < 4.0, (k, {a: b for a, b in v.items() if a != "mask"})
