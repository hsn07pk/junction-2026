"""The surface defect test end to end on the virtual arm: the calibration set (three clean bricks, three taped ones,
each surveyed at several orientations) is surveyed, labelled and calibrated (calibrate_defects.py, hold-out by
physical brick), then bricks of colours and tapes the calibration never saw are surveyed with the fitted model and
decided with the loop's rule. Brick and tape colours exist only in this test's virtual table and labels."""
import json
import os
import sys
import tempfile
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import calibrate_defects as cd  # noqa: E402
import sim_scene as sim  # noqa: E402
import sort_loop as sl  # noqa: E402
import survey as sv  # noqa: E402
import test_survey_sim as ts  # noqa: E402

GREY, BLACK, RED, WHITE = (175, 175, 175), (30, 30, 30), (40, 40, 210), (235, 235, 235)
CAL = {  # physical brick id -> (colour, tapes)
    "b1": ((40, 120, 230), []), "b2": ((215, 215, 210), []), "b3": ((40, 70, 120), []),
    "b4": ((200, 110, 30), [("side1", 0.2, 0.8, 0.3, 0.7, GREY)]),
    "b5": ((150, 120, 235), [("side3", 0.1, 0.5, 0.2, 0.9, BLACK)]),
    "b6": ((40, 210, 230), [("top", 0.3, 0.7, 0.3, 0.7, RED)]),
}
UNSEEN = [  # (colour, tapes, truth): colours and tapes not in the calibration set
    ((60, 160, 40), [], "GOOD"), ((40, 40, 200), [], "GOOD"), ((150, 150, 150), [], "GOOD"),
    ((120, 50, 20), [], "GOOD"), ((170, 210, 230), [], "GOOD"),
    ((40, 40, 200), [("side2", 0.2, 0.7, 0.2, 0.8, WHITE)], "DEFECT"),
    ((215, 215, 210), [("side0", 0.3, 0.7, 0.2, 0.8, GREY)], "DEFECT"),
    ((40, 120, 230), [("side1", 0.4, 0.6, 0.35, 0.65, BLACK)], "DEFECT"),  # 6 mm square
    ((60, 160, 40), [("side0", 0.0, 1.0, 0.0, 1.0, WHITE)], "DEFECT"),  # a whole side covered
    ((200, 110, 30), [("side0", 0.7, 1.0, 0.3, 0.7, (40, 210, 230)), ("side1", 0.0, 0.3, 0.3, 0.7, (40, 210, 230))],
     "DEFECT"),  # over an edge
    ((40, 120, 230), [("top", 0.1, 0.9, 0.4, 0.7, (60, 160, 40))], "DEFECT"),  # across the studs
    ((230, 180, 90), [("side2", 0.2, 0.8, 0.3, 0.7, (120, 50, 20))], "DEFECT"),  # same hue, darker
    ((215, 215, 210), [("top", 0.3, 0.7, 0.3, 0.7, WHITE)], "DEFECT"),  # white on white
    ((140, 60, 120), [("side3", 0.0, 0.3, 0.0, 0.45, GREY)], "DEFECT"),  # in a corner
]
_MODEL = {}


def _calibrated_model():
    if _MODEL:
        return _MODEL
    with tempfile.TemporaryDirectory() as root:
        labels = {"bricks": {b: ("DEFECT" if t else "GOOD") for b, (_, t) in CAL.items()}, "episodes": {},
                  "source": "virtual table"}
        k = 0
        for bid, (bgr, tapes) in CAL.items():
            for yaw in (0.2, 0.9):
                k += 1
                ep = os.path.join(root, "sort", "virt", f"ep_{k:04d}")
                os.makedirs(ep)
                out, _ = ts._run([sim.brick(ts.XY, yaw, bgr, tapes=tapes)], out_dir=ep)
                json.dump({"survey": out}, open(os.path.join(ep, "episode.json"), "w"), default=float)
                labels["episodes"][f"virt/ep_{k:04d}"] = bid
        lf = os.path.join(root, "labels.json")
        json.dump(labels, open(lf, "w"))
        mf = os.path.join(root, "defect_model.json")
        with mock.patch.object(cd, "LOG_ROOT", root), mock.patch.object(cd.dfx, "MODEL_F", mf), \
                mock.patch.object(sys, "argv", ["calibrate_defects.py", lf, "--write"]):
            cd.main()
        _MODEL.update(json.load(open(mf)))
    return _MODEL


def test_defects_calibration_holds_out_each_physical_brick():
    m = _calibrated_model()
    p = m["provenance"]
    assert p["holdout_right"] == p["holdout_total"] == 12 and not p["defects_missed"], p
    return f"model {', '.join(f'{k}={m[k]:.4g}' for k in cd.CHANNELS)}"


def test_defects_unseen_colours_and_tapes():
    """Clean bricks of other colours go blue; tapes of other colours, sizes and places (a small square, a whole side,
    over an edge, across the studs, same hue, white on white, a corner) go green as DEFECT."""
    model = {k: v for k, v in _calibrated_model().items() if k != "provenance"}
    wrong = []
    for i, (bgr, tapes, truth) in enumerate(UNSEEN):
        out, _ = ts._run([sim.brick(ts.XY, (0.15, 0.55, 1.1, 1.35)[i % 4], bgr, tapes=tapes)], model=model)
        label = sl.decide_faces({}, tuple(out.get("shape", ("unknown", ""))), out.get("faces") or {}, sv.FACES)[1]
        if label != truth:
            wrong.append((i, bgr, tapes, truth, label, {f: r.get("verdict") for f, r in (out.get("faces") or {}).items()}))
    assert not wrong, wrong
    return f"{len(UNSEEN)} unseen bricks right"
