"""Wrist-camera self-calibration on the virtual arm: the scene is rendered through a camera that differs from the
model the code uses (focal length, principal point, radial term, camera pose on the gripper, joint offsets), clean
bricks are surveyed with the wrong model, calib_wristcam fits a model from those surveys, and the refined model must
predict surveys it did not see far better and let the survey judge a new clean brick correctly."""
import json
import os
import sys
import tempfile
from unittest import mock

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import calib_wristcam as cw  # noqa: E402
import defects as dfx  # noqa: E402
import sim_scene as sim  # noqa: E402
import sort_loop as sl  # noqa: E402
import survey as sv  # noqa: E402
import test_survey_sim as ts  # noqa: E402

PERTURB = np.array([12.0, 6.0, -5.0, 0.03, 0.01, -0.015, 0.01, 2.0, -2.0, 1.5, 0.5, -0.8, 0.6, 0.4, 0.3])
SPOTS = [(232.0, 38.0), (215.0, 60.0), (245.0, 10.0), (225.0, 20.0)]
COLOURS = [(40, 120, 230), (215, 215, 210), (40, 70, 120), (60, 160, 40)]


def test_calib_wristcam_recovers_a_wrong_camera_model():
    M0 = dict(sv.wc.M)
    th0 = cw.params_of(M0)
    true = cw.model_of(th0 + PERTURB, M0)
    render = sim.render

    def render_true(*a, **k):
        with cw.installed(true):
            return render(*a, **k)

    sp, cfg = sv.spec(), dfx.config()
    jm = cv2.imread(os.path.join(os.path.dirname(HERE), "wrist_jaw_mask.png"), 0)
    jaw = (jm > 0) if jm is not None else None
    with tempfile.TemporaryDirectory() as root, mock.patch.object(sim, "render", render_true):
        k = 0
        for xy, bgr in zip(SPOTS, COLOURS):
            for yaw in (0.2, 0.9):
                k += 1
                ep = os.path.join(root, "sort", "virt", f"ep_{k:04d}")
                os.makedirs(ep)
                out, _ = ts._run([sim.brick(xy, yaw, bgr)], out_dir=ep, xy=xy)
                json.dump({"survey": out}, open(os.path.join(ep, "episode.json"), "w"), default=float)
        with mock.patch.object(cw, "LOG_ROOT", root):
            surveys = cw.surveys_in(None)
        assert len(surveys) == 8, len(surveys)
        test_ids = {surveys[1]["id"], surveys[6]["id"]}
        th, data = cw.fit_model(M0, [s for s in surveys if s["id"] not in test_ids], sp, cfg, jaw)
        held = [dict(s, obs=cw.observe(s, M0, sp, cfg, jaw)) for s in surveys if s["id"] in test_ids]
        e_old, e_new = np.median(cw.pixel_errors(th0, held, sp)), np.median(cw.pixel_errors(th, held, sp))
        assert e_new < 0.4 * e_old and e_new < 4.0, (e_old, e_new)
        with cw.installed(cw.model_of(th, M0)):  # a new clean brick, judged with the refined model
            out, _ = ts._run([sim.brick((228.0, 45.0), 0.5, (40, 40, 200))], xy=(228.0, 45.0))
        label = sl.decide_faces({}, tuple(out.get("shape", ("unknown", ""))), out.get("faces") or {}, sv.FACES)[1]
        assert label == "GOOD", {f: r.get("verdict") for f, r in (out.get("faces") or {}).items()}
    return f"held-out corner error {e_old:.1f} px -> {e_new:.1f} px"
