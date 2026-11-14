"""Inspection v2: brick geometry, the colour-agnostic face test on synthetic faces, and the full-surface decision."""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import faces as fc  # noqa: E402
import kin  # noqa: E402
import sort_loop as sl  # noqa: E402
import survey as sv  # noqa: E402

MODEL = {"k_chroma": 6.0, "k_light": 6.0, "noise_chroma": 1.0, "noise_light": 1.5, "erode_frac": 0.08,
         "max_region_mm2": 6.0, "min_valid_frac": 0.5}  # test parameters only; the loop uses the fitted model file


def test_survey_brick_faces_geometry():
    f = sv.brick_faces((200.0, 50.0), math.radians(30), (31.8, 31.8), 19.2, kin.MAT_Z)
    assert set(f) == set(sv.FACES)
    assert abs(f["top"]["area_mm2"] - 31.8 ** 2) < 1e-6
    for k in range(4):
        n = f[f"side{k}"]["n"]
        assert abs(np.linalg.norm(n) - 1) < 1e-9 and abs(n[2]) < 1e-9
        ctr = np.mean(f[f"side{k}"]["pts"], axis=0)
        assert np.dot(ctr[:2] - np.array([200.0, 50.0]), n[:2]) > 15.0  # outward, half the footprint away


def _face_image(colour, tape=None):
    rng = np.random.default_rng(1)
    img = rng.normal(25, 4, (240, 320, 3)).clip(0, 255).astype(np.uint8)
    poly = np.array([[80, 60], [240, 60], [240, 180], [80, 180]], float)
    cv2.fillConvexPoly(img, poly.astype(np.int32), colour)
    if tape is not None:
        cv2.rectangle(img, (130, 100), (190, 116), tape, -1)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img, poly


def test_survey_face_test_is_colour_agnostic():
    """Clean faces of several plastic colours pass; tape of a different colour, and a same-hue lighter patch, fail."""
    for colour in ((200, 120, 30), (40, 200, 230), (60, 60, 200), (235, 235, 235), (40, 160, 60)):
        img, poly = _face_image(colour)
        assert fc.score(img, poly, 31.8 * 31.8, MODEL)["verdict"] == "CLEAN", colour
        for tape in ((170, 170, 170), (20, 20, 20), (30, 30, 220)):
            if np.abs(np.array(tape) - np.array(colour)).max() < 40:
                continue
            img2, _ = _face_image(colour, tape)
            assert fc.score(img2, poly, 31.8 * 31.8, MODEL)["verdict"] == "DEFECT", (colour, tape)


def test_survey_no_model_means_uncertain():
    assert fc.load_model("/nonexistent/face_model.json") is None


def test_survey_decide_faces_rule():
    req = sv.FACES
    clean = {f: {"verdict": "CLEAN"} for f in req}
    assert sl.decide_faces({}, ("2x2", ""), clean, req)[:2] == ("blue", "GOOD")
    one_bad = dict(clean, side2={"verdict": "DEFECT"})
    assert sl.decide_faces({}, ("2x2", ""), one_bad, req)[:2] == ("green", "DEFECT")
    missing = {f: v for f, v in clean.items() if f != "side3"}
    assert sl.decide_faces({}, ("2x2", ""), missing, req)[:3] == ("green", "UNCERTAIN", "incomplete")
    assert sl.decide_faces({}, ("not_2x2", "aspect 2"), clean, req)[:2] == ("green", "DEFECT")
    assert sl.decide_faces({"verdict": "CONFIDENT_DEFECT", "reasons": ["tape"]}, ("2x2", ""), clean, req)[:2] == ("green", "DEFECT")
    assert sl.decide_faces({}, ("unknown", ""), clean, req)[:3] == ("green", "UNCERTAIN", "shape_unknown")
