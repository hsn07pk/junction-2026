"""Inspection: brick geometry, the parts of the surface defect test on synthetic data, and the full-surface decision."""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import defects as dfx  # noqa: E402
import kin  # noqa: E402
import sort_loop as sl  # noqa: E402
import survey as sv  # noqa: E402

def test_survey_brick_faces_geometry():
    f = sv.brick_faces((200.0, 50.0), math.radians(30), (31.8, 31.8), 19.2, kin.MAT_Z)
    assert set(f) == set(sv.FACES)
    assert abs(f["top"]["area_mm2"] - 31.8 ** 2) < 1e-6
    for k in range(4):
        n = f[f"side{k}"]["n"]
        assert abs(np.linalg.norm(n) - 1) < 1e-9 and abs(n[2]) < 1e-9
        ctr = np.mean(f[f"side{k}"]["pts"], axis=0)
        assert np.dot(ctr[:2] - np.array([200.0, 50.0]), n[:2]) > 15.0  # outward, half the footprint away


def test_defects_no_model_means_uncertain():
    assert dfx.load_model("/nonexistent/defect_model.json") is None
    res = dfx.judge({"ok": False, "why": "x"}, None, dfx.config())
    assert all(r["verdict"] == "UNCERTAIN" for r in res.values())


def test_defects_lag_sigma_measures_noise_not_the_patch():
    """A patch larger than a window leaves the lag differences to the noise."""
    rng = np.random.default_rng(3)
    m = rng.normal(0, 0.01, (40, 60))
    m[10:30, 20:45] += 0.5  # a defect-sized step
    s = dfx._lag_sigma(m, np.ones(m.shape, bool), 8)
    assert 0.008 < s < 0.03, s


def test_defects_clipped_halo_grows_with_the_highlight():
    img = np.full((120, 160, 3), 100, np.uint8)
    cv2.circle(img, (80, 60), 6, (255, 255, 255), -1)
    h = dfx.clipped_halo(img)
    assert h[60, 80] and h[60, 80 + 15] and not h[60, 80 + 25]


def test_defects_body_line_ignores_a_taped_minority():
    rng = np.random.default_rng(4)
    ell = rng.uniform(4.5, 6.0, 3000)
    rg = np.stack([0.5 - 0.03 * (ell - 5.2), 0.3 + 0.01 * (ell - 5.2)], 1) + rng.normal(0, 0.002, (3000, 2))
    rg[:900] = [0.33, 0.33]  # 30 % tape
    b = dfx._body_line(ell, rg, 0.05)
    assert abs(b["b"][0] + 0.03) < 0.005 and abs(b["a"][0] - (0.5 - 0.03 * (b["l0"] - 5.2))) < 0.003, b


def test_defects_trim_edges_drops_a_contaminated_band_only():
    ok = np.ones((40, 60), bool)
    vv, uu = np.mgrid[0:40, 0:60]
    edge_d = np.minimum.reduce([uu, vv, 59 - uu, 39 - vv]).astype(float)
    edge_id = np.argmin(np.stack([vv, 59 - uu, 39 - vv, uu], -1), -1)
    r = np.random.default_rng(5).normal(0, 0.01, (40, 60, 1))
    r[vv < 3] += 0.3  # mixing along one edge (edge 0: v = 0)
    out = dfx.trim_edges(ok, edge_d, edge_id, r, 0.0, 6.0)
    along = (uu >= 3) & (uu <= 56)  # the corners belong to the other edges
    assert not out[(vv < 3) & along].any() and out[(vv >= 4) & (vv < 36) & (uu > 6) & (uu < 53)].all()


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


def test_calibration_run_labels_episodes_in_the_labels_file_only():
    """A calibration cycle adds its episode to the labels file with the operator's truth; nothing else stores it."""
    import json
    import tempfile
    from unittest import mock
    import sort_any as sa
    with tempfile.TemporaryDirectory() as d:
        lf = os.path.join(d, "labels.json")
        with mock.patch.object(sa, "LABELS_F", lf), mock.patch.dict(sa.CFG, {"calibrate": {"brick": "x7", "truth": "GOOD"}}), \
                mock.patch.object(sa, "RUN", os.path.join(d, "cal_a")), mock.patch.object(sa, "EP", {"survey": {"faces": {}}}):
            sa.label_episode(3)
            sa.label_episode(4)
        lab = json.load(open(lf))
        assert lab["bricks"] == {"x7": "GOOD"} and lab["episodes"] == {"cal_a/ep_0003": "x7", "cal_a/ep_0004": "x7"}, lab
        with mock.patch.object(sa, "LABELS_F", lf), mock.patch.dict(sa.CFG, {"calibrate": None}), \
                mock.patch.object(sa, "RUN", os.path.join(d, "run_b")), mock.patch.object(sa, "EP", {"survey": {}}):
            sa.label_episode(5)  # not a calibration run: nothing added
        assert len(json.load(open(lf))["episodes"]) == 2
