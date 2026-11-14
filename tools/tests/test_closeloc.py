"""Closeloc: localisation by frame difference and the close-camera plane map, on synthetic frames and on one recorded
episode."""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import closeloc as cl  # noqa: E402

MAT = (28, 30, 32)          # dark mat, BGR
ORANGE = (30, 110, 235)
WHITE = (225, 228, 230)
RED = (35, 30, 200)


def _frame():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = MAT
    return img


def _brick(img, x, y, w=70, h=50, colour=ORANGE):
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), colour, -1)
    return {"bbox": [x, y, w, h], "centre_px": [x + w / 2.0, y + h / 2.0]}


def test_appeared_brick_is_the_new_one():
    before, after = _frame(), _frame()
    a0 = _brick(before, 100, 100, colour=WHITE)
    a1 = _brick(after, 100, 100, colour=WHITE)
    b1 = _brick(after, 420, 330)
    rec, frac, second = cl.pick_changed([a1, b1], cl.changed_mask(after, before))
    assert rec is b1 and frac > 0.9 and second < 0.05, (frac, second)
    assert a0["bbox"] == a1["bbox"]


def test_vanished_brick_is_found_in_the_first_frame():
    before, after = _frame(), _frame()
    keep = _brick(before, 100, 100, colour=WHITE)
    gone = _brick(before, 300, 200)
    _brick(after, 100, 100, colour=WHITE)
    rec, frac, _ = cl.pick_changed([keep, gone], cl.changed_mask(after, before))
    assert rec is gone and frac > 0.9


def test_small_lighting_change_is_not_a_brick():
    before = _frame()
    k = _brick(before, 200, 200, colour=WHITE)
    after = np.clip(before.astype(np.int16) + 12, 0, 255).astype(np.uint8)   # whole scene a bit brighter
    rec, frac, _ = cl.pick_changed([k], cl.changed_mask(after, before))
    assert rec is None and frac < 0.05


def test_brick_put_down_against_another_scores_lower_than_a_clean_one():
    before, after = _frame(), _frame()
    _brick(before, 200, 200, colour=WHITE)
    _brick(after, 200, 200, colour=WHITE)
    _brick(after, 270, 200)                                                     # touches the white one
    merged = {"bbox": [200, 200, 140, 50], "centre_px": [270.0, 225.0]}
    _, frac_merged, _ = cl.pick_changed([merged], cl.changed_mask(after, before))
    before2, after2 = _frame(), _frame()
    clean = _brick(after2, 270, 200)
    _, frac_clean, _ = cl.pick_changed([clean], cl.changed_mask(after2, before2))
    assert frac_merged < 0.6 < frac_clean, (frac_merged, frac_clean)


def test_jaw_tips_ignore_a_red_brick_that_did_not_move():
    before, hover = _frame(), _frame()
    _brick(before, 285, 330, colour=RED)
    _brick(hover, 285, 330, colour=RED)
    cv2.rectangle(hover, (270, 90), (292, 260), RED, -1)                        # left jaw
    cv2.rectangle(hover, (340, 90), (362, 255), RED, -1)                        # right jaw
    tips = cl.jaw_tips_by_diff(hover, before)
    assert len(tips) == 2, tips
    (lx, ly), (rx, ry) = tips
    assert abs(lx - 281) < 2 and abs(ly - 260) < 2 and abs(rx - 351) < 2 and abs(ry - 255) < 2, tips


def test_brick_below_the_jaws():
    under = {"bbox": [285, 300, 70, 50], "centre_px": [320.0, 325.0]}
    aside = {"bbox": [470, 280, 70, 50], "centre_px": [505.0, 305.0]}
    above = {"bbox": [285, 120, 70, 50], "centre_px": [320.0, 145.0]}
    rec, d = cl.brick_below([aside, above, under], 316.0, 258.0)
    assert rec is under and d < 70


def test_plane_map_fits_predicts_and_rejects_an_outlier():
    H = np.array([[1.9, 0.12, 40.0], [0.05, -1.4, 520.0], [0.0002, 0.0004, 1.0]])
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as d:
        m = cl.ClosePlaneMap(os.path.join(d, "close_pairs.jsonl"))
        for _ in range(12):
            x, y = rng.uniform(150, 300), rng.uniform(-120, 120)
            u, v = cl.ClosePlaneMap._apply(H, np.array([[x, y]]))[0] + rng.normal(0, 0.5, 2)
            m.add(x, y, u, v, "test")
        m.add(200.0, 0.0, 10.0, 10.0, "outlier")
        med = m.refit()
        assert med is not None and med < 2.0 and m.n_used == 12, (med, m.n_used)
        pu, pv = m.predict(220.0, 30.0)
        tu, tv = cl.ClosePlaneMap._apply(H, np.array([[220.0, 30.0]]))[0]
        assert abs(pu - tu) < 2.0 and abs(pv - tv) < 2.0


def test_plane_map_needs_enough_pairs():
    with tempfile.TemporaryDirectory() as d:
        m = cl.ClosePlaneMap(os.path.join(d, "p.jsonl"))
        for i in range(4):
            m.add(100.0 + i, 5.0 * i, 300.0 + i, 200.0 - i, "test")
        assert m.refit() is None and m.predict(100.0, 0.0) is None


def _mask_top_red(img):
    """Sort_loop.mask_jaws rule (red parts touching the image top are the gripper), without importing the loop."""
    red = cl.red_mask(img)
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    out = img.copy()
    for i in range(1, n):
        if st[i, 1] <= 3 and st[i, 4] > 800:
            out[cv2.dilate((lab == i).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = 0
    return out


def test_station_and_picked_brick_on_recorded_frames():
    """Run any_2055 ep1 (close camera moved before the run): the station brick is the one that appeared and the picked
    brick the one that vanished; the old fixed spot (380, 262) is nearest a different brick (the pink/white pair)."""
    fx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "closeloc")
    pre = cv2.imread(os.path.join(fx, "ep1_pre_close.jpg"))
    st = cv2.imread(os.path.join(fx, "ep1_station_close.jpg"))
    if pre is None or st is None:
        import conftest as C
        if not C.IMAGES_PRESENT:
            raise C.Skip("fixtures/closeloc: recorded frames are local data, not in this checkout")
        raise AssertionError("fixture frames missing")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    "anomaly", "det2_f2"))
    import inspector as DET  # noqa: E402
    ch = cl.changed_mask(st, pre)
    seg_st = DET.segment_close(_mask_top_red(st))["bricks"]
    app, fa, fa2 = cl.pick_changed(seg_st, ch)
    van, fv, _ = cl.pick_changed(DET.segment_close(_mask_top_red(pre))["bricks"], ch)
    assert app is not None and abs(app["centre_px"][0] - 495) < 15 and abs(app["centre_px"][1] - 365) < 15, app
    assert van is not None and abs(van["centre_px"][0] - 350) < 15 and abs(van["centre_px"][1] - 341) < 15, van
    assert fa > 0.6 and fa2 < 0.4 and fv > 0.8, (fa, fa2, fv)
    old, _ = cl.nearest(seg_st, 380, 262, 170)
    assert old is not None and np.hypot(old["centre_px"][0] - app["centre_px"][0],
                                        old["centre_px"][1] - app["centre_px"][1]) > 60


def test_people_at_the_top_edge_are_not_a_brick():
    """A forearm moving behind the table enters the close view from the top: its change must be ignored."""
    before, after = _frame(), _frame()
    cv2.rectangle(before, (100, 0), (180, 120), (90, 120, 200), -1)       # skin-coloured arm from the top edge
    cv2.rectangle(after, (120, 0), (200, 130), (90, 120, 200), -1)        # it moved
    arm = {"bbox": [120, 40, 80, 90], "centre_px": [160.0, 85.0]}
    rec, frac, _ = cl.pick_changed([arm], cl.drop_top_edge(cl.changed_mask(after, before)))
    assert rec is None and frac < 0.05, frac


def test_colour_compatibility():
    assert cl.colour_ok("cream", "white") and cl.colour_ok("pink", "red") and not cl.colour_ok("orange", "white")
    assert cl.colour_ok("blue", None) and cl.colour_ok(None, "white") and not cl.colour_ok("green", "blue")


def test_moving_forearm_rejected_on_recorded_frames():
    """Run any_2127 ep7: between the first close frame and the station frame a person's forearm moved at the top-left;
    with top-edge changes dropped and the white target's colour required, no candidate comes from that region."""
    fx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "closeloc")
    pre = cv2.imread(os.path.join(fx, "ep7_pre_close.jpg"))
    st = cv2.imread(os.path.join(fx, "ep7_station_close.jpg"))
    if pre is None or st is None:
        import conftest as C
        if not C.IMAGES_PRESENT:
            raise C.Skip("fixtures/closeloc: recorded frames are local data, not in this checkout")
        raise AssertionError("fixture frames missing")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    "anomaly", "det2_f2"))
    import inspector as DET  # noqa: E402
    ch = cl.drop_top_edge(cl.changed_mask(st, pre))
    seg = [r for r in DET.segment_close(_mask_top_red(st))["bricks"] if cl.colour_ok(r.get("colour"), "white")]
    app, fa, _ = cl.pick_changed(seg, ch)
    assert app is None or app["centre_px"][1] > 250, app      # never the top-left forearm (centre y ~ 91)
    near, _ = cl.nearest(seg, 512, 381, 90)
    assert near is not None and near.get("colour") in ("white", "cream"), near


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
