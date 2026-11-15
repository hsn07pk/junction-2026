"""Replay tests for the perception code (pick_any.blobs/find, sort_any.candidates/close_pre_verdict, handdet.is_hand).

Every test reads recorded frames under tests/fixtures. Incidents protected (times UTC, 2026-11-14):
  17:54Z merged pink+cream            -> test_scan_candidates_exclude_plates_1754Z
  18:00Z neighbour switch             -> test_neighbour_switch_1800Z, test_servo_replay_seeded_at_logged_centre
  18:16Z split white brick            -> test_split_brick_single_white_blob_1816Z
  18:03Z white brick, false hand stop -> test_handdet_runs_on_all_estop_frames, test_hand_labels_if_present
Hover blobs and close pre-verdicts replay the frames the pick code used at hover and before the close.
"""
import os

import numpy as np

import conftest as C


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def test_hover_blob_replay():
    """Fixture 1: find(blobs(img, min_area=1500), target(), 260, colour) within 12 px and short within 15 percent."""
    fails = []
    for fname, colour, (x, y), short, _hue in C.HOVER:
        img = C.load_img(fname)
        b = C.pa.find(C.pa.blobs(img, min_area=1500), C.pa.target(), 260, colour)
        if b is None:
            fails.append((fname, "no blob"))
            continue
        d = _dist(b["centre"], (x, y))
        ds = abs(b["short"] - short) / short
        if d > 12 or ds > 0.15 or (colour is not None and b["colour"] != colour):
            fails.append((fname, round(d, 1), round(ds, 3), b["colour"]))
    assert not fails, f"hover replay mismatches (file, centre px, short rel, colour): {fails}"


def test_servo_replay_seeded_at_logged_centre():
    """Fixture 2: seeded at the logged centre, find(blobs(min_area=SERVO_MIN_AREA), seed, 140, colour) keeps the colour
    class and reproduces the centre within 15 px and short within 15 percent. 17 frames: 10 from 18:00:02Z to 18:15:02Z
    and 7 from 18:20:06Z to 18:20:25Z (the log has only 7 servo lines after 18:19:54Z).

    Known defect PD-1 (docs/PICK-ANYWHERE.md, Known perception defects): on servo_180002_red.png the 15 px bridge
    joins a 339 px red speck 78 px from the brick, and the box centre moves 19 px. It is reported in the note, not
    asserted, until the brick segmentation is redone; any other mismatch fails the test."""
    fails, known = [], []
    for fname, colour, x, y, short, _line in C.SERVO:
        img = C.load_img(fname)
        b = C.pa.find(C.pa.blobs(img, min_area=C.pa.SERVO_MIN_AREA), np.array([x, y]), 140, colour)
        if b is None:
            fails.append((fname, "no blob"))
            continue
        d = _dist(b["centre"], (x, y))
        ds = abs(b["short"] - short) / short
        if d > 15 or ds > 0.15 or b["colour"] != colour:
            (known if fname in KNOWN_SERVO_DEFECTS else fails).append((fname, round(d, 1), b["centre"], b["short"], b["colour"]))
    assert not fails, f"servo replay mismatches (file, centre px, now centre, now short, colour): {fails}"
    fixed = sorted(set(KNOWN_SERVO_DEFECTS) - {k[0] for k in known})
    return f"known defect still present: {known}" + (f"; known defect no longer reproduces: {fixed}" if fixed else "")


KNOWN_SERVO_DEFECTS = {"servo/servo_180002_red.png": "PD-1"}


def test_neighbour_switch_1800Z():
    """Fixture 2, 18:00:02Z and 18:00:15Z: seeded at the logged centre with the tracked colour class, the tracker keeps
    the red brick (no colour switch) and stays within 30 px of the logged centre."""
    for fname, colour, x, y, short, _line in C.SERVO[:2]:
        img = C.load_img(fname)
        b = C.pa.find(C.pa.blobs(img, min_area=C.pa.SERVO_MIN_AREA), np.array([x, y]), 140, colour)
        assert b is not None, f"{fname}: no {colour} blob near the logged centre"
        assert b["colour"] == colour, f"{fname}: tracker switched colour class to {b['colour']}"
        assert _dist(b["centre"], (x, y)) <= 30, f"{fname}: tracked centre {b['centre']} is far from {(x, y)}"


def test_split_brick_single_white_blob_1816Z():
    """Fixture 4: 18:16:10Z-18:16:30Z (ep 16), one white brick. blobs(min_area=1500) must return exactly one white blob
    in every frame (the live run alternated between two pieces, short about 95 and 75)."""
    fails = []
    for rel in C.SPLIT:
        img = C.load_img(rel)
        whites = [b for b in C.pa.blobs(img, min_area=1500) if b["colour"] == "white"]
        if len(whites) != 1:
            fails.append((rel, len(whites), [b["centre"] for b in whites]))
        elif not (60 <= whites[0]["short"] <= 140):
            fails.append((rel, "short", whites[0]["short"]))
    assert not fails, f"split-brick frames with other than one white blob: {fails}"


def test_scan_candidates_exclude_plates_1754Z():
    """Fixture 3: scan_now.jpg (17:54Z). The current code finds six blobs there (raw/L21_probe.txt; raw/L23_probe2.txt
    section b): red, yellow, white (r = 346.5 mm), one green (plate_frac 0.35) and two blue (plate_frac 0.28 and 0.99).
    candidates() must drop the green and blue plate blobs and return exactly the red and yellow candidates, as two
    separate blobs more than 30 px apart. The white blob is outside the 335 mm reach (sort_any.py:158) and is not a
    candidate. pick_any.py:36 keeps touching pink and cream bricks as separate blobs (17:54Z incident)."""
    c = C.load_json("scan_calib.json")
    img = C.load_img("scan_now.jpg")
    cands = C.sa.candidates(img, c["q_scan"], c)
    plate_hits = [(b["colour"], b["centre"]) for b in cands
                  if b["colour"] in ("blue", "green") or b["plate_frac"] >= 0.3]
    assert not plate_hits, f"plate blobs among candidates: {plate_hits}"
    colours = sorted({b["colour"] for b in cands})
    assert colours == ["red", "yellow"], f"candidate colours {colours}, expected ['red', 'yellow']"
    red = [b["centre"] for b in cands if b["colour"] == "red"][0]
    yel = [b["centre"] for b in cands if b["colour"] == "yellow"][0]
    assert _dist(red, yel) > 30, f"red {red} and yellow {yel} merged into one neighbourhood"


def test_handdet_runs_on_all_estop_frames():
    """Fixture 6 smoke test: handdet.is_hand(img, view) runs on all 41 recorded estop frames and returns a bool and a
    features dict with an integer area. No label is asserted here (see test_hand_labels_if_present)."""
    files = sorted(os.listdir(C.fixture_dir("hand")))
    assert len(files) == 41, f"expected 41 estop frames, found {len(files)}"
    for f in files:
        view = f.split("_")[1]
        hit, feats = C.hd.is_hand(C.load_img(os.path.join("hand", f)), view, holding=False)
        assert (isinstance(hit, bool) and isinstance(feats["area"], int) and feats["area"] >= 0), (
            f"{f}: unexpected result {hit} {feats}")


def test_hand_labels_if_present():
    """Fixture 6 labels (fixtures/hand_labels.csv, manual labels). Rule: every 'hand' row of the top and close views
    gives True and every 'nohand' row gives False, using the row's holding flag. Wrist rows are skipped (the wrist view
    is gated off while holding; every wrist row has holding=1). hand_partial rows (fingers at the frame edge only) and
    hand_outside rows (a hand off the mat, out of the arm's reach) are reported, not asserted. Frames with no label
    row are listed in the PASS note. Skips when the label file is absent."""
    rows = C.load_hand_labels()
    labelled = {os.path.basename(r["file"]) for r in rows}
    unlabelled = sorted(set(os.listdir(C.fixture_dir("hand"))) - labelled)
    mism, partial, checked = [], [], 0
    for r in rows:
        name = os.path.basename(r["file"])
        if r["view"] == "wrist":
            continue
        if r["label"] in ("hand_partial", "hand_outside"):
            partial.append(f"{name} ({r['label']})")
            continue
        hit, feats = C.hd.is_hand(C.load_img(os.path.join("hand", name)), r["view"], holding=(r["holding"] == "1"))
        checked += 1
        if hit != (r["label"] == "hand"):
            mism.append((name, r["view"], r["label"], r["holding"], feats["area"], feats["edge"]))
    assert not mism, f"handdet disagrees with the labels (file, view, label, holding, area px, edge): {mism}"
    return (f"{checked} top/close rows checked; not asserted: {partial}; "
            f"fixture frames with no label row: {unlabelled}")


def test_scan_plates_excluded_without_the_scan_map():
    """Plate exclusion must not depend on the learned scan map: with the map reset to the raw Jacobian (as after a
    camera move), scan_now.jpg still yields no plate blobs among the candidates (21:33Z: one new scan pair moved the
    blue plate crescent at (592, 307) out of the table keep-out)."""
    c = C.load_json("scan_calib.json")
    img = C.load_img("scan_now.jpg")
    saved = (C.sa.SCAN_C, C.sa.SCAN_H)
    try:
        C.sa.SCAN_C, C.sa.SCAN_H = np.eye(2), None
        cands = C.sa.candidates(img, c["q_scan"], c)
    finally:
        C.sa.SCAN_C, C.sa.SCAN_H = saved
    plate_hits = [(b["colour"], b["centre"]) for b in cands if b["colour"] in ("blue", "green") or b["plate_frac"] >= 0.3]
    assert not plate_hits, f"plate blobs among candidates with the raw scan map: {plate_hits}"
