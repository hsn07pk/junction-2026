"""Shared setup for the offline replay tests (P2-B-TESTS, builder E1).

pytest is not importable in envs/data-env, so run_tests.py imports this module and the test modules directly.
Everything here reads recorded files under tests/fixtures only. The code under test is imported read-only; its
import-time reads are calibration files and mask PNGs. No test imports hand_watch.py, which is a program with its own
main block. Import-time device check of the modules used here: grep for serial, VideoCapture, realsense, /dev/, Popen and socket on
2026-11-14 18:32Z-18:33Z (E1 laptop-log L5, L6, L8, L9); handdet.py is not in those greps; its imports (cv2 and numpy only) and
device check are in E1 laptop-log L38.
"""
import csv
import importlib.util
import json
import os
import sys
import tempfile

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
# The tree under test: the tools folder that holds this tests folder (a copy under work/ tests the copy, not R/app).
APP_TOOLS = os.path.dirname(HERE)
DET2 = os.path.join(os.path.dirname(APP_TOOLS), "anomaly", "det2_f2")
for _p in (APP_TOOLS, DET2):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name, path):
    """Load one module from the tree under test and register it under its plain name. The app modules insert R/app
    into sys.path at import time, so an ordinary import would pick up R/app's copy of a file."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _name in ("rs_height", "kin", "handdet", "hand_watch", "grasp_auto", "pick_any", "sort_loop", "sort_any"):
    _load(_name, os.path.join(APP_TOOLS, _name + ".py"))
_load("inspector", os.path.join(DET2, "inspector.py"))

import pick_any as pa  # noqa: E402
import sort_any as sa  # noqa: E402
import sort_loop as sl  # noqa: E402
import kin  # noqa: E402
import handdet as hd  # noqa: E402

for _m in (pa, sa, sl, kin, hd, sys.modules["hand_watch"]):
    if os.path.dirname(_m.__file__) != APP_TOOLS:
        raise RuntimeError(f"{_m.__name__} was loaded from {_m.__file__}, not from {APP_TOOLS}")

RUN_DIR = tempfile.mkdtemp(prefix="p2b_tests_run_")  # sort_any.log writes RUN/loop.log; keep it out of the repo
sa.RUN = RUN_DIR
sl.RUN = RUN_DIR


class Skip(Exception):
    """Raised by a test that cannot run on this machine or without supplied labels."""


# Fixture 1: hover blobs. (file, colour or None, logged centre, logged short, logged hover hue)
# Source: logs/sort/any_1740/ep_NNNN/episode.json hover_blob, frame logs/sort/any_1740/ep_NNNN/crops/wrist_hover.png
HOVER = [
    ("hover/ep0001_wrist_hover.png", None, (400.1, 273.1), 107.7, 14),
    ("hover/ep0006_wrist_hover.png", None, (404.0, 265.0), 106.7, 18),
    ("hover/ep0013_wrist_hover.png", "red", (407.5, 268.0), 117.0, 177),
    ("hover/ep0015_wrist_hover.png", "red", (403.4, 268.4), 110.9, 176),
    ("hover/ep0017_wrist_hover.png", "white", (405.0, 270.5), 117.0, 17),
]

# Fixture 2: wrist frames at logged servo times (sort_any_any_1740.out). (file, colour, x, y, short, log line)
SERVO = [
    ("servo/servo_180002_red.png", "red", 422.7, 268.8, 112.6, 104),
    ("servo/servo_180015_red.png", "red", 407.5, 268.0, 117.0, 110),
    ("servo/servo_181348_white.png", "white", 397.8, 287.9, 109.2, 267),
    ("servo/servo_181352_white.png", "white", 408.8, 268.4, 113.6, 269),
    ("servo/servo_181356_white.png", "white", 404.5, 274.5, 107.0, 272),
    ("servo/servo_181451_red.png", "red", 420.2, 346.7, 108.8, 277),
    ("servo/servo_181454_red.png", "red", 435.9, 287.8, 116.0, 278),
    ("servo/servo_181456_red.png", "red", 408.6, 270.2, 113.3, 279),
    ("servo/servo_181500_red.png", "red", 365.4, 297.6, 109.2, 281),
    ("servo/servo_181502_red.png", "red", 403.4, 268.4, 110.9, 282),
    ("servo/servo_182006_white.png", "white", 466.0, 394.1, 71.4, 324),
    ("servo/servo_182010_white.png", "white", 488.8, 305.5, 92.6, 325),
    ("servo/servo_182013_white.png", "white", 406.5, 260.8, 121.2, 326),
    ("servo/servo_182017_white.png", "white", 402.2, 267.7, 120.1, 327),
    ("servo/servo_182019_white.png", "white", 406.9, 257.5, 113.8, 329),
    ("servo/servo_182021_white.png", "white", 390.5, 263.0, 118.0, 330),
    ("servo/servo_182025_white.png", "white", 405.0, 270.5, 117.0, 331),
]

# Fixture 4: split white brick, ep 16 window 18:16:10Z-18:16:30Z, one frame every 2 s
SPLIT = ["split/split_1816%02d.png" % s for s in range(10, 31, 2)]

# Fixture 5: close pre-verdicts. (episode, file, recorded verdict, recorded reasons, want_hue, recorded hue)
# want_hue = hover_blob hue when episode.json stores it; for ep 0008 and 0014 (no hover_blob) it is
# pre_close.hue + pre_close.hue_diff, which reproduces the recorded hue_diff exactly.
CLOSE = [
    ("0001", "close/ep0001_pre_close_full.jpg", "CONFIDENT_GOOD", ["tape_frac_low"], 14, None),
    ("0006", "close/ep0006_pre_close_full.jpg", "CONFIDENT_DEFECT", ["tape_frac_high", "merged_blob_white_plus_colour"], 18, 28),
    ("0008", "close/ep0008_pre_close_full.jpg", "UNCERTAIN", ["tape_frac_low", "close_brick_hue_mismatch_11"], 13, 2),
    ("0013", "close/ep0013_pre_close_full.jpg", "UNCERTAIN", ["merged_blob_two_hues"], 177, 5),
    ("0014", "close/ep0014_pre_close_full.jpg", "UNCERTAIN", ["tape_frac_low", "close_brick_hue_mismatch_82"], 82, 0),
    ("0015", "close/ep0015_pre_close_full.jpg", "CONFIDENT_GOOD", ["tape_frac_low"], 176, 178),
    ("0017", "close/ep0017_pre_close_full.jpg", "CONFIDENT_DEFECT", ["tape_frac_high", "merged_blob_two_hues"], 17, 18),
]


def _has_images(root):
    for _, _, names in os.walk(root):
        if any(n.lower().endswith((".png", ".jpg", ".jpeg")) for n in names):
            return True
    return False


IMAGES_PRESENT = _has_images(FIX)  # recorded camera frames are local data; the public repository has none


def fixture_dir(name):
    """A fixture image folder. Skip when this checkout has no recorded frames at all; a folder missing from a checkout
    that has frames is an error."""
    p = os.path.join(FIX, name)
    if not os.path.isdir(p):
        if not IMAGES_PRESENT:
            raise Skip(f"fixtures/{name}: recorded frames are local data, not in this checkout")
        raise FileNotFoundError(p)
    return p


def load_img(rel):
    img = cv2.imread(os.path.join(FIX, rel))
    if img is None:
        if not IMAGES_PRESENT:
            raise Skip(f"fixture image {rel}: recorded frames are local data, not in this checkout")
        raise FileNotFoundError(rel)
    return img


def load_json(rel):
    with open(os.path.join(FIX, rel), encoding="utf-8") as f:
        return json.load(f)


def load_hand_labels():
    """Rows of fixtures/hand_labels.csv: view, file (repo-relative source path), label, holding, note.
    Raises Skip when the file is absent. Rows are matched to the copies in fixtures/hand/ by basename."""
    path = os.path.join(FIX, "hand_labels.csv")
    if not os.path.exists(path):
        raise Skip("fixtures/hand_labels.csv not supplied (local labelled data)")
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
