"""Safety fixes from the code review (F-01, F-06, F-37, F-47, F-48), tested offline with fakes.

Nothing here reaches the robot, a camera, a serial port or Studio: the motion layer is a fake player, every folder
is a temp tree, and time.sleep is patched where the code waits.
"""
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import time
import types
from unittest import mock

import cv2
import numpy as np

import conftest as C

sa, sl, hd = C.sa, C.sl, C.hd
CLEAR = np.full((480, 640, 3), 128, np.uint8)


@contextlib.contextmanager
def _tree():
    root = tempfile.mkdtemp(prefix="safety_")
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _motion_server():
    """Motion_server.py with its hardware layer (so101_motion) replaced by fakes: only the Server logic is tested."""
    fake = types.ModuleType("so101_motion")
    fake.plan_ramp = lambda a, b, speed, dt, max_accel=60.0: np.linspace(a, b, 20)
    fake.validate_target = lambda q: None
    for name in ("MotionPlayer", "claim_devices", "device_ids_for", "open_follower_held", "resolve_arm"):
        setattr(fake, name, None)
    with mock.patch.dict(sys.modules, {"so101_motion": fake}):
        spec = importlib.util.spec_from_file_location("motion_server_under_test",
                                                      os.path.join(C.APP_TOOLS, "motion_server.py"))
        ms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ms)
    return ms


class _Player:
    dt = 0.001

    def __init__(self, present):
        self.present = np.asarray(present, float)
        self.sent = []
        self.driver = types.SimpleNamespace(send_action=lambda a: self.sent.append(np.asarray(a, float)))

    def current(self):
        return self.present.copy()

    def hold(self):
        pass


def test_safety_interrupted_move_reanchors_target_at_present():
    """F-01: a move cut by an e-stop leaves target = present (arm joints), so the next grip/rel cannot finish it."""
    ms = _motion_server()
    with _tree() as root, mock.patch.object(ms, "ACTIVE", os.path.join(root, "motion_active")), \
            mock.patch.object(ms, "ESTOP", os.path.join(root, "estop")), mock.patch.object(ms, "D", root), \
            mock.patch("time.sleep"):
        present = [10.0, 20.0, 30.0, 40.0, 50.0, 0.0]  # gripper closed on a brick
        srv = ms.Server(_Player(present))
        calls = iter([False, False, False, True] + [True] * 100)  # e-stop after three setpoints
        with mock.patch.object(srv, "estop", side_effect=lambda: next(calls)):
            msg = srv.move([60.0, -20.0, 0.0, 10.0, 0.0, 0.0], speed=20, comp=False)
        assert srv.status == "estop", f"status {srv.status} after an interrupted move ({msg})"
        assert np.allclose(srv.target[:5], present[:5]), f"target {srv.target[:5]} was not re-anchored at {present[:5]}"
        assert srv.target[5] == srv.goal[5] == 0.0, "the held gripper target changed on the e-stop"


def test_safety_motion_server_refuses_to_move_without_a_live_watchdog():
    """No heartbeat, or one older than 3 s: estop() is true and a move is refused with the reason."""
    ms = _motion_server()
    with _tree() as root, mock.patch.object(ms, "ESTOP", os.path.join(root, "estop")), \
            mock.patch.object(ms, "HEARTBEAT", os.path.join(root, "hand_watch_alive")):
        srv = ms.Server(_Player([0.0] * 6))
        assert srv.move([1.0] * 5 + [25.0], speed=5, comp=False) == "refused: hand watchdog not running"
        with open(ms.HEARTBEAT, "w") as f:
            f.write(f"{time.time() - 10.0!r}\n")
        assert srv.estop() is True, "a 10 s old heartbeat let the server move"
        with open(ms.HEARTBEAT, "w") as f:
            f.write(f"{time.time()!r}\n")
        assert srv.estop() is False, "a fresh heartbeat and no estop file still read as estop"


def test_safety_grasp_descent_never_below_min_tip():
    """F-47: from any hover height, the two descent moves stop the jaw tips at least MIN_TIP_MM above the mat by FK,
    on the first try and on the lower retry."""
    for fk_tip, retry in ((52.0, False), (52.0, True), (30.0, False), (12.0, True), (60.0, True)):
        moves = []
        with _tree() as out:
            os.makedirs(os.path.join(out, "crops"))
            with mock.patch.object(sa.ga, "OUT", out, create=True), \
                    mock.patch.object(sa.kin, "fk", return_value=(np.array([0.0, 0.0, sa.kin.MAT_Z + fk_tip]), None)), \
                    mock.patch.object(sa.pa, "frame", return_value=CLEAR), \
                    mock.patch.object(sa.ga, "fresh", return_value=CLEAR), \
                    mock.patch.object(sa.pa, "move_rel", side_effect=lambda q, g, d, speed=None, **k: moves.append(d) or q), \
                    mock.patch.object(sa.pa, "blobs", return_value=[]), \
                    mock.patch.object(sa.pa, "nearest", return_value=None), \
                    mock.patch.object(sa.ga, "snap", return_value={"wrist": None}), \
                    mock.patch.object(sa.ga, "cmd", return_value={"status": "ok"}), \
                    mock.patch.object(sa.ga, "state", return_value={"present": [0.0] * 5 + [2.0]}), \
                    mock.patch.object(sa, "EP", {}), mock.patch("time.sleep"):
                sa.grasp([0.0] * 5, {"short": 100.0, "aspect": 1.0, "hue": 20}, retry=retry)
        desc = moves[:next((i for i, d in enumerate(moves) if d[2] > 0), len(moves))]  # up to the lift after the close
        end = fk_tip + sum(d[2] for d in desc)
        assert len(desc) <= 2 and all(d[2] <= 0 for d in desc), f"fk {fk_tip}: the grasp descent was {moves}"
        assert end >= sa.MIN_TIP_MM - 1e-6, (f"fk {fk_tip} retry {retry}: jaws end {end:.1f} mm above the mat "
                                             f"(minimum {sa.MIN_TIP_MM})")


def test_safety_go_scan_never_sweeps_after_a_failed_lift():
    """F-48: low pose + every lift IK failing -> ga.Stop, and no joint-space move to the scan pose is sent."""
    low = (np.array([0.0, 0.0, sa.kin.MAT_Z + 20.0]), None)
    with mock.patch.object(sa, "scan_cal", return_value={"q_scan": [0.0] * 5}), \
            mock.patch.object(sa.ga, "state", return_value={"present": [0.0] * 6}), \
            mock.patch.object(sa.kin, "fk", return_value=low), \
            mock.patch.object(sa.pa, "move_rel", return_value=None), \
            mock.patch.object(sa, "tmove") as tmove:
        try:
            sa.go_scan(False)
            raised = False
        except sa.ga.Stop:
            raised = True
    assert raised, "go_scan did not stop after the lift failed"
    tmove.assert_not_called()


def test_safety_stale_live_frame_is_not_clear():
    """F-37: a fresh heartbeat but a live frame older than FRAME_MAX_AGE (dead recorder) is not a clear workspace."""
    with _tree() as root, contextlib.redirect_stdout(io.StringIO()):
        run_dir = os.path.join(root, "run")
        os.makedirs(os.path.join(root, "logs", "monitor"))
        os.makedirs(run_dir)
        open(os.path.join(root, "logs", "monitor", "hand_watch.log"), "w").close()
        for view in ("top", "close"):
            cv2.imwrite(os.path.join(run_dir, f"live_{view}.jpg"), CLEAR)
        with open(os.path.join(run_dir, "hand_watch_alive"), "w") as f:
            f.write(f"{time.time()!r}\n")
        with mock.patch.object(sl, "RUNTIME_DIR", run_dir), mock.patch.object(sl.ga, "RUN_ROOT", run_dir), \
                mock.patch.object(sl.ga, "LOG_ROOT", os.path.join(root, "logs")), \
                mock.patch.object(sl, "HAND_LOG", os.path.join(root, "logs", "monitor", "hand_watch.log")), \
                mock.patch.object(sl, "RUN", root), mock.patch.object(sl, "WATCHDOG_DOWN", False), \
                mock.patch.object(hd, "HEARTBEAT", os.path.join(run_dir, "hand_watch_alive")):
            assert sl.hands_seen(2) is False, "fresh frames and heartbeat: the clear workspace was not read as clear"
            old = time.time() - 5.0
            os.utime(os.path.join(run_dir, "live_top.jpg"), (old, old))
            assert sl.hands_seen(2) is True, "a 5 s old top frame was read as a clear workspace"


def test_safety_watchdog_heartbeat_only_with_live_frames_and_survives_errors():
    """F-37/F-06: hand_watch writes no heartbeat while a frame is stale, and an exception in a pass does not end it."""
    import hand_watch as hw
    with _tree() as root:
        run_dir = os.path.join(root, "run")
        os.makedirs(run_dir)
        os.makedirs(os.path.join(root, "logs", "monitor"))
        hb = os.path.join(run_dir, "hand_watch_alive")
        for view in ("top", "close"):
            cv2.imwrite(os.path.join(run_dir, f"live_{view}.jpg"), CLEAR)
        old = time.time() - 5.0
        os.utime(os.path.join(run_dir, "live_close.jpg"), (old, old))
        passes = []

        seen = {}

        def stop_after(_s):
            passes.append(1)
            if len(passes) == 2:  # pass 2 ran with the stale close frame; then make it live again
                seen["heartbeat_while_stale"] = os.path.exists(hb)
                os.utime(os.path.join(run_dir, "live_close.jpg"), None)
            if len(passes) >= 4:
                raise KeyboardInterrupt

        boom = iter([RuntimeError("cv2 failed"), None, None, None])

        def watch_once(*a):
            e = next(boom)
            if e:
                raise e

        with mock.patch.object(hw, "RUN_ROOT", run_dir), mock.patch.object(hw, "LOG_ROOT", os.path.join(root, "logs")), \
                mock.patch.object(hd, "HEARTBEAT", hb), mock.patch.object(hw, "watch_once", side_effect=watch_once), \
                mock.patch("time.sleep", side_effect=stop_after):
            try:
                hw.main()
            except KeyboardInterrupt:
                pass
        assert len(passes) == 4, f"the watchdog stopped after {len(passes)} passes (an exception must not end it)"
        assert seen.get("heartbeat_while_stale") is False, "a heartbeat was written while the close frame was stale"
        assert hd.heartbeat_fresh(hb), "no heartbeat after the frames became live again"
        with open(os.path.join(root, "logs", "monitor", "hand_watch.log")) as f:
            assert "WATCHDOG ERROR" in f.read(), "the failed pass was not logged"


def test_safety_far_band_needs_the_top_camera():
    """A close-view skin blob touching only the top edge and inside the top 30 % is someone behind the table; one that
    reaches deeper, or touches a side, still counts on its own."""
    import handdet
    assert handdet.far_band_only({"bbox": [100, 0, 120, 90], "edge": "T"}, 480)
    assert handdet.far_band_only({"bbox": [500, 0, 140, 120], "edge": "TR"}, 480)
    assert not handdet.far_band_only({"bbox": [100, 0, 120, 200], "edge": "T"}, 480)     # reaches onto the mat
    assert not handdet.far_band_only({"bbox": [0, 60, 120, 60], "edge": "L"}, 480)       # from the side
    assert not handdet.far_band_only({"bbox": None, "edge": "T"}, 480)


def test_safety_hand_refit_never_misses_a_labelled_hand():
    """fit_handdet: the refitted area rules catch every labelled hand (edge and inside), with the margin below the
    smallest one; a frame with only a small skin-coloured speck stays clear."""
    import csv
    import tempfile
    import fit_handdet as fh
    skin = (110, 140, 200)  # BGR inside the skin box (hue 10, sat 115, Cr 160, Cb 103)
    with tempfile.TemporaryDirectory() as d:
        rows = []
        for i, (box, label) in enumerate((((175, 0, 260, 60), "hand"), ((400, 0, 470, 40), "hand"),
                                          ((300, 150, 360, 210), "hand"), ((330, 200, 340, 210), "none"))):
            img = np.zeros((480, 640, 3), np.uint8)
            cv2.rectangle(img, box[:2], box[2:], skin, -1)
            cv2.imwrite(os.path.join(d, f"f{i}.png"), img)
            rows.append({"path": f"f{i}.png", "view": "top", "label": label})
        tsv = os.path.join(d, "labels.tsv")
        with open(tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "view", "label"], delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        data = fh.load(tsv)
        alone, edge = fh.fit(data, 0.8)
        err = fh.errors(data, alone, edge)
        assert not err["missed_hands"] and not err["false_stops"], (alone, edge, err)
        hands_edge = min(r["f"]["area"] for r in data if r["hand"] and r["f"]["edge"])
        assert edge["top"] == int(0.8 * hands_edge), (edge, hands_edge)
