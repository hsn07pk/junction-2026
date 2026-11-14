"""P2-C-LOOPFIX tests: the heartbeat gate on the loop's clear check, the watchdog's heartbeat, and the give-up path.

Offline: every path the code touches is pointed at a temp folder, time.sleep is patched where the code waits, and the
motion layer (grasp_auto.cmd and state) is mocked. Nothing here reaches the robot, a camera, a serial port or Studio.
"""
import contextlib
import io
import os
import shutil
import tempfile
import time
from unittest import mock

import cv2
import numpy as np

import conftest as C
import hand_watch as hw

sa, sl, hd = C.sa, C.sl, C.hd
CLEAR = np.full((480, 640, 3), 128, np.uint8)  # flat grey: no skin tone, no blob


class _Halt(Exception):
    """Ends an endless wait or loop under test; raised from a patched time.sleep."""


@contextlib.contextmanager
def _tree():
    root = tempfile.mkdtemp(prefix="loopfix_")
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _png(path, img):
    ok, buf = cv2.imencode(".png", img)  # lossless; cv2.imread detects the format from the content, not the name
    assert ok
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def _frames(run_dir, hand):
    """Live top and close frames: flat grey, or a skin-toned block in the top view (inside the detector's ROI)."""
    top = CLEAR.copy()
    if hand:
        top[100:160, 300:360] = (100, 140, 200)  # BGR of HSV (12, 127, 200): inside the skin rule
    _png(os.path.join(run_dir, "live_top.jpg"), top)
    _png(os.path.join(run_dir, "live_close.jpg"), CLEAR)


def _heartbeat(folder, t):
    with open(os.path.join(folder, "hand_watch_alive"), "w") as f:
        f.write(f"{t!r}\n")


def _detector_hand(run_dir):
    """The detector's verdict on the live frames, with the gripper open (holding=False), as the clear check uses it."""
    for view in ("top", "close"):
        img = cv2.imread(os.path.join(run_dir, f"live_{view}.jpg"))
        if img is not None and hd.is_hand(img, view, False)[0]:
            return True
    return False


@contextlib.contextmanager
def _loop_paths(root, heartbeat_age):
    """Point the loop's clear check at root/run (live frames, flags, heartbeat) and root/logs/monitor (hand log); the
    loop log goes to root. heartbeat_age: None leaves no heartbeat, else the heartbeat's age in seconds."""
    run_dir = os.path.join(root, "run")
    hand_log = os.path.join(root, "logs", "monitor", "hand_watch.log")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(hand_log), exist_ok=True)
    open(hand_log, "w").close()
    if os.path.exists(os.path.join(run_dir, "hand_watch_alive")):
        os.remove(os.path.join(run_dir, "hand_watch_alive"))
    if heartbeat_age is not None:
        _heartbeat(run_dir, time.time() - heartbeat_age)
    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch.object(sl, "RUNTIME_DIR", run_dir))
        # main() and the stop/pause checks resolve their folders from grasp_auto's roots: keep them in the temp tree
        st.enter_context(mock.patch.object(sl.ga, "RUN_ROOT", run_dir))
        st.enter_context(mock.patch.object(sl.ga, "LOG_ROOT", os.path.join(root, "logs")))
        st.enter_context(mock.patch.object(sl, "HAND_LOG", hand_log))
        st.enter_context(mock.patch.object(sl, "RUN", root))
        st.enter_context(mock.patch.object(sl, "WATCHDOG_DOWN", False))
        st.enter_context(mock.patch.object(hd, "HEARTBEAT", os.path.join(run_dir, "hand_watch_alive")))
        yield run_dir


def test_loopfix_heartbeat_file_is_atomic_and_fresh_only_within_3s():
    with _tree() as root:
        path = os.path.join(root, "hand_watch_alive")
        hd.write_heartbeat(path)
        assert not os.path.exists(path + ".tmp"), "temp file left behind by the atomic write"
        assert hd.heartbeat_fresh(path) is True, "a heartbeat written just now is not fresh"
        with open(path) as f:
            assert abs(float(f.read()) - time.time()) < 1.0, "heartbeat is not the current epoch time"
        for label, t in (("3.5 s old", time.time() - 3.5), ("60 s in the future", time.time() + 60)):
            _heartbeat(root, t)
            assert hd.heartbeat_fresh(path) is False, f"{label} heartbeat counted as fresh"
        with open(path, "w") as f:
            f.write("not a number\n")
        assert hd.heartbeat_fresh(path) is False, "unreadable heartbeat counted as fresh"
        assert hd.heartbeat_fresh(os.path.join(root, "missing")) is False, "missing heartbeat counted as fresh"


def test_loopfix_fresh_heartbeat_clear_check_follows_the_hand_detector():
    with _tree() as root, _loop_paths(root, heartbeat_age=0.1) as run_dir, \
            contextlib.redirect_stdout(io.StringIO()):
        for hand in (False, True):
            _frames(run_dir, hand)
            verdict = _detector_hand(run_dir)
            assert verdict is hand, f"fixture frames not as labelled: hand={hand}, detector={verdict}"
            got = sl.hands_seen(2)
            assert got is verdict, f"clear check said {got} but the detector says {verdict} (hand={hand})"


def test_loopfix_stale_or_missing_heartbeat_is_not_clear():
    with _tree() as root, contextlib.redirect_stdout(io.StringIO()):
        for age, label in ((10.0, "stale (10 s old)"), (None, "missing")):
            with _loop_paths(root, heartbeat_age=age) as run_dir:
                _frames(run_dir, hand=False)  # the detector sees a clear workspace in both cases
                assert _detector_hand(run_dir) is False, "fixture frames are not clear to the detector"
                assert sl.hands_seen(2) is True, f"{label} heartbeat: the workspace was read as clear"


def test_loopfix_hand_watchdog_down_is_logged_once_per_stale_period():
    with _tree() as root, _loop_paths(root, heartbeat_age=None) as run_dir, \
            contextlib.redirect_stdout(io.StringIO()):
        _frames(run_dir, hand=False)
        for i in range(5):
            assert sl.hands_seen(2) is True, f"call {i + 1} with no heartbeat read the workspace as clear"
        _heartbeat(run_dir, time.time())
        assert sl.hands_seen(2) is False, "a fresh heartbeat and a clear view were not read as clear"
        _heartbeat(run_dir, time.time() - 10.0)
        assert sl.hands_seen(2) is True, "a stale heartbeat was read as clear"
        with open(os.path.join(root, "loop.log")) as f:
            n = sum("hand watchdog not running" in line for line in f)
    assert n == 2, f"'hand watchdog not running' logged {n} times; expected once per stale period (2 periods)"


def test_loopfix_stale_heartbeat_keeps_the_estop_and_wait_clear_waits():
    with _tree() as root, _loop_paths(root, heartbeat_age=10.0) as run_dir, \
            contextlib.redirect_stdout(io.StringIO()):
        _frames(run_dir, hand=False)
        estop = os.path.join(run_dir, "estop")
        with open(estop, "w") as f:
            f.write("test\n")
        halted = False
        with mock.patch("time.sleep", side_effect=_Halt):
            try:
                sl.wait_clear("test")
            except _Halt:
                halted = True
        assert halted, "wait_clear returned while the heartbeat was stale"
        assert os.path.exists(estop), "estop was cleared while the watchdog was down"
        _heartbeat(run_dir, time.time())
        sl.wait_clear("test")
        assert not os.path.exists(estop), "fresh heartbeat and a clear view: the estop should clear"


def test_loopfix_wait_for_bricks_does_not_return_until_the_heartbeat_is_fresh():
    cand = {"centre": [300.0, 200.0], "xy": [10.0, 20.0], "d_mm": [5.0, 6.0], "colour": "red", "border": False}
    with _tree() as root, _loop_paths(root, heartbeat_age=10.0) as run_dir, \
            mock.patch.object(sa, "R", root), \
            mock.patch.object(sa, "scan_cal", return_value={"q_scan": [0.0] * 5}), \
            mock.patch.object(sa, "candidates", return_value=[cand]) as cands, \
            mock.patch.object(sa.pa, "frame", return_value=CLEAR), \
            contextlib.redirect_stdout(io.StringIO()):
        _frames(run_dir, hand=False)
        halted = False
        with mock.patch("time.sleep", side_effect=_Halt):
            try:
                sa.wait_for_bricks()
            except _Halt:
                halted = True
        assert halted, "wait_for_bricks returned with a stale heartbeat: the arm would move"
        assert cands.call_count == 0, "wait_for_bricks scanned for bricks while the heartbeat was stale"
        _heartbeat(run_dir, time.time())
        with mock.patch("time.sleep", side_effect=lambda _s: None):
            got, _cands, _waited = sa.wait_for_bricks()
        assert got is cand, "fresh heartbeat and a stable brick: wait_for_bricks should return that brick"


def test_loopfix_hand_watch_writes_the_heartbeat_on_every_pass():
    with _tree() as root:
        os.makedirs(os.path.join(root, "run"))
        os.makedirs(os.path.join(root, "logs", "monitor"))
        hb = os.path.join(root, "run", "hand_watch_alive")
        passes = []

        def stop_after_three(_seconds):
            passes.append(1)
            if len(passes) >= 3:
                raise _Halt()

        for view in ("top", "close"):  # live frames: the heartbeat is written only while they are fresh
            cv2.imwrite(os.path.join(root, "run", f"live_{view}.jpg"), CLEAR)
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(hw, "R", root), \
                mock.patch.object(hw, "RUN_ROOT", os.path.join(root, "run")), \
                mock.patch.object(hw, "LOG_ROOT", os.path.join(root, "logs")), \
                mock.patch.object(hw, "STATE", os.path.join(root, "run", "mserver", "state.json")), \
                mock.patch.object(hd, "HEARTBEAT", hb), \
                mock.patch.object(hd, "write_heartbeat", wraps=hd.write_heartbeat) as writes, \
                mock.patch("time.sleep", side_effect=stop_after_three):
            try:
                hw.main()
            except _Halt:
                pass
        assert len(passes) == 3, f"expected 3 passes, ran {len(passes)}"
        assert writes.call_count >= 3, f"{writes.call_count} heartbeat writes in 3 passes"
        assert hd.heartbeat_fresh(hb) is True, "the heartbeat written by hand_watch.main() is not fresh"


def test_loopfix_recover_gives_up_after_40_attempts_and_writes_needs_operator():
    with _tree() as root, \
            mock.patch.object(sa, "RUN", root), \
            mock.patch.object(sa, "EP", {"episode": 7, "failure": "stop: servo lost brick"}), \
            mock.patch.object(sa.ga, "cmd", side_effect=sa.ga.Stop("motion server timeout")) as cmd, \
            mock.patch.object(sa.ga, "state", side_effect=sa.ga.Stop("motion server timeout")), \
            mock.patch.object(sa.sl, "wait_clear", side_effect=sa.ga.Stop("motion server timeout")), \
            mock.patch("time.sleep"), \
            contextlib.redirect_stdout(io.StringIO()):
        ok = sa.recover()
        assert ok is False, "recover() returned True although every attempt was interrupted"
        assert cmd.call_count == 40, f"{cmd.call_count} attempts, expected 40"
        marker = os.path.join(root, "needs_operator")
        assert os.path.exists(marker), "recover() gave up but wrote no needs_operator marker"
        with open(marker) as f:
            text = f.read()
        wanted = ("utc: 20", "episode: 7", "last failure: stop: servo lost brick", "recovery gave up after 40 attempts")
        for want in wanted:
            assert want in text, f"needs_operator lacks {want!r}:\n{text}"
        with open(os.path.join(root, "loop.log")) as f:
            assert "needs operator: recovery gave up after 40 attempts" in f.read(), "the give-up was not logged as needs operator"


def _argv(run):
    return ["sort_any.py", "--run", run, "--n", "3"]


def test_loopfix_main_does_not_start_the_cycle_loop_when_startup_recover_gives_up():
    holding_nothing = {"target": [0.0] * 5 + [10.0], "present": [0.0] * 5 + [25.0]}
    with _tree() as root, _loop_paths(root, heartbeat_age=0.1) as run_dir, \
            mock.patch.object(sa, "R", root), \
            mock.patch.object(sa, "RUN", root), \
            mock.patch.object(sa, "EP", {}), \
            mock.patch.object(sa.ga, "state", return_value=holding_nothing), \
            mock.patch.object(sa, "go_scan", side_effect=sa.ga.Stop("ik fail descending to hover")), \
            mock.patch.object(sa, "recover", return_value=False) as recover, \
            mock.patch.object(sa, "run_one", return_value=None) as run_one, \
            mock.patch("sys.argv", _argv("t_startup")), \
            contextlib.redirect_stdout(io.StringIO()):
        _frames(run_dir, hand=False)
        sa.main()
    assert recover.call_count == 1, f"recover() called {recover.call_count} times"
    run_one.assert_not_called()


def test_loopfix_main_does_not_start_the_cycle_loop_when_carryover_recover_gives_up():
    holding_brick = {"target": [0.0] * 6, "present": [0.0] * 5 + [3.0]}
    with _tree() as root, _loop_paths(root, heartbeat_age=0.1) as run_dir, \
            mock.patch.object(sa, "R", root), \
            mock.patch.object(sa, "RUN", root), \
            mock.patch.object(sa, "EP", {}), \
            mock.patch.object(sa.ga, "OUT", root, create=True), \
            mock.patch.object(sa.ga, "state", return_value=holding_brick), \
            mock.patch.object(sa, "finish_held", side_effect=sa.ga.Stop("grip slipped")), \
            mock.patch.object(sa, "recover", return_value=False) as recover, \
            mock.patch.object(sa, "run_one", return_value=None) as run_one, \
            mock.patch("sys.argv", _argv("t_carry")), \
            contextlib.redirect_stdout(io.StringIO()):
        _frames(run_dir, hand=False)
        for name in ("top_0001.mkv", "close_0001.mkv"):  # save_ep() links the newest top and close recordings
            open(os.path.join(root, "logs", "monitor", name), "w").close()
        sa.main()
        assert recover.call_count == 1, f"recover() called {recover.call_count} times"
        run_one.assert_not_called()
        episode = os.path.join(root, "logs", "sort", "t_carry", "ep_0000", "episode.json")
        assert os.path.exists(episode), "the carry-over episode was not saved before main() stopped"


def test_decide_follows_od10_failsafe():
    """OD-10: blue only for a measured 2x2 with a CONFIDENT_GOOD view; every UNCERTAIN case goes to green."""
    import sort_loop as sl
    good = {"verdict": "CONFIDENT_GOOD", "tape_score": 0.01}
    unc = {"verdict": "UNCERTAIN"}
    assert sl.decide(good, ("2x2", "a"), [])[:2] == ("blue", "GOOD")
    assert sl.decide(unc, ("2x2", "a"), [good])[:3] == ("blue", "GOOD", "held_only")
    assert sl.decide(good, ("not_2x2", "aspect 2.0"), [])[:2] == ("green", "DEFECT")
    assert sl.decide({"verdict": "CONFIDENT_DEFECT", "reasons": ["tape"]}, ("2x2", "a"), [])[:2] == ("green", "DEFECT")
    assert sl.decide(unc, ("2x2", "a"), [unc])[:3] == ("green", "UNCERTAIN", "no_evidence")
    assert sl.decide(good, ("unknown", ""), [])[:3] == ("green", "UNCERTAIN", "shape_unknown")
