# P2-B-TESTS offline replay suite (builder E1)

Offline regression tests for the perception and kinematics code: pick_any, sort_any, handdet (the e-stop detector that sort_loop also uses) and kin. They replay recorded frames and logged numbers only. Nothing here touches the robot, a camera, a serial port or Studio.

## Run

    nice -n 19 env OMP_NUM_THREADS=4 envs/data-env/bin/python -I -B app/tools/tests/run_tests.py

About 2 s or less (1.7 s in the last run, E1 raw/L54_suite_run4.txt). It prints one line per test (PASS, FAIL, ERROR or SKIP, the time, then a note or the reason), then a TOTAL line. Exit code 1 if any test FAILs or ERRORs. pytest is not importable in envs/data-env, so run_tests.py is a plain runner. Nothing was installed.

## Files

- test_perception.py: 8 tests for pick_any, sort_any and handdet.
- test_kin.py: 2 tests for kin.
- conftest.py: loaders and the fixture tables HOVER, SERVO, SPLIT and CLOSE, each row naming its episode, frame or log line.
- run_tests.py: the runner.
- fixtures/: 84 data files (5 hover, 7 close, 17 servo, 11 split, 41 hand, scan_now.jpg, scan_calib.json, hand_labels.csv) and MANIFEST.tsv, one row per data file with source path, source time, expected values, how verified and sha256.

## What each test protects

- test_hover_blob_replay: hover wrist frames of episodes 0001, 0006, 0013, 0015 and 0017. find(blobs(img, min_area=1500), target(), 260, colour) must give a blob within 12 px of the recorded centre and within 15 percent of the recorded short side, with the recorded colour where one is recorded.
- test_servo_replay_seeded_at_logged_centre: 17 wrist frames at logged servo times (10 from 18:00:02Z to 18:15:02Z, 7 from 18:20:06Z to 18:20:25Z). Seeded at the logged centre, the tracker must stay within 15 px of it and within 15 percent of the logged short side, with the same colour class.
- test_neighbour_switch_1800Z (18:00Z neighbour switch): frames 18:00:02Z and 18:00:15Z. The tracker keeps the red brick, with no colour switch, within 30 px of the logged centre.
- test_split_brick_single_white_blob_1816Z (18:16Z split white brick): 11 frames from 18:16:10Z to 18:16:30Z, every 2 s (episode 16). blobs(min_area=1500) must return exactly one white blob per frame, with short between 60 and 140 px. The spec describes the live run as alternating between shorts of about 95 and 75; the fixture frames have one white blob each, with shorts from 76 to 127 px.
- test_scan_candidates_exclude_plates_1754Z (17:54Z pink and cream merge): scan_now.jpg. candidates() drops the green and blue plate blobs and returns only red and yellow candidates, as two blobs more than 30 px apart.
- test_close_pre_verdict_replay: seven pre-close frames (episodes 0001, 0006, 0008, 0013, 0014, 0015 and 0017). The verdict and the exact reasons list must match. The hue must be within 2 units where episode.json records one.
- test_handdet_runs_on_all_estop_frames (smoke test): handdet.is_hand runs on all 41 e-stop frames and returns a bool and an integer area. No label is asserted. This includes the 14 wrist frames from 18:03:08Z to 18:03:22Z (none at 18:03:20Z).
- test_hand_labels_if_present (18:03Z white brick, false hand stops): for the 20 labelled top and close rows (11 top, 9 close), is_hand with the row's holding flag must give True for "hand" and False for "nohand". Wrist rows are skipped: all 15 have holding=1, and the wrist view is gated off while holding. The four hand_partial rows are listed, not asserted. The two top frames with no label row are listed in the PASS note. So the 18:03Z wrist frames are only smoke-tested, not asserted.
- test_fk_scan_and_hover_heights: fk(q_scan) is more than 50 mm above fk(q_hover), and fk(q_hover) is within 5 mm of MAT_Z + HOVER_DZ (kin.py:24 and kin.py:25).
- test_ik_rel_round_trip_5x5_grid_60mm: round trip over a 5 x 5 grid of +/-60 mm at hover height, with error below 1.5 mm.

## Last run (E1 raw/L54_suite_run4.txt)

    TOTAL 10 PASS 8 FAIL 2 ERROR 0 SKIP 0 wall 1.7 s

- FAIL test_servo_replay_seeded_at_logged_centre: servo_180002_red.png (18:00:02Z). The centre is (413.4, 286.0) against the logged (422.7, 268.8), 19.6 px apart; short 123.8 against 112.6. The other 16 frames are within 1.1 px. Cause not established; see STATUS.md.
- FAIL test_ik_rel_round_trip_5x5_grid_60mm: ik_rel returns None for (+60, +30) and (+60, +60) mm from q_hover, also with a 5 mm and 0.2 rad tolerance. The other 23 points converge with error at most 0.28 mm.


## Notes

- hand_watch.py is never imported: it is a program with a main block (line 95), and sort_loop uses handdet directly. The suite imports pick_any, grasp_auto, sort_any, sort_loop, inspector, kin (which imports pick_v2.kinematics) and handdet read-only, and points sort_any.RUN and sort_loop.RUN at a temp folder.
- fixtures/hand_labels.csv is a manual-label file. The suite reads it and never writes it.
- Servo and split frames are ffmpeg extracts from data/wrist/raw segments. Method, offsets and checksums: MANIFEST.tsv and the E1 scratch tool extract_frames.py (not part of the suite).
- A changed expectation or a changed fixture needs a MANIFEST.tsv row with its reason. Do not widen a tolerance to clear a FAIL.
