# app/tools — SO-101 DUPLO sorter runtime

Start here: `docs/PICK-ANYWHERE.md` covers how the loop works, how to run, stop and resume it, the safety rules
and the failure log.

## Runtime (used by the pick-anywhere loop)

| module | role |
|---|---|
| `sort_any.py` | **The loop.** Scan → candidates → approach (servo) → grasp → inspect → decide → place, with recording, recovery and carry-over. CLI: `--run --n --start --speed --profile {safe,fast}` |
| `pick_any.py` | Wrist-camera perception and motion helpers: colour-class `blobs`, `find` (colour lock), `servo`, `align_yaw`, `move_rel`, size→height |
| `sort_loop.py` | Shared helpers: `hands_seen`, `wait_clear`, `close_verdict`, `decide`, `save_crop`, `run_fast`. Its own `cycle`/`main` is the legacy spot-cell loop |
| `grasp_auto.py` | Motion-server client (`cmd`, `move`, `state`), camera frames (`fresh`, `snap`), `TARGET_PX`, static jaw mask |
| `kin.py` | URDF forward kinematics and differential IK (`ik_rel`), `MAT_Z`, wrist Jacobian `m_base` |
| `handdet.py` | Hand detector shared by the watchdog and the loop's clear check |
| `hand_watch.py` | Hand watchdog process (writes `run/estop`) |
| `motion_server.py` | The only process that talks to the follower: compensated moves, 30 Hz joint logging, e-stop hold |
| `mcmd.py`, `mseq.py` | Send one command, or a named pose sequence, to the motion server |
| `jaw_check.py` | Check-only jaw position test (run at start and every 5 cycles) |
| `monitor_cams.sh`, `wrist_rec.sh` | Continuous recorders: top/close MKV and wrist MJPEG segments, absolute timestamps |
| `arm_free_watch.sh` | Read-only: writes `run/arm_state` (FREE or BUSY) for the follower and RealSense nodes |

## Calibration and measurement tools

- `scan_calib.py`, `calib_M.py`, `calib_wrist.py`, `jaw_calib.py`: build the calibration JSONs described in
  `docs/PICK-ANYWHERE.md`.
- `rs_*.py`, `jprobe.py`, `servo_step.py`, `wrist_brick.py`: RealSense depth and jaw measurements (I-59).

## Review and offline tools

- `review_ep.py`: contact sheet per episode.
- `peek.py`: three-camera snapshot.
- `studio_sessions.py`: lists Studio runtime sessions.
- `../eyeinhand/`: wrist camera model fitted from recordings (offline; not used by the loop yet).
- `tests/`: replay regression tests and fixtures (`tests/fixtures/hand_labels.csv` holds the labelled e-stop
  images).
- `work/lint/` at the repo root: stdlib static checks (`undef.py`, `attrs.py`) and `smoke_offline.py`.

## Superseded (kept for reference)

- `motion_server_v1.py`.
- The spot-cell `cycle`/`main` in `sort_loop.py`.
- Backups of every edited file are in `run/bak_<module>_<UTC hhmm>.py`.

## Conventions

- Joints are normalized SO-101 units: body −100..100, gripper 0..100. Lengths are mm in the base frame.
- Every module that moves the arm goes through the motion server. Restart processes only by their pid files.
- Every change carries a UTC stamp and its reason in a comment (the incident it fixes).
