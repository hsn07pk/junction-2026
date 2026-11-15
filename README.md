# junction-2026: SO-101 DUPLO quality sorter

An SO-101 arm picks DUPLO bricks from anywhere on the mat, inspects every face and places each brick on a plate:

- **Blue plate**: GOOD, a 2x2 brick whose every inspected face is clean.
- **Green plate**: DEFECT (tape or any other foreign mark on any face, or not 2x2), and every brick the inspection
  could not clear (OD-10: uncertain goes green).

Nothing in the code knows a brick colour or a tape colour. Thresholds are statistics of the images themselves or
values fitted from labelled calibration runs on the cell, each with how it was obtained.

## Hardware

- SO-101 follower arm (Feetech STS3215 servos) on a USB serial adapter.
- Wrist camera on the gripper (USB, 640x480, MJPEG).
- Two Intel RealSense D415 cameras: a top view and a low close view of the work area.
- Intel Core Ultra laptop.

## How a cycle works

1. **Scan**: from the scan pose the wrist camera finds the bricks; the plates are found in the same frame by colour and
   shape (a round region of plate colour, centre and radius on the mat), so bricks on or against a plate are skipped
   and placement goes to wherever the plates are.
2. **Approach**: the scan map gives the first estimate, differential IK moves to a hover pose, and wrist-camera visual
   servoing centres the brick between the jaws and aligns the jaws with it.
3. **Survey in place**: the brick is measured from the hover (box model fitted to its silhouette: footprint in whole
   studs, position, yaw). When nothing else stands close enough to hide a face, the wrist camera photographs the top
   and all four sides from planned poses. Otherwise the brick is put down alone at a free inspection spot, surveyed
   there and picked up again (straight from the measured pose).
4. **Defect test** (`tools/defects.py`): each view is registered to the brick model (silhouette, then edges), every
   face is resampled onto its own surface grid, and squares of the smallest defect size are tested for colour
   (chromaticity against the brick's body colour) and lightness (against a shading plane). A square counts only when
   every view that sees it agrees, so highlights, shadows and registration slips are not defects. The threshold
   follows from the number of squares tested (a contrario); the misfit of clean plastic is fitted on clean bricks only.
5. **Underside** (`tools/underside.py`, off until calibrated): the held brick's bottom is turned toward the close
   camera; tape over the opening changes the cavity's brightness relative to the rim and its colour.
6. **Decide and place**: blue only for a measured 2x2 whose every required face was tested clean; the brick is carried
   by IK to a free drop point on the plate.
7. **Safety**: a hand watchdog reads the cameras and stops the arm when a hand enters the cell; the motion server
   refuses to move without its heartbeat. The jaws are checked at start-up and again when the scan frame suggests
   they moved or a grasp closed on nothing.

## Calibration

The cell is calibrated by running it: see [docs/CALIBRATION.md](docs/CALIBRATION.md). In short: calibration runs of a
few labelled bricks (`--calibrate ID --truth GOOD|DEFECT`), then `calib_wristcam.py` (camera model from the surveys),
`plan_survey.py` (view library), `calibrate_defects.py` (defect model, checked brick by brick on held-out bricks).

## Layout

| Path | What |
|---|---|
| `tools/start_all.sh`, `tools/stop_all.sh`, `tools/park_release.sh` | Start the stack; stop it (arm folded, torque off, cameras freed) |
| `tools/motion_server.py` | The only process that talks to the arm: compensated joint moves, joint logging, e-stop hold |
| `tools/sort_any.py` | The sort loop: scan, approach, survey, decide, place, recovery, calibration runs |
| `tools/survey.py`, `tools/scene.py` | Survey planning, brick measurement, silhouettes; plates, free spots, isolation |
| `tools/defects.py`, `tools/calibrate_defects.py` | Surface defect test and its calibration from labelled runs |
| `tools/calib_wristcam.py`, `eyeinhand/wristcam.py` | Wrist camera model and its self-calibration from surveys |
| `tools/closecam.py`, `tools/underside.py` | Close camera model from the jaws; underside inspection |
| `tools/pick_any.py`, `tools/kin.py`, `pick_v2/kinematics.py` | Wrist-camera tracking and servoing; URDF kinematics and IK |
| `tools/handdet.py`, `tools/hand_watch.py`, `tools/fit_handdet.py` | Hand watchdog and its refit from labelled frames |
| `tools/jaw_check.py` | Jaw check and static jaw mask |
| `config/` | Brick spec, inspection settings, cell calibration, hand model, fitted models, survey plan |
| `tools/tests/` | Offline tests, including a simulated wrist camera |

## Running

```
bash tools/start_all.sh --run <name> --n 150     # start everything, run 150 cycles
bash tools/start_all.sh --status                 # what is running
bash tools/stop_all.sh                           # finish the cycle, fold the arm, stop everything
python -I -B tools/tests/run_tests.py            # offline tests (data environment)
```

`tools/paths.py` derives the repository and the workspace from the checkout. The workspace (the parent folder, or
`JUNCTION_ROOT`) holds `envs/`, `run/` and `logs/`. The URDF is the SO-101 description from
edge-developer-kit-reference-scripts; `tools/kin.py` looks for it under `repos/` in the workspace. Tests that replay
recorded camera frames skip when the frames are absent; the frames are not published.

## Known limits

- A black brick on the black mat may not separate from it: the survey cannot measure it, so it goes green.
- The underside test needs the close camera model and its own calibration before it can be switched on.
- A face seen in only one view cannot tell a shadow falling on it from a dark mark; the survey plans views that keep
  the arm's shadow off the brick, and calibration runs show whether it happens on the cell.
