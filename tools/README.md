# tools: the sorter's runtime, calibration and tests

## Runtime

| module | role |
|---|---|
| `sort_any.py` | The loop: scan, candidates, approach (servo), survey, grasp, decide, place; recovery, carry-over, calibration runs. CLI: `--run --n --start --speed --profile {safe,fast} --calibrate ID --truth GOOD,DEFECT` |
| `survey.py` | Survey views (library at any brick pose), brick measurement by box fitting, silhouettes, alignment |
| `scene.py` | Mat, plates (colour and shape), free spots, isolation of a brick on the scan frame |
| `defects.py` | Surface defect test from all survey views (registration, face textures, a-contrario windows) |
| `underside.py`, `closecam.py` | Underside inspection with the close camera; close camera model from the jaws |
| `pick_any.py` | Wrist-camera blobs, colour lock, servo, yaw alignment, relative moves |
| `sort_loop.py` | Shared helpers: workspace-clear rule, loop log, crops, the OD-10 decision (`decide_faces`) |
| `grasp_auto.py` | Motion-server client, live camera frames, snapshots, gripper settle, jaw mask and servo target |
| `kin.py`, `cell.py` | URDF kinematics: top-down IK, differential IK, IK with a pointing direction; cell calibration values |
| `handdet.py`, `hand_watch.py` | Hand detector (model in `config/hand_model.json`) and the watchdog process |
| `motion_server.py`, `mcmd.py`, `mseq.py` | The only process that talks to the arm; one command; a pose sequence |
| `jaw_check.py` | Jaw check and static jaw mask |
| `rest.py` | Fold to the taught rest pose (used by the stop scripts) |
| `start_all.sh`, `stop_all.sh`, `park_release.sh` | Start the stack; stop it; fold, release and free the cameras |
| `monitor_cams.sh`, `wrist_rec.sh` | Continuous recorders with absolute timestamps |

## Calibration (see ../docs/CALIBRATION.md)

| tool | writes |
|---|---|
| `calibrate_defects.py` | `config/defect_model.json` from labelled surveys (hold-out by physical brick) |
| `calib_wristcam.py` | `eyeinhand/wrist_cam_model.json` from surveys (bundle adjustment, k-fold check) |
| `plan_survey.py` | `config/survey_plan.json`, `config/survey_library.json` |
| `closecam.py` | `config/close_cam_model.json` from jaw pixels and FK |
| `underside.py --fit` | `config/underside_model.json` |
| `fit_handdet.py` | `config/hand_model.json` from labelled frames (never misses a labelled hand) |
| `jaw_check.py` | `jaw_calib.json`, `wrist_jaw_mask.png` |

## Tests

`python -I -B tests/run_tests.py`. `tests/sim_scene.py` renders the wrist camera's view (studs, tape, a lamp's
reflection, shadows, blur, exposure, noise, tone curve, JPEG) through the camera model, so the survey, the defect test,
its calibration and the camera self-calibration run end to end offline. Replays of recorded frames skip when the
frames are absent.

## Conventions

- Joints are normalised SO-101 units: body -100..100, gripper 0..100. Lengths are mm in the base frame.
- Every module that moves the arm goes through the motion server. Restart processes only by their pid files.
- No brick or tape colour appears in the code; measured and fitted values live in `config/` with how they were obtained.
