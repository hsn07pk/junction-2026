# Calibrating the cell

Every value the sorter depends on is either measured by the cell itself or fitted from labelled runs. This is the
order to do it in after the robot, the mat, a camera or the gripper moved. Commands run from the workspace root with
the data environment (`envs/data-env/bin/python -I -B ...`); `start_all.sh` starts the stack.

## 0. Before anything moves

- `bash app/tools/start_all.sh --no-loop`: cameras, hand watchdog, motion server. Nothing moves without the watchdog.
- `config/cell_calibration.json` holds the measured cell geometry (mat height, hover height, grasp height, gripper
  readings, taught rest pose). Each entry says how it was measured; re-measure the ones whose part moved.

## 1. Jaws

The loop runs `jaw_check.py` at start-up (two wrist rolls at the scan pose): it writes the static jaw mask and the
servo target, and stops the loop with a message when a jaw is loose.

## 2. Plates

Start any run once: every scan records where the plates are (`run/plates_seen.json`). Nothing else to do; plates may be
moved between runs.

## 3. Calibration runs

Pick a few bricks whose truth you know, at least three clean and three with defects, of different colours. Give each
an id. For each brick, alone on the mat:

```
bash app/tools/start_all.sh --run cal_<id> --n 3 --calibrate <id> --truth GOOD      # or --truth DEFECT
```

Every cycle surveys the brick twice (where it lies, and set down turned at the inspection spot) and leaves it on the
mat; it never goes to a plate. The episodes are added to `logs/sort/labels.json` with the truth you gave. The truth
lives only in that file.

## 4. Wrist camera model

```
python -I -B app/tools/calib_wristcam.py --write
python -I -B app/tools/plan_survey.py
```

`calib_wristcam.py` fits the camera model to the corners of every surveyed brick (bundle adjustment with the arm's
joints) and writes it only when it predicts held-out surveys better than the current one; the previous model is kept
as `wrist_cam_model.prev.json`. `plan_survey.py` recomputes the survey views with the new model.

## 5. Defect model

```
python -I -B app/tools/calibrate_defects.py --write
```

It replays every labelled survey with the current camera model, fits the misfit of clean plastic on the clean bricks
only, holds out each physical brick in turn, and writes `config/defect_model.json` only when every held-out survey is
judged right and every defect is found. Without this file every face is UNCERTAIN and every brick goes green.

## 6. Underside (optional)

```
python -I -B app/tools/closecam.py --collect --fit --write    # nothing red on the mat; the arm moves through a grid
```

Then repeat step 3 (calibration runs record the bottom too), and `python -I -B app/tools/underside.py --fit --write`.
Set `"enabled": true` in `config/underside.json`; from then on the bottom is a required face.

## 7. Hand watchdog (when it stops for no hand, or should have stopped)

Add the frames to the labels TSV (path, view, label hand or none) and run
`python -I -B app/tools/fit_handdet.py LABELS.tsv --write`. The refit never misses a labelled hand.

## 8. Check

A short mixed run with known bricks: `logs/sort/<run>/cycles.jsonl` has every decision and its reasons, and each
episode's `episode.json` has the survey (views, registration, per-face test values).
