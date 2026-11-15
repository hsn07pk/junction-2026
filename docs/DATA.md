# Data recorded by the sorter

## Per run and episode

`logs/sort/<run>/`:

- `cycles.jsonl`: one line per episode: task text, plate, label, decision confidence and reasons, shape, success,
  failure, timings.
- `loop.log`: the loop's log.
- `ep_NNNN/episode.json`: everything about one episode: the scan target, the approach, the grasp, the surveys (the
  measured brick pose and footprint, the views with their joints, each view's registration, per-face test values and
  verdicts, the body colour line), the underside record when on, the decision and the placement.
- `ep_NNNN/survey/*.jpg`: the survey frames exactly as they were judged (JPEG written once, then judged).
- `ep_NNNN/joints.csv`: joint positions at 30 Hz during the episode (motion server).
- `ep_NNNN/*.jpg`, `crops/`: phase frames of the three cameras.

Continuous recordings: `logs/monitor/` (top and close cameras, MKV) and the wrist MJPEG segments, with absolute
timestamps, so any episode can be cut from them.

## Labels

`logs/sort/labels.json` is written by calibration runs only:

```
{"bricks": {"<brick id>": "GOOD" | "DEFECT"}, "episodes": {"<run>/ep_0001": "<brick id>"}, "source": "..."}
```

It is the only place a brick's truth is stored; the sorting loop never reads it.

## Data card

- Content: an SO-101 arm picking DUPLO bricks on a black mat and placing them on two plates; joint trajectories;
  wrist, top and close camera frames; per-episode decisions with their evidence.
- Task text per episode: "move the <colour> brick to the <blue|green> plate" (colour = the tracker's colour class).
- People: the cameras can see hands entering the cell (the watchdog stops the arm then). Recordings with people are not
  published; the public repository holds code and configuration only.
- Known biases: one mat, one lighting setup, a handful of brick colours and tape types; decisions on uncertain bricks
  are green by rule, so the label distribution is not the defect rate.
