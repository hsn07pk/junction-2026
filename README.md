# junction-2026: SO-101 DUPLO quality sorter

An SO-101 arm picks DUPLO bricks from anywhere on the mat, inspects each one and places it on a plate:

- **Blue plate**: GOOD, a 2x2 brick with no defect.
- **Green plate**: DEFECT (tape or any other foreign mark, or not 2x2), and every brick the inspection could not clear.

## Hardware

- SO-101 follower arm (Feetech STS3215 servos) on a USB serial adapter.
- Wrist camera on the gripper (USB, 640x480).
- Two Intel RealSense D415 cameras: a top view and a low close view of the work area.
- Intel Core Ultra laptop; inference on CPU, GPU or NPU through OpenVINO.

## How it works

1. **Scan**: from the scan pose the wrist camera finds the bricks on the mat. Plates are excluded in the image.
2. **Approach**: a learned scan map gives the first estimate, differential IK moves to a hover pose, and
   wrist-camera visual servoing centres the brick between the jaws and aligns the jaws with it. The servo learns a
   correction of its image Jacobian online.
3. **Grasp**: descent to the height given by the brick's image size, close, lift, hold check from the gripper position.
4. **Inspect**: the shape (2x2 or not) is measured in the wrist view. The close camera judges the brick at the
   inspection station; frame differencing finds the brick that was put down there.
5. **Decide and place**: blue only for a measured 2x2 brick with clean views, green for everything else.
6. **Safety**: a hand watchdog reads the top and close cameras and stops the arm when a hand enters the cell. The loop
   resumes when the cell is clear. Jaw and camera calibration are checked at start-up and every few cycles.

## Layout

| Path | What |
|---|---|
| `tools/start_all.sh`, `tools/stop_all.sh` | Start and stop the whole stack: recorders, watchdog, motion server, loop |
| `tools/motion_server.py` | One persistent connection to the arm; joint moves with gravity compensation; file-based command and state |
| `tools/sort_any.py`, `tools/sort_loop.py` | The sort loop and its decision rule |
| `tools/pick_any.py` | Wrist-camera brick detection, tracking and servoing |
| `tools/kin.py`, `pick_v2/kinematics.py` | URDF forward and inverse kinematics |
| `tools/closeloc.py` | Close-camera localisation by frame differencing |
| `anomaly/det2_f2/inspector.py` | Close-view brick segmentation and tape check |
| `tools/handdet.py`, `tools/hand_watch.py` | Hand watchdog |
| `tools/jaw_check.py` | Jaw and wrist-camera calibration check |
| `tools/paths.py` | Every path, derived from the checkout location |
| `tools/tests/` | Offline test suite |

## Running

```
bash tools/start_all.sh --run <name> --n 150     # start everything, run 150 cycles
bash tools/start_all.sh --status                 # what is running
bash tools/stop_all.sh                           # finish the cycle, park the arm, stop everything
python -I -B tools/tests/run_tests.py            # offline tests (data environment)
```

`tools/paths.py` derives the repository and the workspace from the checkout. The workspace (the parent folder, or
`JUNCTION_ROOT`) holds `envs/`, `run/` and `logs/`. The URDF is the SO-101 description from
edge-developer-kit-reference-scripts; `tools/kin.py` looks for it under `repos/` in the workspace.

Tests that replay recorded camera frames skip when the frames are absent. The frames are not published because they
show people.

## Status

- A taped brick was sorted to blue twice on 14 November: the close camera saw only its clean faces. The inspection is
  being rebuilt to see every face. The robot shows each face to a calibrated camera (the free sides by wrist rotation,
  the bottom by wrist tilt, the gripped sides after a 90 degree re-grip), and each face is tested against its own base
  colour, so tape or marks of any colour on a brick of any colour count. Blue only when every face was seen clean.
- Fixed values that remain (plate positions, inspection station pose, colour ranges, detector thresholds) are being
  replaced by calibration routines and fitted model files.
