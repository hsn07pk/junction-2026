#!/bin/bash
# Stop the sorting cell cleanly: the loop finishes its cycle, the arm folds to its rest pose with the gripper open and
# torque goes off, then the hand watchdog and the camera recorders stop. The watchdog keeps running until the arm is
# at rest, because the motion server refuses to move without it.
#   bash app/tools/stop_all.sh [--now]   (--now: do not wait for the cycle to finish)
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # workspace root, from this script's location
cd "$R" || exit 1
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

if alive run/sort_any.pid; then
  if [ "${1:-}" = "--now" ]; then
    kill -TERM "$(cat run/sort_any.pid)"; say "sort loop stopped now"
  else
    touch run/sort_stop; say "sort loop asked to stop after this cycle"
    for i in $(seq 1 240); do alive run/sort_any.pid || break; sleep 1; done
    alive run/sort_any.pid && { kill -TERM "$(cat run/sort_any.pid)"; say "sort loop did not finish in 4 min: stopped"; }
  fi
fi
if alive run/mserver/pid; then
  (cd app/tools && $R/envs/data-env/bin/python -B -c "
import grasp_auto as ga
st = ga.state()
q = st['present']
ga.move([q[0], -40.0, 80.0, 66.0, -45.0, 25.0], speed=15, comp=False)   # up first: never sweep low
ga.move([-6.0, -98.0, 96.6, 66.0, -45.0, 25.0], speed=15, comp=False)   # folded rest pose
ga.cmd({'op': 'release'})                                               # torque off, port closed
print('arm at rest, torque off')
") && say "arm folded to rest, torque off" || say "could not move the arm to rest (e-stop or watchdog?): motion server left as is"
fi
for p in hand_watch monitor_cams wrist_recorder; do
  f=run/$p.pid; alive "$f" && { kill -TERM "$(cat "$f")"; say "$p stopped"; }
done
say "cell stopped"
