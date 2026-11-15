#!/bin/bash
# Hand the robot over: fold the arm to its rest pose under a seeing watchdog, switch torque off, close the follower
# port, then free every camera this stack opened (recorders and their ffmpeg children, watchdog, motion server).
# The recorders and the watchdog are (re)started first, so the watchdog sees the cell during the fold; a move that the
# watchdog stops (hand in the cell) is retried when the cell is clear again.
#   bash app/tools/park_release.sh
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # workspace root, from this script's location
cd "$R" || exit 1
PY=$R/envs/data-env/bin/python
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
stop_tree() {  # a recorder script and every process it started
  local f=$1
  alive "$f" || return 0
  local p; p=$(cat "$f")
  pkill -TERM -P "$p" 2>/dev/null
  kill -TERM "$p" 2>/dev/null
  say "$(basename "$f" .pid) stopped with its children"
}
stray_ffmpeg() { pgrep -f "ffmpeg .*-i /dev/video(8|14|16)( |$)"; }
if alive run/sort_any.pid; then
  say "the sort loop is running: stop it first with stop_all.sh"; exit 2
fi
if alive run/mserver/pid; then
  if ! alive run/monitor_cams.pid || ! alive run/wrist_recorder.pid || ! alive run/hand_watch.pid; then
    for p in $(stray_ffmpeg); do kill -TERM "$p" 2>/dev/null && say "stray recorder ffmpeg $p stopped"; done
    sleep 1
    bash app/tools/start_all.sh --no-loop > /dev/null 2>&1
    say "recorders and watchdog (re)started for the fold"
  fi
  (cd app/tools && $PY -B -c "
import os
import time
import grasp_auto as ga
import handdet
import rest
t0 = time.time()
while not handdet.heartbeat_fresh() or time.time() - os.path.getmtime(os.path.join(ga.RUN_ROOT, 'live_top.jpg')) > 1.0:
    if time.time() - t0 > 30:
        raise SystemExit('watchdog or live frames not up after 30 s: arm left holding its pose')
    time.sleep(0.5)
done = False
for attempt in range(120):
    try:
        e = os.path.join(ga.RUN_ROOT, 'estop')
        if os.path.exists(e) and time.time() - os.path.getmtime(e) > 3.0:
            os.remove(e)
        rest.fold()
        done = True
        break
    except ga.Stop as ex:
        print('fold stopped by the watchdog (attempt', attempt + 1, '):', ex, flush=True)
        time.sleep(2.0)
if not done:
    raise SystemExit('cell never clear: arm left holding its pose')
ga.cmd({'op': 'release'})                                               # torque off, port closed
print('arm at rest, torque off')
") && say "arm folded to rest, torque off" || { say "fold failed: arm left as is, stack left running"; exit 3; }
  kill -TERM "$(cat run/mserver/pid)" 2>/dev/null && say "motion server stopped (follower port free)"
fi
alive run/hand_watch.pid && { kill -TERM "$(cat run/hand_watch.pid)"; say "watchdog stopped"; }
stop_tree run/monitor_cams.pid
stop_tree run/wrist_recorder.pid
sleep 1
for p in $(stray_ffmpeg); do kill -TERM "$p" 2>/dev/null && say "stray recorder ffmpeg $p stopped"; done
say "robot and cameras free"
