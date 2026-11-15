#!/bin/bash
# Start the complete sorting cell in a safe order: cameras (recordings + live frames), hand watchdog, motion server,
# sort loop. Running parts are kept, so it is safe to run again. It refuses when another program holds the arm or a
# camera (stop Physical AI Studio sessions first).
#
#   bash app/tools/start_all.sh                start everything, new run, 150 cycles
#   bash .../start_all.sh --run NAME --n 150 --start 1 --profile safe|fast    choose the run
#   bash .../start_all.sh --run cal_b1 --n 3 --calibrate b1 --truth GOOD      calibration run of one brick
#   bash .../start_all.sh --no-loop                                           cameras, watchdog, motion server only
#   bash .../start_all.sh --status                                            show what is running
# Stop everything cleanly with: bash app/tools/stop_all.sh
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # workspace root, from this script's location
PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00
PY=$R/envs/data-env/bin/python
cd "$R" || exit 1
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
age() { echo $(( $(date +%s) - $(stat -c %Y "$1" 2>/dev/null || echo 0) )); }

RUN="any_$(date -u +%Y%m%d_%H%M)"; N=150; START=1; PROFILE=safe; LOOP=1; STATUS=0; CALIBRATE=""; TRUTH=""; BENT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=$2; shift 2 ;;
    --n) N=$2; shift 2 ;;
    --start) START=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --calibrate) CALIBRATE=$2; shift 2 ;;
    --truth) TRUTH=$2; shift 2 ;;
    --accept-bent-jaw) BENT=1; shift ;;
    --no-loop) LOOP=0; shift ;;
    --status) STATUS=1; shift ;;
    *) echo "unknown option $1"; exit 64 ;;
  esac
done

status() {
  for p in monitor_cams:run/monitor_cams.pid wrist_recorder:run/wrist_recorder.pid hand_watch:run/hand_watch.pid \
           motion_server:run/mserver/pid sort_loop:run/sort_any.pid; do
    n=${p%%:*}; f=${p#*:}
    alive "$f" && say "$n running (pid $(cat "$f"))" || say "$n NOT running"
  done
  for c in top close wrist; do say "live_$c.jpg $(age run/live_$c.jpg) s old"; done
  hb=$($PY -c "import time;print(round(time.time()-float(open('run/hand_watch_alive').read()),1))" 2>/dev/null || echo none)
  say "watchdog heartbeat ${hb} s old (motion needs <= 3 s)"
  [ -f run/estop ] && say "e-stop active: $(head -c 100 run/estop)"
  [ -f run/mserver/state.json ] && say "motion server: $(cut -c1-110 run/mserver/state.json)"
  if [ -f run/sort_run_name ]; then
    r=$(cat run/sort_run_name); say "run $r: $(tail -1 logs/sort/$r/loop.log 2>/dev/null | cut -c1-150)"
    [ -f "logs/sort/$r/needs_operator" ] && say "NEEDS OPERATOR: $(tr '\n' ' ' < logs/sort/$r/needs_operator)"
  fi
}
if [ "$STATUS" = 1 ]; then status; exit 0; fi

# 1. the arm's port must be ours or free
DEV=$(readlink -f "$PORT" 2>/dev/null)
[ -n "$DEV" ] || { say "arm port $PORT not present: is the follower arm plugged in and powered?"; exit 2; }
HOLD=$(fuser "$DEV" 2>/dev/null | tr -s ' ')
if [ -n "$HOLD" ] && ! alive run/mserver/pid; then say "arm port $DEV is held by pid(s)$HOLD (not this system): stop that program first"; exit 3; fi
SESS=$($PY -B app/tools/studio_sessions.py 2>/dev/null | grep -ci follower)
[ "${SESS:-0}" = "0" ] || { say "a Physical AI Studio session uses the follower arm: stop it in Studio first"; exit 4; }

# 2. cameras must be ours or free
for v in /dev/video8 /dev/video14 /dev/video16; do
  [ -e "$v" ] || { say "camera $v not present: check the USB cables"; exit 5; }
  h=$(fuser "$v" 2>/dev/null | tr -s ' ')
  if [ -n "$h" ] && ! alive run/monitor_cams.pid && ! alive run/wrist_recorder.pid; then say "camera $v is held by pid(s)$h (not this system)"; exit 5; fi
done
FREE=$(df --output=avail -BG "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge 5 ] || { say "only ${FREE} GB free on disk: recordings need space"; exit 6; }

# 3. recordings and live frames (the watchdog and the loop read run/live_*.jpg)
mkdir -p logs/monitor run/mserver
alive run/monitor_cams.pid || { nohup bash app/tools/monitor_cams.sh >> logs/monitor/monitor_cams.out 2>&1 & say "top/close recorder started"; }
alive run/wrist_recorder.pid || { nohup bash app/tools/wrist_rec.sh >> logs/wrist_rec.out 2>&1 & say "wrist recorder started"; }
for i in $(seq 1 20); do
  ok=1; for c in top close wrist; do [ "$(age run/live_$c.jpg)" -le 2 ] || ok=0; done
  [ $ok = 1 ] && break; sleep 1
done
for c in top close wrist; do
  a=$(age run/live_$c.jpg); [ "$a" -le 2 ] || { say "camera $c is not streaming (live frame ${a} s old): see logs/monitor/monitor_cams.out, logs/wrist_rec.out"; exit 7; }
done
say "cameras streaming"

# 4. hand watchdog (no motion is possible without its heartbeat)
alive run/hand_watch.pid || { nohup $PY -u -B app/tools/hand_watch.py >> logs/monitor/hand_watch.out 2>&1 & echo $! > run/hand_watch.pid; say "hand watchdog started"; }
for i in $(seq 1 10); do
  hb=$($PY -c "import time;print(int(time.time()-float(open('run/hand_watch_alive').read())<=3))" 2>/dev/null || echo 0)
  [ "$hb" = 1 ] && break; sleep 1
done
[ "$hb" = 1 ] || { say "hand watchdog heartbeat missing: see logs/monitor/hand_watch.out"; exit 8; }
say "hand watchdog alive"

# 5. motion server (torque on at the present pose; it moves only on request, with a live watchdog)
if ! alive run/mserver/pid; then
  nohup $R/envs/motion-env/bin/python -u -B app/tools/motion_server.py > run/mserver/stdout.log 2>&1 &
  for i in $(seq 1 30); do sleep 1; alive run/mserver/pid && grep -q '"ready"\|"ok"' run/mserver/state.json 2>/dev/null && break; done
fi
alive run/mserver/pid || { say "motion server did not start: see run/mserver/stdout.log"; exit 9; }
say "motion server ready"

# 6. sort loop (it waits for a clear workspace, finishes a held brick, checks the jaws, then sorts)
if [ "$LOOP" = 1 ]; then
  if alive run/sort_any.pid; then
    say "sort loop already running (run $(cat run/sort_run_name 2>/dev/null)): kept"
  else
    rm -f run/sort_stop run/sort_pause
    echo "$RUN" > run/sort_run_name
    (cd app/tools && nohup $PY -u -B sort_any.py --run "$RUN" --n "$N" --start "$START" --profile "$PROFILE" ${CALIBRATE:+--calibrate "$CALIBRATE" --truth "$TRUTH"} ${BENT:+--accept-bent-jaw} \
        >> "$R/logs/sort_any_$RUN.out" 2>&1 & echo $! > "$R/run/sort_any.pid")
    sleep 3
    alive run/sort_any.pid || { say "sort loop exited at start: see logs/sort_any_$RUN.out"; tail -5 "logs/sort_any_$RUN.out"; exit 10; }
    say "sort loop started: run $RUN ($N cycles, profile $PROFILE); log logs/sort/$RUN/loop.log"
  fi
fi
status
say "running. Stop cleanly with: bash $R/app/tools/stop_all.sh"
