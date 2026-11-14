#!/bin/bash
# Real-time monitor: both RealSense colour streams recorded to MKV at the native 30 fps with ABSOLUTE wall-clock
# timestamps (-ts abs + -copyts: packet pts = Unix time), so robot episodes can be cut exactly from the joint logs;
# mirrored as JPEGs (8 fps, written atomically: temp file + rename) for the hand watchdog and the loop.
# Each camera restarts by itself if its ffmpeg exits (a dead recorder would otherwise leave a stale frame; the
# watchdog then stops writing its heartbeat and the arm does not move).
# Stop: kill $(cat run/monitor_cams.pid)
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # workspace root, from this script's location
mkdir -p "$R/logs/monitor" "$R/run"
echo $$ > "$R/run/monitor_cams.pid"

record() {  # $1 = view (top|close), $2 = device
  while true; do
    TS=$(date -u +%Y%m%dT%H%M%SZ)
    ffmpeg -hide_banner -loglevel error -f v4l2 -ts abs -video_size 640x480 -framerate 30 -i "$2" \
      -map 0 -copyts -c:v libx264 -preset veryfast -crf 23 -g 30 "$R/logs/monitor/$1_$TS.mkv" \
      -map 0 -vf fps=8 -update 1 -atomic_writing 1 -y "$R/run/live_$1.jpg" >> "$R/logs/monitor/ffmpeg_$1.log" 2>&1
    echo "$(date -u +%FT%TZ) $1 ffmpeg exited $?; restarting in 2 s" >> "$R/logs/monitor/ffmpeg_$1.log"
    sleep 2
  done
}

record top /dev/video8 &
P1=$!
record close /dev/video14 &
P2=$!
trap 'kill $P1 $P2 2>/dev/null; pkill -P $P1 2>/dev/null; pkill -P $P2 2>/dev/null; rm -f "$R/run/monitor_cams.pid"; exit 0' TERM INT
wait
