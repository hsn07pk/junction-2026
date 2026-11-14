#!/bin/bash
# Continuous wrist-camera recorder: MJPEG 640x480@30 copied into 60 s MKV segments
# with ABSOLUTE wall-clock timestamps (-ts abs + -copyts, no reset), names in UTC, restarts ffmpeg if it exits.
# Stop with: kill $(cat run/wrist_recorder.pid)  (stops the loop and ffmpeg)
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # workspace root, from this script's location
OUT=$R/data/wrist/raw
mkdir -p "$OUT" "$R/logs"
echo $$ > "$R/run/wrist_recorder.pid"
trap 'kill $FF 2>/dev/null; rm -f "$R/run/wrist_recorder.pid"; exit 0' TERM INT
while true; do
  TZ=UTC ffmpeg -hide_banner -loglevel warning -f v4l2 -ts abs -input_format mjpeg -video_size 640x480 -framerate 30 \
    -i /dev/video16 -map 0 -c copy -copyts -f segment -segment_time 60 -strftime 1 \
    "$OUT/w_%Y%m%dT%H%M%SZ.mkv" -map 0 -vf fps=10 -update 1 -atomic_writing 1 -y "$R/run/live_wrist.jpg" >> "$R/logs/wrist_rec.log" 2>&1 &
  FF=$!
  wait $FF
  echo "$(date -u +%FT%TZ) ffmpeg exited $?; restarting in 3 s" >> "$R/logs/wrist_rec.log"
  sleep 3
done
