#!/usr/bin/env python3
"""Cut each camera's continuous recording to the time window of one or more sort runs and save one video per camera.

The window runs from the first episode's start to the last episode's end over the given runs (logs/sort/RUN/
ep_NNNN/episode.json, t_start and t_end, UNIX seconds UTC), with a margin. The recordings are found and timed exactly as
make_demo.py does (absolute pts when the container has them, else the file name's UTC second). Each camera's pieces are
re-encoded to H.264 and joined in time order.

usage: export_cams.py --root R --runs demoA,demoB --out-dir R/logs/demo [--margin 3]"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_demo as md  # noqa: E402


def window(root: str, runs: list[str], margin: float) -> tuple[float, float]:
    t0, t1 = None, None
    for run in runs:
        for f in glob.glob(os.path.join(root, "logs", "sort", run, "ep_*", "episode.json")):
            try:
                e = json.load(open(f))
            except (OSError, ValueError):
                continue
            a, b = e.get("t_start"), e.get("t_end")
            if isinstance(a, (int, float)):
                t0 = a if t0 is None else min(t0, a)
            if isinstance(b, (int, float)):
                t1 = b if t1 is None else max(t1, b)
    if t0 is None or t1 is None:
        raise SystemExit(f"no episode times in runs {runs}")
    return t0 - margin, t1 + margin


def export(cam: str, segs: list, t0: float, t1: float, out: str, tmp: str) -> float:
    parts = []
    for s in segs:
        a, b = max(t0, s.start), min(t1, s.end if s.end is not None else t1)
        if b - a < 0.2:
            continue
        p = os.path.join(tmp, f"{cam}_{len(parts):03d}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a - s.start:.3f}", "-i", s.path, "-t", f"{b - a:.3f}",
                        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-r", "30", p], check=True)
        parts.append(p)
    if not parts:
        return 0.0
    lst = os.path.join(tmp, f"{cam}_list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", out], check=True)
    return sum(1 for _ in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run names under logs/sort")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--margin", type=float, default=3.0)
    a = ap.parse_args()
    runs = [r for r in a.runs.split(",") if r]
    t0, t1 = window(a.root, runs, a.margin)
    os.makedirs(a.out_dir, exist_ok=True)
    tmp = os.path.join(a.out_dir, "parts")
    os.makedirs(tmp, exist_ok=True)
    notes = []
    found = md.scan_recordings(a.root, notes)
    tag = "_".join(runs)
    for cam, files in found.items():
        near = [(ep, p) for ep, p in files if t0 - 4000 <= ep <= t1]  # only segments that can overlap the window
        segs = md.load_segments(cam, near, notes)
        out = os.path.join(a.out_dir, f"{tag}_{cam}.mp4")
        n = export(cam, segs, t0, t1, out, tmp)
        print(f"{cam}: {n} piece(s) -> {out if n else 'nothing recorded in the window'}")
    for n in notes:
        print("note:", n)
    print(f"window {md.utc(t0)} .. {md.utc(t1)} ({t1 - t0:.0f} s)")


if __name__ == "__main__":
    main()
