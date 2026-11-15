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


def windows(root: str, runs: list[str], margin: float, episodes: set | None) -> list[tuple[float, float]]:
    """One window over all episodes of the runs, or one window per chosen episode (in time order)."""
    spans = []
    for run in runs:
        for f in sorted(glob.glob(os.path.join(root, "logs", "sort", run, "ep_*", "episode.json"))):
            num = int(os.path.basename(os.path.dirname(f)).split("_")[1])
            if episodes is not None and num not in episodes:
                continue
            try:
                e = json.load(open(f))
            except (OSError, ValueError):
                continue
            a, b = e.get("t_start"), e.get("t_end")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > a:
                spans.append((a - margin, b + margin))
    if not spans:
        raise SystemExit(f"no episode times in runs {runs}")
    spans.sort()
    if episodes is None:
        return [(spans[0][0], max(b for _, b in spans))]
    return spans


def export(cam: str, segs: list, spans: list, out: str, tmp: str) -> float:
    parts = []
    for t0, t1 in spans:
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
    ap.add_argument("--episodes", default=None, help="only these episodes (list or ranges, as make_demo.py), joined in order")
    a = ap.parse_args()
    runs = [r for r in a.runs.split(",") if r]
    eps = set(md.episode_arg(a.episodes)) if a.episodes else None
    spans = windows(a.root, runs, a.margin, eps)
    t0, t1 = spans[0][0], spans[-1][1]
    os.makedirs(a.out_dir, exist_ok=True)
    tmp = os.path.join(a.out_dir, "parts")
    os.makedirs(tmp, exist_ok=True)
    notes = []
    found = md.scan_recordings(a.root, notes)
    tag = "_".join(runs)
    for cam, files in found.items():
        files = sorted(files)
        before = [f for f in files if f[0] <= t0]
        near = before[-1:] + [f for f in files if t0 < f[0] <= t1]  # the segment running at t0 and those starting later
        segs = md.load_segments(cam, near, notes)
        out = os.path.join(a.out_dir, f"{tag}_{cam}.mp4")
        n = export(cam, segs, spans, out, tmp)
        print(f"{cam}: {n} piece(s) -> {out if n else 'nothing recorded in the window'}")
    for n in notes:
        print("note:", n)
    print(f"{len(spans)} window(s) {md.utc(t0)} .. {md.utc(t1)} ({sum(b - a for a, b in spans):.0f} s in all)")


if __name__ == "__main__":
    main()
