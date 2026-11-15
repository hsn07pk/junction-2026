#!/usr/bin/env python3
"""Make the SO-101 DUPLO sorter demo video (demo.mp4) from one finished run.

Reads under the workspace root (--root R):
  logs/sort/RUN/cycles.jsonl              one JSON object per episode
  logs/sort/RUN/ep_NNNN/episode.json      t_start and t_end (UNIX seconds, UTC)
  logs/monitor/top_TS.mkv, close_TS.mkv   continuous top and close recordings
  data/wrist/raw/w_TS.mkv                 continuous wrist segments
where TS is the UTC start second, YYYYMMDDTHHMMSSZ.

Clock rule: a segment whose ffprobe container start_time is above 1e9 has absolute
epoch pts and starts there; otherwise its file name gives its start. A segment ends
where the next segment of the same camera starts. When its own media ends more than
GAP_TOL_S before that, it ends at its media end and the span in between is a gap.

Output: 1920x1080, 30 fps, H.264 (-preset veryfast -crf 23, yuv420p). Frames are piped
to ffmpeg. Only ffmpeg and ffprobe are run, through subprocess. Needs numpy and opencv.
"""

import argparse
import bisect
import datetime as dt
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np

FPS = 30
OUT_W, OUT_H = 1920, 1080
MAIN_H = 960                      # picture area; the caption bar takes the rest
CAP_H = OUT_H - MAIN_H            # 120 px caption bar
INSET_W, INSET_H = 384, 288
INSET_GAP = 16
MARGIN = 24
GAP_TOL_S = 1.0
EPOCH_MIN = 1e9
TITLE_S = 4.0
END_S = 6.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (175, 175, 175)
BLUE = (255, 130, 30)             # BGR
GREEN = (70, 215, 70)
BG = (24, 24, 24)
CAMS = ("top", "wrist", "close")
MONITOR_RE = re.compile(r"(top|close)_(\d{8}T\d{6}Z)\.mkv")
WRIST_RE = re.compile(r"w_(\d{8}T\d{6}Z)\.mkv")


@dataclass
class Segment:
    cam: str
    path: str
    start: float
    end: object        # float, or None when the end is unknown
    width: int
    height: int


def fail(msg, code=2):
    print("make_demo: error: " + msg, file=sys.stderr)
    sys.exit(code)


def note(msg):
    print("make_demo: " + msg, file=sys.stderr, flush=True)


def utc(epoch):
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def name_epoch(ts):
    return dt.datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def as_float(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def episode_arg(text):
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m and int(m.group(1)) <= int(m.group(2)):
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            out.add(int(part))
        else:
            raise argparse.ArgumentTypeError("bad episode item %r (use 1,3,5 or 2-9)" % part)
    return out


def episode_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        m = re.search(r"(\d+)", v)
        return int(m.group(1)) if m else None
    return None


def resolve_run(root, run):
    base = os.path.join(root, "logs", "sort")
    if not os.path.isdir(base):
        fail("no logs/sort folder under " + root)
    if run:
        if os.sep in run or run in (".", ".."):
            fail("--run must be a folder name under logs/sort")
        d = os.path.join(base, run)
        if not os.path.isfile(os.path.join(d, "cycles.jsonl")):
            fail(d + "/cycles.jsonl not found")
        return d
    cands = [os.path.join(base, n) for n in os.listdir(base)
             if os.path.isfile(os.path.join(base, n, "cycles.jsonl"))]
    if not cands:
        fail("no run with cycles.jsonl under " + base)
    chosen = max(cands, key=lambda p: os.path.getmtime(os.path.join(p, "cycles.jsonl")))
    note("no --run given, using the newest run: " + os.path.basename(chosen))
    return chosen


def read_cycles(path, notes):
    recs = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                notes.append("cycles.jsonl line %d is not valid JSON, skipped" % lineno)
                continue
            num = episode_number(obj.get("episode")) if isinstance(obj, dict) else None
            if num is None:
                notes.append("cycles.jsonl line %d has no episode number, skipped" % lineno)
                continue
            recs[num] = obj
    return recs


def read_episode(run_dir, num):
    path = os.path.join(run_dir, "ep_%04d" % num, "episode.json")
    if not os.path.isfile(path):
        return None, "no episode.json"
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None, "episode.json unreadable (still being written?)"
    if not isinstance(meta, dict):
        return None, "episode.json is not an object"
    t0, t1 = as_float(meta.get("t_start")), as_float(meta.get("t_end"))
    if t0 is None or t1 is None or t1 <= t0:
        return None, "no valid t_start and t_end"
    return {"t_start": t0, "t_end": t1, "meta": meta}, None


def collect_episodes(run_dir, cycles, wanted, notes):
    nums = set(cycles)
    for name in os.listdir(run_dir):
        m = re.fullmatch(r"ep_(\d+)", name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            nums.add(int(m.group(1)))
    if wanted is not None:
        for n in sorted(wanted - nums):
            notes.append("episode %d is not in this run" % n)
        nums &= wanted
    eps = []
    for num in sorted(nums):
        info, why = read_episode(run_dir, num)
        if info is None:
            notes.append("episode %d skipped: %s" % (num, why))
            continue
        info.update(num=num, cycle=cycles.get(num) or {})
        eps.append(info)
    return eps


def field_of(ep, key):
    for obj in (ep["cycle"], ep["meta"]):
        val = obj.get(key) if isinstance(obj, dict) else None
        if val not in (None, ""):
            return val
    return None


def decision_of(label):
    lab = str(label).strip().upper() if label is not None else ""
    if lab == "GOOD":
        return "GOOD", "BLUE: GOOD", BLUE
    if lab == "DEFECT":
        return "DEFECT", "GREEN: DEFECT", GREEN
    if lab == "UNCERTAIN":
        return "UNCERTAIN", "GREEN: UNCERTAIN", GREEN
    if lab:
        return "OTHER", "DECISION " + lab[:24], GREY
    return "NONE", "NO DECISION", GREY


def cycle_seconds(ep):
    for obj in (ep["cycle"], ep["meta"]):
        if not isinstance(obj, dict):
            continue
        timings = obj.get("timings")
        for v in (timings.get("cycle_s") if isinstance(timings, dict) else None, obj.get("cycle_s")):
            f = as_float(v)
            if f is not None:
                return f
    return ep["t_end"] - ep["t_start"]


def conf_text(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return "%.2f" % v
    s = str(v).strip()
    return s or None


def reasons_text(v):
    if isinstance(v, list):
        items = [str(x).strip() for x in v if str(x).strip()]
    elif v is None:
        items = []
    else:
        items = [str(v).strip()] if str(v).strip() else []
    return "; ".join(items) if items else "none recorded"


def probe(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=start_time,duration:stream=codec_type,width,height",
                        "-of", "json", path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise ValueError("ffprobe failed")
    try:
        data = json.loads(p.stdout or "{}")
        fmt = data.get("format", {}) or {}
        video = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
        if not video:
            raise ValueError("no video stream")
        return {"start": as_float(fmt.get("start_time")), "duration": as_float(fmt.get("duration")),
                "width": int(video[0]["width"]), "height": int(video[0]["height"])}
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(str(exc)) from None


def load_segments(cam, files, notes):
    recs = []
    for name_ep, path in files:
        try:
            info = probe(path)
        except ValueError as exc:
            notes.append("%s unreadable (%s), skipped" % (os.path.basename(path), exc))
            continue
        pts_start = info["start"]
        start = pts_start if (pts_start is not None and pts_start > EPOCH_MIN) else name_ep
        dur = info["duration"]
        if dur is None or dur <= 0:
            media_end = None
        elif dur > EPOCH_MIN:
            # with absolute pts the muxer stores the absolute end time as the duration
            media_end = dur if dur >= start else None
        else:
            media_end = start + dur
        recs.append([start, media_end, path, info["width"], info["height"]])
    recs.sort(key=lambda r: r[0])
    segs = []
    for i, (start, media_end, path, w, h) in enumerate(recs):
        nxt = recs[i + 1][0] if i + 1 < len(recs) else None
        if nxt is None:
            end = media_end
        elif media_end is None or nxt - media_end <= GAP_TOL_S:
            end = nxt
        else:
            end = media_end
        segs.append(Segment(cam, path, start, end, w, h))
    return segs


def scan_recordings(root, notes):
    found = {cam: [] for cam in CAMS}
    for folder in (os.path.join(root, "logs", "monitor"), os.path.join(root, "data", "wrist", "raw")):
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            cam, ts = None, None
            m = MONITOR_RE.fullmatch(name)
            if m:
                cam, ts = m.group(1), m.group(2)
            m = WRIST_RE.fullmatch(name)
            if m:
                cam, ts = "wrist", m.group(1)
            if cam is None:
                continue
            try:
                found[cam].append((name_epoch(ts), os.path.join(folder, name)))
            except ValueError:
                notes.append("%s has an impossible time stamp, skipped" % name)
    return found


def segment_at(segs, starts, t):
    i = bisect.bisect_right(starts, t) - 1
    for j in (i, i - 1):
        if j >= 0 and segs[j].end is not None and segs[j].start <= t < segs[j].end:
            return segs[j]
    return None


def coverage_runs(segs, starts, t0, step, n):
    runs = []
    for k in range(n):
        seg = segment_at(segs, starts, t0 + k * step)
        if runs and runs[-1][2] is seg:
            runs[-1][1] = k
        else:
            runs.append([k, k, seg])
    return runs


def plan_episode(ep, speed, segs, starts):
    step = speed / FPS
    span = ep["t_end"] - ep["t_start"]
    n = max(1, int(math.floor(span * FPS / speed + 1e-6)))
    runs, gaps = {}, []
    for cam in CAMS:
        runs[cam] = coverage_runs(segs[cam], starts[cam], ep["t_start"], step, n)
        if not segs[cam]:
            gaps.append("%s view not recorded" % cam.upper())
            continue
        for k0, k1, seg in runs[cam]:
            if seg is None:
                gaps.append("%s view missing %.1f-%.1f s" % (cam.upper(), k0 * step,
                                                              min(span, (k1 + 1) * step)))
    kind, dec_text, dec_col = decision_of(field_of(ep, "label"))
    return {"num": ep["num"], "t0": ep["t_start"], "t1": ep["t_end"], "n": n, "runs": runs,
            "kind": kind, "dec_text": dec_text, "dec_col": dec_col, "cyc": cycle_seconds(ep),
            "conf": conf_text(field_of(ep, "decision_conf")),
            "reasons": reasons_text(field_of(ep, "decision_reasons")), "gaps": gaps}


def totals_of(plans):
    t = {"bricks": len(plans), "good": 0, "defect": 0, "uncertain": 0, "undecided": 0}
    cycles = []
    for p in plans:
        if p["kind"] in ("GOOD", "DEFECT", "UNCERTAIN"):
            t[p["kind"].lower()] += 1
        else:
            t["undecided"] += 1
        cycles.append(p["cyc"])
    t["green"] = t["defect"] + t["uncertain"]
    t["median"] = statistics.median(cycles) if cycles else None
    return t


def read_exact(stream, n):
    parts, got = [], 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            return None
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def frame_stream(runs, t0, step, rate_str, cam):
    """Yield one frame (ndarray) or None per output frame, decoding one run at a time."""
    for k0, k1, seg in runs:
        n = k1 - k0 + 1
        if seg is None:
            for _ in range(n):
                yield None
            continue
        nbytes = seg.width * seg.height * 3
        offset = max(0.0, t0 + k0 * step - seg.start)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "%.4f" % offset, "-i", seg.path,
               "-map", "0:v:0", "-an", "-frames:v", str(n), "-vf", "fps=" + rate_str,
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        got, last = 0, None
        try:
            while got < n:
                buf = read_exact(proc.stdout, nbytes)
                if buf is None:
                    break
                last = np.frombuffer(buf, np.uint8).reshape(seg.height, seg.width, 3)
                got += 1
                yield last
        finally:
            proc.stdout.close()
            proc.wait()
        if got < n:
            note("%s: %s gave %d of %d frames" % (cam, os.path.basename(seg.path), got, n))
        # a shortfall of a few frames is segment-end rounding: hold the last frame; more is a gap
        hold = last if (n - got) <= 3 else None
        for _ in range(n - got):
            yield hold


def fit(img, box_w, box_h):
    h, w = img.shape[:2]
    s = min(box_w / float(w), box_h / float(h))
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_LINEAR)


def fit_text(text, max_w, scale, thick, min_scale=0.35):
    s = scale
    while True:
        (tw, _), _ = cv2.getTextSize(text, FONT, s, thick)
        if tw <= max_w:
            return s, text
        if s - 0.05 < min_scale:
            break
        s = round(s - 0.05, 2)
    t = text
    while t:
        t = t[:-1]
        (tw, _), _ = cv2.getTextSize(t + "...", FONT, min_scale, thick)
        if tw <= max_w:
            return min_scale, t + "..."
    return min_scale, ""


def caption_bar(head, accent, tail):
    bar = np.zeros((CAP_H, OUT_W, 3), np.uint8)
    bar[:] = BG
    cv2.rectangle(bar, (0, 0), (OUT_W - 1, 9), accent, -1)
    scale1, _ = fit_text("".join(t for t, _ in head), OUT_W - 60, 1.2, 3)
    x = 30
    for text, colour in head:
        cv2.putText(bar, text, (x, 62), FONT, scale1, colour, 3, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, FONT, scale1, 3)
        x += tw
    scale2, tail_fit = fit_text(tail, OUT_W - 60, 0.85, 2)
    cv2.putText(bar, tail_fit, (30, 106), FONT, scale2, GREY, 2, cv2.LINE_AA)
    return bar


def build_base(p, idx, total):
    head = [("EP %d (%d of %d)   " % (p["num"], idx, total), WHITE), (p["dec_text"], p["dec_col"]),
            ("   CYCLE %.1f s" % p["cyc"], WHITE)]
    if p["conf"]:
        head.append(("   CONF " + p["conf"], WHITE))
    tail = "REASONS: " + p["reasons"]
    if p["gaps"]:
        tail += "   NOTE: " + "; ".join(p["gaps"])
    base = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    base[MAIN_H:] = caption_bar(head, p["dec_col"], tail)
    return base


def speed_label(speed):
    return "REAL TIME" if abs(speed - 1.0) < 1e-9 else "TIME-LAPSE x%g" % speed


def compose(base, fr, elapsed, label):
    frame = base.copy()
    if fr["top"] is not None:
        tv = fit(fr["top"], OUT_W, MAIN_H)
        h, w = tv.shape[:2]
        y0, x0 = (MAIN_H - h) // 2, (OUT_W - w) // 2
        frame[y0:y0 + h, x0:x0 + w] = tv
    else:
        s, t = fit_text("TOP VIEW NOT RECORDED", OUT_W - 200, 1.6, 3)
        (tw, _), _ = cv2.getTextSize(t, FONT, s, 3)
        cv2.putText(frame, t, ((OUT_W - tw) // 2, MAIN_H // 2), FONT, s, GREY, 3, cv2.LINE_AA)
    x, y = OUT_W - INSET_W - MARGIN, MARGIN
    for cam in ("wrist", "close"):
        if fr[cam] is not None:
            ins = cv2.resize(fr[cam], (INSET_W, INSET_H), interpolation=cv2.INTER_AREA)
            frame[y:y + INSET_H, x:x + INSET_W] = ins
            cv2.rectangle(frame, (x, y), (x + INSET_W - 1, y + INSET_H - 1), WHITE, 2)
            cv2.rectangle(frame, (x, y), (x + 96, y + 30), (0, 0, 0), -1)
            cv2.putText(frame, cam.upper(), (x + 8, y + 22), FONT, 0.7, WHITE, 2, cv2.LINE_AA)
        y += INSET_H + INSET_GAP
    cv2.putText(frame, "%s   +%.1f s" % (label, elapsed), (30, 50), FONT, 1.0, WHITE, 2, cv2.LINE_AA)
    return frame


def make_card(lines):
    img = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    img[:] = BG
    for text, scale, colour, y, thick in lines:
        s, t = fit_text(text, OUT_W - 160, scale, thick)
        (tw, _), _ = cv2.getTextSize(t, FONT, s, thick)
        cv2.putText(img, t, ((OUT_W - tw) // 2, y), FONT, s, colour, thick, cv2.LINE_AA)
    return img


def title_card(run_name, n, first, last):
    return make_card([
        ("SO-101 DUPLO SORTER", 2.4, WHITE, 430, 4),
        ("Run " + run_name, 1.1, GREY, 520, 2),
        ("%d episodes, recorded %s to %s" % (n, utc(first), utc(last)), 0.9, GREY, 600, 2),
    ])


def end_card(t):
    med = "%.1f s" % t["median"] if t["median"] is not None else "n/a"
    lines = [
        ("RUN TOTALS", 2.2, WHITE, 330, 4),
        ("Bricks: %d" % t["bricks"], 1.5, WHITE, 430, 3),
        ("BLUE (GOOD): %d" % t["good"], 1.5, BLUE, 510, 3),
        ("GREEN (DEFECT or UNCERTAIN): %d   (defect %d, uncertain %d)"
         % (t["green"], t["defect"], t["uncertain"]), 1.2, GREEN, 590, 2),
        ("Median cycle time: " + med, 1.5, WHITE, 680, 3),
    ]
    if t["undecided"]:
        lines.append(("No decision: %d" % t["undecided"], 1.1, GREY, 760, 2))
    return make_card(lines)


def describe_runs(runs):
    return " ".join("%s[%d-%d]" % (os.path.basename(s.path) if s is not None else "GAP", a, b)
                    for a, b, s in runs)


def print_plan(plans, title_n, end_n, speed, out_path, notes, run_name):
    print("RUN %s  speed x%g  output %s" % (run_name, speed, out_path))
    for line in notes:
        print("note: " + line)
    total = title_n + end_n
    print("clip 0: title card, %d frames (%.2f s)" % (title_n, title_n / float(FPS)))
    for idx, p in enumerate(plans, 1):
        total += p["n"]
        print("clip %d: EP %d, %d frames (%.2f s video from %.2f s real), window %s to %s"
              % (idx, p["num"], p["n"], p["n"] / float(FPS), p["t1"] - p["t0"], utc(p["t0"]), utc(p["t1"])))
        print("   caption: EP %d | %s | cycle %.1f s | conf %s | reasons: %s"
              % (p["num"], p["dec_text"], p["cyc"], p["conf"] or "n/a", p["reasons"]))
        for cam in CAMS:
            print("   %s: %s" % (cam, describe_runs(p["runs"][cam])))
        for g in p["gaps"]:
            print("   note: " + g)
    print("clip %d: end card, %d frames (%.2f s)" % (len(plans) + 1, end_n, end_n / float(FPS)))
    print("total: %d frames = %.2f s at %d fps, %dx%d" % (total, total / float(FPS), FPS, OUT_W, OUT_H))


def emit(enc, img, count):
    data = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
    for _ in range(count):
        enc.stdin.write(data)


def render(plans, title_img, end_img, title_n, end_n, out_path, speed, rate_str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (OUT_W, OUT_H), "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-r", str(FPS), "-movflags", "+faststart", out_path]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    failed = False
    try:
        emit(enc, title_img, title_n)
        for idx, p in enumerate(plans, 1):
            note("rendering EP %d (%d of %d), %d frames" % (p["num"], idx, len(plans), p["n"]))
            base = build_base(p, idx, len(plans))
            streams = {cam: frame_stream(p["runs"][cam], p["t0"], speed / FPS, rate_str, cam)
                       for cam in CAMS}
            try:
                for k in range(p["n"]):
                    fr = {cam: next(streams[cam]) for cam in CAMS}
                    emit(enc, compose(base, fr, k * speed / FPS, speed_label(speed)), 1)
            finally:
                for g in streams.values():
                    g.close()
        emit(enc, end_img, end_n)
    except BrokenPipeError:
        failed = True
    finally:
        try:
            enc.stdin.close()
        except BrokenPipeError:
            failed = True
        rc = enc.wait()
    if failed or rc != 0:
        fail("ffmpeg encoder failed (exit code %s)" % rc, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Make the SO-101 DUPLO sorter demo video (demo.mp4).")
    ap.add_argument("--root", required=True, help="workspace root R (holds logs/ and data/)")
    ap.add_argument("--run", help="run folder under logs/sort (default: newest run with cycles.jsonl)")
    ap.add_argument("--episodes", type=episode_arg, help="episodes to show, e.g. 1,3,5 or 2-9 (default: all)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="time-lapse factor for the motion (default 1 = real time, 4 = four times faster)")
    ap.add_argument("--out", default="demo.mp4", help="output file (default demo.mp4)")
    ap.add_argument("--dry-run", action="store_true", help="print the clip plan and render nothing")
    args = ap.parse_args(argv)
    if not (math.isfinite(args.speed) and args.speed > 0):
        ap.error("--speed must be a positive number")
    for tool in (["ffprobe"] if args.dry_run else ["ffprobe", "ffmpeg"]):
        if shutil.which(tool) is None:
            fail(tool + " not found on PATH")

    root = os.path.abspath(args.root)
    run_dir = resolve_run(root, args.run)
    run_name = os.path.basename(run_dir)
    notes = []
    cycles = read_cycles(os.path.join(run_dir, "cycles.jsonl"), notes)
    eps = collect_episodes(run_dir, cycles, args.episodes, notes)
    if not eps:
        fail("no usable episodes in " + run_dir)

    found = scan_recordings(root, notes)
    segs, starts = {}, {}
    for cam in CAMS:
        segs[cam] = load_segments(cam, sorted(found[cam]), notes)
        starts[cam] = [s.start for s in segs[cam]]
        note("%s: %d segment(s) found" % (cam, len(segs[cam])))

    speed = args.speed
    plans = [plan_episode(ep, speed, segs, starts) for ep in eps]
    rate = Fraction(FPS) / Fraction(str(speed))
    rate_str = str(rate.numerator) if rate.denominator == 1 else "%d/%d" % (rate.numerator, rate.denominator)
    title_n, end_n = int(round(TITLE_S * FPS)), int(round(END_S * FPS))
    print_plan(plans, title_n, end_n, speed, args.out, notes, run_name)
    if args.dry_run:
        print("dry run: nothing rendered")
        return 0

    title_img = title_card(run_name, len(plans), min(p["t0"] for p in plans), max(p["t1"] for p in plans))
    end_img = end_card(totals_of(plans))
    render(plans, title_img, end_img, title_n, end_n, args.out, speed, rate_str)
    note("wrote " + os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
