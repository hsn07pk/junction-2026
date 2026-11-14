"""Per-episode review table for one sort run: outcome, decision evidence, timings, servo effort, interruptions.
Reads the run's loop.log and ep_*/episode.json; prints one line per episode and the run totals, and flags what needs
a look (no evidence, stop, slow phase, large first servo error, station fallback, verdict disagreement).
usage: run_report.py RUN [--since N]"""
import argparse
import glob
import json
import os
import re
import statistics

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP, R  # noqa: E402  (roots from this file's location)
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--since", type=int, default=0)
a = ap.parse_args()
base = f"{R}/logs/sort/{a.run}"
log = open(f"{base}/loop.log").read().splitlines() if os.path.exists(f"{base}/loop.log") else []

ep_lines = {}
cur = None
for ln in log:
    m = re.search(r" ep (\d+):", ln)
    if m:
        cur = int(m.group(1))
    if cur is not None:
        ep_lines.setdefault(cur, []).append(ln)

rows, flags_all = [], []
for f in sorted(glob.glob(f"{base}/ep_*/episode.json")):
    try:
        d = json.load(open(f))
    except ValueError:
        continue
    k = int(d.get("episode", 0))
    if k < a.since:
        continue
    lines = ep_lines.get(k, [])
    stops = sum(1 for ln in lines if " STOP " in ln)
    servo = [ln for ln in lines if "servo it" in ln]
    first_err = None
    if servo:
        m = re.search(r"err \[(-?[\d.]+), (-?[\d.]+)\]", servo[0])
        if m:
            first_err = round((float(m.group(1)) ** 2 + float(m.group(2)) ** 2) ** 0.5, 1)
    pre = d.get("pre_close") or {}
    st = d.get("station") or {}
    t = d.get("timings") or {}
    flags = []
    if d.get("decision_conf") == "no_evidence":
        flags.append("NO_EVIDENCE")
    if stops:
        flags.append(f"STOPS{stops}")
    if st and st.get("method") not in (None, "appeared"):
        flags.append(f"STATION_{st.get('method')}")
    if pre and pre.get("method") not in (None, "diff_first_frame"):
        flags.append(f"PRE_{pre.get('method')}")
    vs = [v.get("verdict") for v in [pre, st, d.get("first_frame_view") or {}] if v and v.get("verdict")]
    if "CONFIDENT_DEFECT" in vs and "CONFIDENT_GOOD" in vs:
        flags.append("VERDICTS_DISAGREE")
    if first_err and first_err > 30:
        flags.append(f"SERVO0_{first_err}")
    if t.get("cycle_s") and t["cycle_s"] > 90:
        flags.append("SLOW")
    if not d.get("success"):
        flags.append("FAIL:" + str(d.get("failure"))[:60])
    rows.append({"ep": k, "ok": d.get("success"), "label": d.get("label"), "plate": d.get("plate"), "conf": d.get("decision_conf"),
                 "colour": d.get("colour"), "shape": d.get("shape"), "pre": f"{pre.get('method', '-')}/{pre.get('verdict', '-')}",
                 "station": f"{st.get('method', '-')}/{st.get('verdict', '-')}/{st.get('changed_frac', '-')}",
                 "cycle": t.get("cycle_s"), "pick": t.get("pick_s"), "inspect": t.get("inspect_s"), "place": t.get("place_s"),
                 "servo_n": len(servo), "servo0_err": first_err, "grip": d.get("grip_closed"), "flags": flags})
    flags_all += flags

for r in rows:
    print(f"ep{r['ep']:>3} ok={str(r['ok'])[0]} {r['label']}->{r['plate']} {r['conf']} {r['colour']} {r['shape']} pre={r['pre']} "
          f"st={r['station']} t={r['cycle']}(p{r['pick']} i{r['inspect']} pl{r['place']}) servo={r['servo_n']} e0={r['servo0_err']} "
          f"grip={r['grip']} {' '.join(r['flags'])}")
ok = [r for r in rows if r["ok"]]
cyc = [r["cycle"] for r in ok if r["cycle"]]
print(f"TOTAL episodes={len(rows)} success={len(ok)} no_evidence={sum('NO_EVIDENCE' in r['flags'] for r in rows)} "
      f"stops={sum(int(f[5:]) for r in rows for f in r['flags'] if f.startswith('STOPS'))} "
      f"cycle_median={statistics.median(cyc) if cyc else None} cycle_min={min(cyc) if cyc else None}")
for ln in log:
    if "jaw check" in ln or "scan map" in ln and ln is log[-1]:
        pass
jc = [ln for ln in log if "jaw check" in ln]
sm = [ln for ln in log if "scan map" in ln]
print("jaw:", jc[-1][:260] if jc else None)
print("scan:", sm[-1][:160] if sm else None)
