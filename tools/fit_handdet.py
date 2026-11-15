"""Refit the hand detector's area rules (config/hand_model.json) from labelled frames, safety first.

Labels: a TSV with a header and the columns path, view (top | close | wrist), label (hand | none) and optionally
holding (1 when the gripper held a brick). Paths are relative to the TSV's folder. Add every new false stop (label
none) and every hand the watchdog should have seen (label hand) to it.

Per view, the edge rule's area is the largest value that still catches every labelled hand touching the frame edge,
and the area that alone means a hand the largest value that catches every other labelled hand; both are then
lowered by the margin (a hand never seen in the labels may be smaller). A missed hand is unsafe and a false stop is
only slow, so the fit never trades a labelled hand for fewer false stops. Prints the false stops of the current and
the refitted rules on the labelled frames; --write replaces the model file (the previous one is kept as .prev).
Usage: python -I -B fit_handdet.py LABELS.tsv [--margin 0.8] [--write]"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handdet as hd  # noqa: E402


def load(tsv: str) -> list:
    base = os.path.dirname(os.path.abspath(tsv))
    out = []
    with open(tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            img = cv2.imread(os.path.join(base, row["path"]))
            if img is None or row["view"] not in hd.ROI or row["label"] not in ("hand", "none"):
                continue
            holding = str(row.get("holding", "0")).strip() in ("1", "true", "yes")
            out.append({"path": row["path"], "view": row["view"], "hand": row["label"] == "hand",
                        "f": hd.features(img, row["view"], holding)})
    return out


def fit(rows: list, margin: float) -> tuple[dict, dict]:
    alone, edge = dict(hd.AREA_ALONE), dict(hd.AREA_EDGE)
    for v in hd.ROI:
        hands = [r["f"] for r in rows if r["view"] == v and r["hand"]]
        if not hands:
            continue  # no labelled hand in this view: keep its rules
        at_edge = [f["area"] for f in hands if f["edge"]]
        inside = [f["area"] for f in hands if not f["edge"]]
        if at_edge:
            edge[v] = int(margin * min(at_edge))
        if inside:
            alone[v] = int(margin * min(inside))
        alone[v] = max(alone[v], edge[v])
    return alone, edge


def errors(rows: list, alone: dict, edge: dict) -> dict:
    out = {"missed_hands": [], "false_stops": []}
    for r in rows:
        f, v = r["f"], r["view"]
        hit = f["area"] >= alone[v] or (f["area"] >= edge[v] and bool(f["edge"]))
        if r["hand"] and not hit:
            out["missed_hands"].append(r["path"])
        if not r["hand"] and hit:
            out["false_stops"].append(r["path"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--margin", type=float, default=0.8, help="fraction of the smallest labelled hand kept as the rule")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows = load(a.labels)
    print(f"{len(rows)} labelled frames ({sum(r['hand'] for r in rows)} hands)")
    cur = errors(rows, hd.AREA_ALONE, hd.AREA_EDGE)
    alone, edge = fit(rows, a.margin)
    new = errors(rows, alone, edge)
    print(json.dumps({"current": {"area_alone": hd.AREA_ALONE, "area_edge": hd.AREA_EDGE,
                                  "missed_hands": len(cur["missed_hands"]), "false_stops": len(cur["false_stops"])},
                      "refitted": {"area_alone": alone, "area_edge": edge, "missed_hands": len(new["missed_hands"]),
                                   "false_stops": len(new["false_stops"])}}, indent=1))
    if a.write:
        if new["missed_hands"]:
            sys.exit(f"not written: the refitted rules miss labelled hands {new['missed_hands']}")
        m = dict(hd.M)
        m.update(area_alone=alone, area_edge=edge,
                 provenance=f"{hd.M.get('provenance', '')}; refitted {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                            f"from {len(rows)} labelled frames ({a.labels}), margin {a.margin}: missed hands 0, false "
                            f"stops {len(new['false_stops'])} (were {len(cur['false_stops'])})")
        shutil.copy(hd.MODEL_F, hd.MODEL_F + ".prev")
        with open(hd.MODEL_F + ".tmp", "w") as f:
            json.dump(m, f, indent=1)
        os.replace(hd.MODEL_F + ".tmp", hd.MODEL_F)
        print("wrote", hd.MODEL_F)


if __name__ == "__main__":
    main()
