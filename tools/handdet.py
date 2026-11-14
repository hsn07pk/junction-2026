"""Hand detector shared by the hand watchdog (hand_watch.py) and the sort loop's clear check (sort_loop.hands_seen).

One source of truth: the 18:03Z tight e-stop loop happened because the watchdog and the loop's clear check used
different rules (the loop ignored the wrist view, the watchdog did not), so the loop kept clearing an e-stop the
watchdog kept setting.

Model (fitted on the 41 labelled e-stop images 16:19-18:13Z, manual visual labels in
work/handdet/labels_features.tsv; see docs/PICK-ANYWHERE.md):
- skin colour: YCrCb Cr 138-170, Cb 90-125 AND HSV hue 5-20, saturation 45-150, value >= 90 (unchanged), plus a
  per-pixel value floor of 130 in the close view (warm speckles on the dark mat had median value 103-108; hands
  167-249);
- close view: pixels in the gripper zone (red jaw components touching the top edge, dilated 25 px, convex hull) are
  ignored -- a white brick carried between the jaws has skin hue 7 / saturation 69 and enters from the top edge just
  like a hand (18:10:59Z false stop);
- a blob is a hand when it is large, or medium-sized and touching the image edge (people reach in from outside the
  frame; every labelled hand touches an edge, no labelled brick false positive does except the carried brick).

A false stop is safe; a missed hand is not: the top view keeps its plain area rule (no false positive in the data).
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

# x0, y0, x1, y1. The top view must start at y0 = 0: people reach in over the far edge next to the arm, and a 30 px
# crop missed 6 of the 17 labelled top-view hands and 25 hands in the 17:03Z recording (no extra false stop).
ROI = {"top": (175, 0, 640, 345), "close": (0, 0, 640, 480), "wrist": (0, 0, 640, 480)}
# per view: (area that alone means a hand, smaller area that means a hand when the blob touches the frame edge)
AREA_ALONE = {"top": 900, "close": 6000, "wrist": 20000}
AREA_EDGE = {"top": 900, "close": 2000, "wrist": 20000}
PERSIST = {"top": 1, "close": 2, "wrist": 2}  # consecutive frames before an e-stop
CLOSE_VALUE_FLOOR = 130


def skin_mask(img: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((ycc[..., 1] >= 138) & (ycc[..., 1] <= 170) & (ycc[..., 2] >= 90) & (ycc[..., 2] <= 125)
         & (h >= 5) & (h <= 20) & (s >= 45) & (s <= 150) & (v >= 90)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def gripper_zone(img: np.ndarray, margin: int = 25) -> np.ndarray:
    """Close view: convex hull of the red jaw components that touch the top edge, dilated by `margin` px."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((H <= 8) | (H >= 170)) & (S >= 100) & (V >= 50)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    zone = np.zeros(red.shape, np.uint8)
    pts = []
    for i in range(1, n):
        if st[i, 1] <= 3 and st[i, 4] > 800:
            ys, xs = np.nonzero(lab == i)
            pts.append(np.stack([xs, ys], 1))
    if not pts:
        return zone.astype(bool)
    hull = cv2.convexHull(np.concatenate(pts).astype(np.int32))
    cv2.fillConvexPoly(zone, hull, 1)
    return cv2.dilate(zone, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)).astype(bool)


def features(img: np.ndarray, view: str, holding: bool = False) -> dict:
    """Largest skin blob in the view's ROI: area, edge contact, fill, median hue/saturation/value.

    holding: the gripper is closed on a brick. In the close view every skin component that reaches into the gripper
    zone is then the carried brick (18:10:59Z: a white brick between the jaws, hue 7 / sat 70, touching the top
    edge), so the whole component is dropped, not just its pixels inside the zone."""
    x0, y0, x1, y1 = ROI[view]
    roi = img[y0:y1, x0:x1]
    m = skin_mask(roi) > 0
    if view == "close":
        v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 2]
        m &= v >= CLOSE_VALUE_FLOOR
        zone = gripper_zone(roi)
        if holding and zone.any():
            n0, lab0 = cv2.connectedComponents(m.astype(np.uint8))
            for k in np.unique(lab0[zone & m]):
                if k:
                    m &= lab0 != k
        m &= ~zone
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 1:
        return {"area": 0, "edge": "", "fill": 0.0, "hue": 0, "sat": 0, "val": 0}
    i = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h, a = (int(t) for t in st[i])
    Hh, Ww = m.shape
    edge = "".join(s for s, c in (("T", y <= 1), ("B", y + h >= Hh - 1), ("L", x <= 1), ("R", x + w >= Ww - 1)) if c)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[lab == i]
    return {"area": a, "edge": edge, "bbox": [x, y, w, h], "fill": round(a / float(w * h), 3), "hue": int(np.median(hsv[:, 0])),
            "sat": int(np.median(hsv[:, 1])), "val": int(np.median(hsv[:, 2]))}


FAR_BAND = 0.30  # close view: top fraction of the frame that lies beyond the table's far edge


def far_band_only(f: dict, height: int) -> bool:
    """Close-view blob that touches only the top edge and stays in the top FAR_BAND of the frame: someone behind the
    table, beyond the far edge. Such a blob alone does not stop the arm; the top camera must confirm a hand."""
    if not f.get("bbox") or f.get("edge") not in ("T", "TL", "TR"):
        return False
    x, y, w, h = f["bbox"]
    return y + h <= FAR_BAND * height


def is_hand_features(f: dict, view: str) -> bool:
    return f["area"] >= AREA_ALONE[view] or (f["area"] >= AREA_EDGE[view] and bool(f["edge"]))


def is_hand(img: np.ndarray, view: str, holding: bool = False) -> tuple[bool, dict]:
    f = features(img, view, holding)
    return is_hand_features(f, view), f


HEARTBEAT = os.path.join(os.environ.get("JUNCTION_RUN_ROOT") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "run"), "hand_watch_alive")  # hand_watch.py writes it on every pass
HEARTBEAT_MAX_AGE = 3.0  # s: a missing, unreadable, future or older heartbeat means the watchdog is not running


def write_heartbeat(path: str | None = None) -> None:
    """Write the epoch time to the heartbeat file: temp file, then rename, so a reader never sees a partial file."""
    path = path or HEARTBEAT
    with open(path + ".tmp", "w") as f:
        f.write(f"{time.time()!r}\n")  # full precision: a stamp rounded up would read as 'in the future' at once
    os.replace(path + ".tmp", path)


def heartbeat_fresh(path: str | None = None) -> bool:
    """True when the heartbeat was written at most HEARTBEAT_MAX_AGE seconds ago."""
    try:
        with open(path or HEARTBEAT) as f:
            t = float(f.read())
    except (OSError, ValueError):
        return False
    return 0.0 <= time.time() - t <= HEARTBEAT_MAX_AGE
