"""Locate the brick a close-camera frame is about by comparing it with the same camera's frame from the start of the
episode, taken with the arm at the scan pose (out of the close camera's way). Nothing here depends on fixed pixel
positions, so it keeps working after the close camera is moved.

- appeared brick: present in `after` but not in `before` (the brick put down at the station)
- vanished brick: present in `before` but not in `after` (the brick that was picked)
- jaw tips: red parts present while hovering but not in the episode's first frame (the bricks did not move)
- ClosePlaneMap: table-plane homography (mm -> px) fitted from the pairs the loop records, to predict where a brick
  at a known table position appears in the close view
"""
import json
import os

import cv2
import numpy as np

DIFF_THR = 30          # max-channel absolute difference (0-255) after a 5x5 blur counts as "changed"
MIN_FRAC = 0.45        # a brick box must be at least this much changed to count as appeared/vanished
JAW_MIN_AREA = 800     # px; smaller new-red specks are not jaws


def changed_mask(a, b, thr=DIFF_THR):
    """Uint8 mask of pixels that differ between two BGR frames of the same camera."""
    da = cv2.GaussianBlur(a, (5, 5), 0).astype(np.int16)
    db = cv2.GaussianBlur(b, (5, 5), 0).astype(np.int16)
    m = (np.abs(da - db).max(axis=2) >= thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def box_fraction(mask, bbox):
    x, y, w, h = (int(v) for v in bbox)
    x0, y0 = max(x, 0), max(y, 0)
    roi = mask[y0:y + h, x0:x + w]
    return float(roi.mean()) if roi.size else 0.0


def pick_changed(bricks, changed, min_frac=MIN_FRAC):
    """Brick record (an entry of a segmenter's 'bricks' list, with 'bbox' = x, y, w, h) whose box is most covered by
    changed pixels. Returns (record or None, its fraction, the runner-up fraction)."""
    scored = sorted(((box_fraction(changed, r["bbox"]), i) for i, r in enumerate(bricks)), reverse=True)
    if not scored:
        return None, 0.0, 0.0
    best, second = scored[0][0], (scored[1][0] if len(scored) > 1 else 0.0)
    return (bricks[scored[0][1]] if best >= min_frac else None), round(best, 3), round(second, 3)


def drop_top_edge(changed, rows=2):
    """Remove changed regions connected to the top image edge: people and arms beyond the table enter the close view
    from the top, a brick on the mat never touches it."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(changed)
    out = changed.copy()
    for i in range(1, n):
        if st[i, cv2.CC_STAT_TOP] < rows:
            out[lab == i] = 0
    return out


# colour the wrist camera reports for the centred brick -> colours the close-view inspector may call the same brick
COMPATIBLE = {"white": {"white", "cream"}, "yellow": {"yellow", "cream"}, "red": {"red", "pink", "orange"},
              "orange": {"orange", "red"}, "pink": {"pink", "red"}, "blue": {"blue"}, "green": {"green"}}


def colour_ok(close_colour, wrist_colour):
    """True when the close-view colour can be the brick the wrist camera centred (unknown on either side: True)."""
    if not wrist_colour or not close_colour or wrist_colour not in COMPATIBLE:
        return True
    return close_colour in COMPATIBLE[wrist_colour]


def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h <= 8) | (h >= 170)) & (s >= 100) & (v >= 50)).astype(np.uint8)


def jaw_tips_by_diff(hover, before, min_area=JAW_MIN_AREA):
    """Tips of the gripper jaws in a close frame taken while hovering: red parts that were not red in the episode's
    first frame. Returns up to two (x, y) tips, each the mean x of the lowest 6 rows of one of the two largest parts,
    sorted left to right."""
    new = red_mask(hover) & (1 - cv2.dilate(red_mask(before), np.ones((9, 9), np.uint8)))
    new = cv2.morphologyEx(new, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(new)
    parts = sorted((i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area),
                   key=lambda i: -st[i, cv2.CC_STAT_AREA])[:2]
    tips = []
    for i in parts:
        ys, xs = np.nonzero(lab == i)
        yb = ys.max()
        tips.append((float(xs[ys >= yb - 6].mean()), float(yb)))
    return sorted(tips)


def brick_below(bricks, x, y, max_dx=90.0):
    """Nearest brick whose centre is below image row y and within max_dx of column x (side view: the brick under the
    jaws is below the tips). Returns (record or None, distance)."""
    best = None
    for r in bricks:
        cx, cy = r["centre_px"]
        if cy <= y or abs(cx - x) > max_dx:
            continue
        d = float(np.hypot(cx - x, cy - y))
        if best is None or d < best[0]:
            best = (d, r)
    return (best[1], round(best[0], 1)) if best else (None, None)


def nearest(bricks, x, y, max_d):
    best = None
    for r in bricks:
        d = float(np.hypot(r["centre_px"][0] - x, r["centre_px"][1] - y))
        if d <= max_d and (best is None or d < best[0]):
            best = (d, r)
    return (best[1], round(best[0], 1)) if best else (None, None)


class ClosePlaneMap:
    """Table plane (mm, robot base frame) -> close-camera pixel homography, fitted with RANSAC from recorded pairs.
    A camera move shows up as large residuals on the newest pairs; refit() keeps only pairs consistent with the
    newest ones."""

    def __init__(self, path, min_pairs=6, ransac_px=12.0):
        self.path, self.min_pairs, self.ransac_px = path, min_pairs, ransac_px
        self.H, self.median_px, self.n_used = None, None, 0

    def pairs(self):
        if not os.path.exists(self.path):
            return []
        out = []
        for line in open(self.path):
            try:
                p = json.loads(line)
                out.append((float(p["x_mm"]), float(p["y_mm"]), float(p["u"]), float(p["v"])))
            except (ValueError, KeyError):
                continue
        return out

    def add(self, x_mm, y_mm, u, v, kind):
        with open(self.path, "a") as f:
            f.write(json.dumps({"x_mm": round(x_mm, 1), "y_mm": round(y_mm, 1), "u": round(u, 1), "v": round(v, 1),
                                "kind": kind}) + "\n")

    def refit(self, newest=20):
        p = np.array(self.pairs()[-newest:], dtype=np.float64)
        self.H = None
        if len(p) < self.min_pairs:
            return None
        H, inl = cv2.findHomography(p[:, :2], p[:, 2:], cv2.RANSAC, self.ransac_px)
        if H is None:
            return None
        inl = inl.ravel().astype(bool)
        if inl.sum() < self.min_pairs:
            return None
        res = np.linalg.norm(self._apply(H, p[inl, :2]) - p[inl, 2:], axis=1)
        self.H, self.median_px, self.n_used = H, float(np.median(res)), int(inl.sum())
        return self.median_px

    @staticmethod
    def _apply(H, xy):
        q = np.hstack([xy, np.ones((len(xy), 1))]) @ H.T
        return q[:, :2] / q[:, 2:3]

    def predict(self, x_mm, y_mm):
        if self.H is None:
            return None
        u, v = self._apply(self.H, np.array([[x_mm, y_mm]], dtype=np.float64))[0]
        return float(u), float(v)
