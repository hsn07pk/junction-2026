"""P2-B-DET2 E2: per-brick verdicts for DUPLO bricks lying on the black mat (CPU only, no pretrained weights).

v3 (P2-B-DET2 V1 verifier-fixer, after D-075/D-079): a hand needs an in-mat, non-brick-shaped skin blob that touches the
  border or is large; two-colour blobs are split into colour pieces before any shape call; a close-view blob with aspect
  >= 1.5 is never a 2x2 call; tape evidence still gives CONFIDENT_DEFECT on merged or split blobs.
v2 (after the D-068 review):
  - top view: static background masked (left plate-rim strip x < 172, bottom grey aluminium rail y >= 338);
  - merged blobs (two saturated hues >= 36 deg apart, white plus saturated colour, or area > 2.9 calibrated 2x2)
    get shape 'unknown'; merged colours are never 2x4 evidence;
  - close view: tape = dark, low-saturation, textured area fraction inside the brick silhouette (D-068);
    thresholds in models/tape_thresholds.json, written by calib_tape.py from shared/inspection-labels.tsv.

Frame API:   scan(top_rgb, close_rgb=None) -> list of brick dicts (one per brick per view).
             scan_full(...) also returns hands, plates, border and rejection counts.
             raises HandsPresent when skin-hued blobs are in the mat ROI of the top view (refuse; route nothing).
Single brick (LOOP-CONTRACT section 3): Inspector.inspect_rgb(rgb, camera_key=None), Inspector.locate_brick(rgb).
find_target(bricks, colour='red') picks the brick the team's text refers to.
Verdicts: CONFIDENT_DEFECT (route GREEN directly), CONFIDENT_GOOD (reported; still inspected), UNCERTAIN (inspect).
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOP_X0 = 172  # static background: left plate-rim strip (x < 172) in the top view
TOP_Y1 = 338  # static background: bottom grey aluminium rail (y >= 338) in the top view
TOP_ROI = (TOP_X0, 0, 640, TOP_Y1)  # x0, y0, x1, y1 of the usable black mat in the top view
CLOSE_ROI = (120, 40, 640, 480)
ROBOT_POLY = np.array([[322, 0], [425, 0], [425, 122], [404, 122], [404, 146], [380, 146], [380, 122],
                       [358, 122], [358, 146], [330, 146], [324, 120]], np.int32)  # SO-101 at home, top view
A2X2_TOP = 1250.0  # px footprint of a 2x2 in the top view (calibrated on samples, see report)
A2X2_CLOSE = 5000.0  # px footprint of a 2x2 in the close view (approximate; close shape is not used)
HANDS_MIN_AREA = 900
HANDS_MAX_S = 130
PLATE_TOP_AREA = 12000  # plates are 19000-21500 px in the top view (live still 153426); a 3-brick cluster is ~5000 (V1)
PLATE_CLOSE_AREA = 14000
MIN_AREA_TOP = 350
MIN_AREA_CLOSE = 1500
CONF_SHAPE = 0.8
MERGED_AREA_REL = 2.9  # a blob larger than 2.9 calibrated 2x2 footprints never gets a shape call
HANDS_MIN_IN_MAT = 300  # a skin blob counts as a hand only if it puts this many px on the mat (V1)
HANDS_INTERIOR_MIN = 2500  # an interior, non-brick-shaped skin blob this large is still a hand (fail-safe, V1)
ELONGATED_CLOSE = 1.5  # close-view aspect at or above this is never a single 2x2 (merged pair, 2x4, brick on its side)
SPLIT_MIN_CHROMA = 60  # chroma pixels needed before a two-colour split is tried (V1)
SPLIT_SEP_UNITS = 12  # hue separation of the two colour clusters, OpenCV units (24 degrees) (V1)
CONCAVE_MIN_DEPTH = 4.0  # px: outline defects shallower than this are rounded corners, not a neck between bricks (V1)
CONCAVE_TOP_K = 4  # deepest outline defects tried as the two ends of a cut (V1)
COMPACT_FILL = 0.85  # a blob filling this share of its minAreaRect (aspect <= 2.8) is one brick and is never cut (V1)
PLATE_CHROMA_SHARE = 0.3  # a big close-view blob with this share of blue or green chroma is a plate, not a brick cluster (V1)
TAPE_V = (50, 130)  # dark grey tape: above the black mat, below the bright brick faces
TAPE_S_MAX = 90  # low saturation
TEX_SD = 4.0  # 5x5 local std of V above this counts as textured
TAPE_ERODE_K = 7  # F2: silhouette erosion ellipse; 7 = 3 px as in V1 and E2
TAPE_MEDIAN = 3  # F2: 3x3 median on V before the tape tests (JPEG ringing; 0 = off)
TAPE_OPEN = 3  # F2: 3x3 opening of the texture mask (removes 1-2 px ringing lines; 0 = off)
TAPE_MIN_PATCH = 30  # F2: smallest connected texture patch in px (scattered ringing; 0 = off)
TOP_MEDIAN = 3  # F2: 3x3 median on the top foreground mask (JPEG flips small footprints; 0 = off)
NOT2X2_FILL_CLEAR = 0.85  # F2: a 2x4-like top blob filling this share of its minAreaRect has a clear footprint
NOT2X2_CLEAR_CONF = 0.85  # F2: confidence of a clear not_2x2 footprint (>= CONF_SHAPE)
CLOSE_FRAGMENT_NO_CD = True  # F2: a split, elongated close piece (cut fragment) never gives a tape defect call
K3 = np.ones((3, 3), np.uint8)
ELL7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
THR_PATH = os.path.join(HERE, "models", "tape_thresholds.json")
BANK_PATH = os.path.join(HERE, "models", "tape_bank.npz")  # PatchCore-style kNN bank: recorded, not used for verdicts


class HandsPresent(RuntimeError):
    """Raised by scan() when hands are visible: the frame must not be used for routing."""


def _split(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)


def _circ_median(h):
    return float((np.median((h + 90) % 180) - 90) % 180)


def colour_name(h, s, v):
    if s < 25 and v > 150:  # V1: a washed-out yellow brick on the live top view has median S 58 (was called white)
        return "white"
    if h >= 150 or h < 6:
        return "red" if s >= 140 else "pink"
    if 6 <= h < 14:
        return "orange" if s >= 120 else "cream"
    if 14 <= h < 40:  # V1: the live close-view yellow brick has median hue 18 with S 121 (was called orange)
        return "yellow"
    if 40 <= h < 90:
        return "green"
    if 90 <= h < 135:
        return "blue"
    return "other"


def _components(mask, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            out.append((lab == i, a, stats[i]))
    return out


def _roi(shape, box):
    x0, y0, x1, y1 = box
    roi = np.zeros(shape, bool)
    roi[y0:y1, x0:x1] = True
    return roi


def fill_holes(m):
    """Fill background regions enclosed by the mask (flood fill from the padded border)."""
    pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    pad[1:-1, 1:-1] = m.astype(np.uint8) * 255
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8), (0, 0), 128)
    return m | (flood[1:-1, 1:-1] == 0)


def silhouette_hull(m):
    """Convex hull of the bright mask (the brick outline is convex in both views): encloses dark side-face tape."""
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    if cnts:
        pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(out, [cv2.convexHull(pts)], 1)
    return out.astype(bool)


def merged_flags(h, s, v, m):
    """True when one blob holds two colour populations: two saturated hues >= 36 deg apart (>= 15% of the
    saturated pixels), or true white (>= 25%) plus strongly coloured pixels (>= 25%)."""
    n = int(m.sum())
    sat = m & (s > 85) & (v > 90)
    n_sat = int(sat.sum())
    strong = m & (s > 100) & (v > 90)  # clearly coloured: pale yellow highlights do not count
    white = m & (s < 45) & (v > 170)  # true white only
    if int(strong.sum()) >= 0.25 * n and int(white.sum()) >= 0.25 * n:
        return True, "white_plus_colour"
    if n_sat >= 60:
        hist = np.bincount((h[sat] // 9).astype(np.int64), minlength=20)  # 18-degree bins on the hue circle
        p = int(np.argmax(hist))
        dist = np.abs(np.arange(20) - p)
        dist = np.minimum(dist, 20 - dist)
        if hist[dist >= 2].sum() >= 0.15 * n_sat:
            return True, "two_hues"
    return False, ""


def shape_rule(aspect, rel, fill=0.0):
    """Footprint shape from the minAreaRect aspect ratio and the area relative to a calibrated 2x2."""
    if 0.6 <= rel <= 1.6 and aspect <= 1.2:
        return "2x2", float(np.clip(1.0 - (aspect - 1.0) / 0.2, 0.5, 1.0))
    if 1.5 <= rel <= 3.0 and 1.7 <= aspect <= 2.8:
        conf = float(np.clip(0.6 + (aspect - 1.7) / 0.5, 0.6, 1.0))
        if fill >= NOT2X2_FILL_CLEAR:  # F2: a clean rectangle is a clear footprint, not a borderline shape call
            conf = max(conf, NOT2X2_CLEAR_CONF)
        return "not_2x2", conf
    # aspect > 2.8 is cables, clamps or edge fragments in the top view, and a brick lying on its side
    # (footprint aspect about 1.7 with area about 0.6 of a 2x2) falls outside both windows: never a defect call
    return "unknown", 0.4


def _brick_record(h, s, v, m, a, st, a2):
    ys, xs = np.nonzero(m)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    aspect = float(max(rw, rh) / max(min(rw, rh), 1.0))
    hue = _circ_median(h[m])
    sat = float(np.median(s[m]))
    val = float(np.median(v[m]))
    merged, why = merged_flags(h, s, v, m)
    rel = a / a2
    fill = float(a) / max(rw * rh, 1.0)
    if not merged and rel > MERGED_AREA_REL:
        merged, why = True, "oversize"
    if merged:
        shape, conf = "unknown", 0.4
    else:
        shape, conf = shape_rule(aspect, rel, fill)
    return {"area": int(a), "centre_px": [round(float(cx), 1), round(float(cy), 1)], "angle_deg": round(float(ang), 1),
            "bbox": [int(st[0]), int(st[1]), int(st[2]), int(st[3])],
            "footprint_px": [round(float(rw), 1), round(float(rh), 1)], "aspect": round(aspect, 3),
            "colour": colour_name(hue, sat, val), "hsv_med": [round(hue, 1), round(sat, 1), round(val, 1)],
            "shape": shape, "shape_conf": round(conf, 3), "fill": round(fill, 3),
            "merged": merged, "merged_reason": why, "_mask": m}


def brick_like(m, a):
    """Compact rectangle with a 2x2 or 2x4 footprint area. Used only by the hands gate: a brick is never a hand."""
    pts = np.column_stack(np.nonzero(m)[::-1]).astype(np.float32)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    aspect = max(rw, rh) / max(min(rw, rh), 1.0)
    return 0.6 <= a / A2X2_TOP <= 3.0 and aspect <= 2.8 and a / max(rw * rh, 1.0) >= 0.7


def hand_blobs(h, s, v):
    """Skin-hued blobs that refuse the frame (D-075 fix). A blob counts only if it puts at least HANDS_MIN_IN_MAT px on
    the mat and is not brick-shaped, and either touches the image border (hands come in from the frame edge) or is an
    interior blob of at least HANDS_INTERIOR_MIN px. Pink and red bricks on the mat are brick-shaped: no refusal."""
    skin = ((h <= 18) | (h >= 172)) & (s >= 45) & (s <= HANDS_MAX_S) & (v >= 90)
    roi = _roi(h.shape, TOP_ROI)
    H, W = h.shape
    out = []
    for m, a, st in _components(skin, HANDS_MIN_AREA):
        in_mat = int((m & roi).sum())
        if in_mat < HANDS_MIN_IN_MAT or brick_like(m, a):
            continue
        x, y, w, hh = int(st[0]), int(st[1]), int(st[2]), int(st[3])
        touches = x <= 0 or y <= 0 or x + w >= W or y + hh >= H
        if touches or a >= HANDS_INTERIOR_MIN:
            out.append((m, {"area": int(a), "in_mat": in_mat, "bbox": [x, y, w, hh], "touches_border": bool(touches)}))
    return out


def colour_split(h, s, v, m, a2):
    """Two colour clusters in one blob (D-075 fix): hue peaks at least SPLIT_SEP_UNITS apart, each holding at least 20%
    of the chroma pixels. Every pixel goes to the nearer hue centre, the labels are median-smoothed, and each cluster
    must give at least one piece of 0.6 calibrated 2x2 footprints. Returns the pieces (mask, area, stats) or None."""
    ch = m & (s >= 45) & (v >= 80)
    n = int(ch.sum())
    if n < SPLIT_MIN_CHROMA:
        return None
    hist = np.bincount(np.minimum(h[ch].astype(np.int64) // 3, 59), minlength=60)
    p1 = int(np.argmax(hist))
    sep = np.minimum(np.abs(np.arange(60) - p1), 60 - np.abs(np.arange(60) - p1)) * 3
    cand = np.where(sep >= SPLIT_SEP_UNITS)[0]
    if len(cand) == 0:
        return None
    p2 = int(cand[np.argmax(hist[cand])])
    if hist[p1] < 0.2 * n or hist[p2] < 0.2 * n:
        return None
    hue = h.astype(np.int32)
    dist = []
    for c in (p1 * 3 + 1, p2 * 3 + 1):
        dist.append(np.minimum(np.abs(hue - c), 180 - np.abs(hue - c)))
    lab = np.where(m, np.where(dist[0] <= dist[1], 1, 2), 0).astype(np.uint8)
    lab = cv2.medianBlur(lab, 5)
    pieces = []
    for k in (1, 2):
        got = _components((lab == k) & m, 0.6 * a2)
        if not got:
            return None
        pieces.extend(got)
    return pieces


def concave_split(m, a2):
    """Touching bricks cut along a line between two concavities of the outline (V1 live-frame fix). The ends are the
    deepest convexity defects (at least CONCAVE_MIN_DEPTH px); the line must lie inside the blob. A cut is accepted only
    if it leaves at least two pieces and no fragment below 0.6 calibrated 2x2 footprints. None when no cut qualifies.
    Replaces the distance-peak watershed (V1 first pass), which over-seeded one 2x4 into three pieces."""
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    if max(rw, rh) / max(min(rw, rh), 1.0) <= 2.8 and float(m.sum()) / max(rw * rh, 1.0) >= COMPACT_FILL:
        return None  # a compact rectangle is one brick (2x2 or 2x4): its tape or notches are never a neck (V1)
    defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
    if defects is None:
        return None
    cands = []
    for _, _, far, depth in defects.reshape(-1, 4):
        if depth / 256.0 >= CONCAVE_MIN_DEPTH:
            x0, y0 = cnt[far][0]
            cands.append((depth / 256.0, (int(x0), int(y0))))
    cands.sort(key=lambda t: -t[0])
    cands = cands[:CONCAVE_TOP_K]
    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            p, q = cands[i][1], cands[j][1]
            line = np.zeros(m.shape, np.uint8)
            cv2.line(line, p, q, 1, 2)
            n = max(int(np.hypot(p[0] - q[0], p[1] - q[1])), 2)
            xs = np.linspace(p[0], q[0], n).astype(int)
            ys = np.linspace(p[1], q[1], n).astype(int)
            if float(m[ys, xs].mean()) < 0.9:
                continue
            k, lab, stats, _ = cv2.connectedComponentsWithStats((m & (line == 0)).astype(np.uint8), 8)
            parts = [(lab == c, int(stats[c, cv2.CC_STAT_AREA]), stats[c]) for c in range(1, k)
                     if int(stats[c, cv2.CC_STAT_AREA]) >= 20]
            if len(parts) >= 2 and all(pa >= 0.6 * a2 for _, pa, _ in parts):
                length = float(np.hypot(p[0] - q[0], p[1] - q[1]))
                if best is None or length < best[0]:  # the shortest valid cut is the neck; a diagonal cut across a brick is long
                    best = (length, parts)
    return None if best is None else best[1]


def concave_pieces(m, a2, depth=0):
    """Concave_split applied again to every piece above 1.5 calibrated 2x2 footprints (at most three cuts deep), so a
    red-white-white cluster is cut into its three bricks. None when the blob is not split at all."""
    cut = concave_split(m, a2)
    if cut is None:
        return None
    out = []
    for pm, pa, pst in cut:
        sub = concave_pieces(pm, a2, depth + 1) if (pa > 1.5 * a2 and depth < 3) else None
        out.extend(sub if sub is not None else [(pm, pa, pst)])
    return out


def pieces_of(h, s, v, m, a, st, a2):
    """Bricks of one blob: its colour pieces when it holds two hues, else its concavity pieces (concave_pieces) when it
    is larger than 1.5 calibrated 2x2 footprints, else the blob itself. Each entry is (mask, area, stats, split).
    The top-view rail stays at TOP_Y1 = 338: the live rail-band false bricks came from an adaptive cut (V1 first pass)."""
    pcs = colour_split(h, s, v, m, a2)
    if pcs is None and a > 1.5 * a2:
        pcs = concave_pieces(m, a2)
    if pcs is None:
        return [(m, a, st, False)]
    return [(pm, pa, pst, True) for pm, pa, pst in pcs]


def segment_top(bgr):
    h, s, v = _split(bgr)
    roi = _roi(h.shape, (TOP_X0, 0, h.shape[1], TOP_Y1))
    robot = np.zeros(h.shape, np.uint8)
    cv2.fillPoly(robot, [ROBOT_POLY], 1)
    hand_px, hands = np.zeros(h.shape, bool), []
    for m, rec in hand_blobs(h, s, v):
        hands.append(rec)
        hand_px |= cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    fg = roi & (v > 70) & (robot == 0) & ~hand_px
    if TOP_MEDIAN:  # F2: JPEG ringing flipped small white footprints between 2x2 and not_2x2; a median pass fixes that
        fg = cv2.medianBlur(fg.astype(np.uint8) * 255, TOP_MEDIAN) > 127
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, K3).astype(bool)
    bricks, plates = [], 0
    for m, a, st in _components(fg, MIN_AREA_TOP):
        if a > PLATE_TOP_AREA:
            plates += 1
            continue
        for pm, pa, pst, split in pieces_of(h, s, v, m, a, st, A2X2_TOP):
            rec = _brick_record(h, s, v, pm, pa, pst, A2X2_TOP)
            rec["view"] = "top"
            rec["tape_score"] = None  # tape is judged in the close view only
            rec["split"] = split
            bricks.append(rec)
    return {"bricks": bricks, "hands": hands, "plates_rejected": plates}


def close_tape(v, s, m):
    """D-068 close-view tape rule: dark, low-saturation, textured pixels inside the brick silhouette.
    Silhouette = convex hull of the bright mask, eroded by 3 px (drops the brick-to-mat edge band).
    Pixels of the bright mask are not counted."""
    sil = silhouette_hull(m)
    er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TAPE_ERODE_K, TAPE_ERODE_K))
    inner = cv2.erode(sil.astype(np.uint8), er).astype(bool)
    area = max(int(inner.sum()), 1)
    vu = cv2.medianBlur(v.astype(np.uint8), TAPE_MEDIAN) if TAPE_MEDIAN else v.astype(np.uint8)
    vf = vu.astype(np.float32)
    mu = cv2.blur(vf, (5, 5))
    sd = np.sqrt(np.maximum(cv2.blur(vf * vf, (5, 5)) - mu * mu, 0.0))
    dark = inner & ~m & (vu >= TAPE_V[0]) & (vu < TAPE_V[1]) & (s < TAPE_S_MAX)
    tex = dark & (sd > TEX_SD)
    if TAPE_OPEN:
        tex = cv2.morphologyEx(tex.astype(np.uint8), cv2.MORPH_OPEN,
                               np.ones((TAPE_OPEN, TAPE_OPEN), np.uint8)).astype(bool)
    if TAPE_MIN_PATCH:  # F2: JPEG ringing is many small patches; tape is one connected dark patch
        n, lab, stats, _ = cv2.connectedComponentsWithStats(tex.astype(np.uint8), 8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= TAPE_MIN_PATCH
        tex = keep[lab]
    return {"sil_area": area, "dark_frac": round(float(dark.sum()) / area, 4),
            "tape_frac": round(float(tex.sum()) / area, 4)}


def segment_close(bgr):
    h, s, v = _split(bgr)
    H, W = h.shape
    roi = _roi(h.shape, CLOSE_ROI)
    fg = cv2.morphologyEx((roi & (((s > 85) & (v > 90)) | ((v > 170) & (s < 70)))).astype(np.uint8), cv2.MORPH_OPEN, K3).astype(bool)
    bricks, rejected, partial = [], {"plate_or_gripper": 0, "small": 0}, 0
    for m, a, st in _components(fg, MIN_AREA_CLOSE):
        x, y, w, hh = int(st[0]), int(st[1]), int(st[2]), int(st[3])
        cut = x <= 1 or y <= 1 or x + w >= W - 1 or y + hh >= H - 1  # cut by the frame: tape only, never GOOD (V1)
        partial += int(cut)
        if a > PLATE_CLOSE_AREA:  # V1: a big blob is cut only at outline necks, and never when it holds plate colour (blue or green)
            chroma_hue = h[m & (s >= 45) & (v >= 80)].astype(np.int32)
            plate_share = float(((chroma_hue >= 40) & (chroma_hue < 135)).mean()) if chroma_hue.size else 0.0
            necks = None if plate_share >= PLATE_CHROMA_SHARE else concave_pieces(m, A2X2_CLOSE)
            if necks is None:
                rejected["plate_or_gripper"] += 1
                continue
            pieces = [(pm, pa, pst, True) for pm, pa, pst in necks]
        else:  # V1: blobs up to the plate limit are not split (the colour and outline splits made extra close-view 'defects')
            pieces = [(m, a, st, False)]
        for pm, pa, pst, split in pieces:
            if pa > PLATE_CLOSE_AREA:
                rejected["plate_or_gripper"] += 1
                continue
            rec = _brick_record(h, s, v, pm, pa, pst, A2X2_CLOSE)
            if rec["colour"] == "green":  # green plates are never bricks in this set (V1)
                rejected["plate_or_gripper"] += 1
                continue
            rec["view"] = "close"
            rec["split"] = split
            rec["partial"] = cut
            if not split and rec["aspect"] >= ELONGATED_CLOSE and not rec["merged"]:
                rec["merged"], rec["merged_reason"] = True, "elongated_close"
            rec["tape_features"] = tape_features(h, s, v, pm, rec["hsv_med"][0], rec["hsv_med"][1])
            rec["tape_features"].update(close_tape(v, s, pm))
            rec["tape_score"] = rec["tape_features"]["tape_frac"]
            bricks.append(rec)
    return {"bricks": bricks, "partial_at_border": partial, "rejected": rejected}


def tape_features(h, s, v, m, hue, sat):
    """Older mask-only features, kept for the glare check and for comparison (not the D-068 rule)."""
    hm, sm, vm = h[m], s[m], v[m]
    dh = np.abs(((hm - hue) + 90) % 180 - 90)
    off = (dh > 14) | (np.abs(sm - sat) > 0.35 * max(sat, 40.0))
    return {"off_frac": round(float(off.mean()), 4),
            "glare_frac": round(float(((vm > 235) & (sm < 70)).mean()), 4),
            "dark_frac_mask": round(float((vm < 80).mean()), 4)}


PATCH = 8
CROP = 96


def patch_desc(bgr, m, st):
    """6-d descriptor per 8x8 patch (circular hue, sat, value, value std, gradient) of a brick crop."""
    x, y, w, hh = int(st[0]), int(st[1]), int(st[2]), int(st[3])
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + hh + pad)
    crop = cv2.resize(bgr[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_AREA)
    mk = cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), (CROP, CROP), interpolation=cv2.INTER_NEAREST).astype(bool)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gm = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1) ** 2)
    out = []
    for i in range(0, CROP, PATCH):
        for j in range(0, CROP, PATCH):
            pm = mk[i:i + PATCH, j:j + PATCH]
            if pm.mean() < 0.6:
                continue
            hp = hsv[i:i + PATCH, j:j + PATCH][pm]
            ang = hp[:, 0] * np.pi / 90.0
            out.append([np.cos(ang).mean(), np.sin(ang).mean(), hp[:, 1].mean() / 255.0, hp[:, 2].mean() / 255.0,
                        hp[:, 2].std() / 255.0, gm[i:i + PATCH, j:j + PATCH][pm].mean() / 255.0])
    return np.asarray(out, np.float32).reshape(-1, 6)


_BANK = {"loaded": False, "patches": None}


def load_thresholds():
    if not os.path.exists(THR_PATH):
        return None
    with open(THR_PATH) as fh:
        return json.load(fh)


def load_bank():
    """(kNN bank patches or None, tape thresholds or None). The kNN bank is kept for the record only."""
    if not _BANK["loaded"]:
        _BANK["loaded"] = True
        if os.path.exists(BANK_PATH):
            _BANK["patches"] = np.load(BANK_PATH)["patches"]
    return _BANK["patches"], load_thresholds()


def patch_score(desc, bank):
    """PatchCore-style score: 95th percentile over patches of the distance to the nearest good patch."""
    if desc.shape[0] == 0 or bank is None or len(bank) == 0:
        return None
    d = np.sqrt(((desc[:, None, :] - bank[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(np.percentile(d, 95))


def triage(rec, thr):
    """Per-brick verdict. Top view: shape only (tape not measured). Close view: tape only (D-068).
    Merged and split blobs never give CONFIDENT_GOOD; tape evidence on them still gives CONFIDENT_DEFECT (V1)."""
    reasons = []
    if rec.get("merged"):
        reasons.append("merged_blob_" + rec.get("merged_reason", ""))
    if rec.get("split"):
        reasons.append("colour_split_piece")
    if rec["view"] == "top":
        clean = rec.get("fill", 0.0) >= NOT2X2_FILL_CLEAR  # F2: a split piece with a clean rectangle footprint is a brick
        if (not rec.get("merged") and (not rec.get("split") or clean) and rec["shape"] == "not_2x2"
                and rec["shape_conf"] >= CONF_SHAPE):  # V1: a fragment cut from a touching cluster never gives a defect call
            return "CONFIDENT_DEFECT", ["not_2x2"]
        reasons.append("tape_not_measured_top_view")
        if rec["shape"] != "2x2":
            reasons.append("shape_" + rec["shape"])
        return "UNCERTAIN", reasons
    score = rec.get("tape_score")
    if score is None or thr is None:
        return "UNCERTAIN", reasons + ["no_tape_threshold"]
    if score >= thr["defect"]:
        if CLOSE_FRAGMENT_NO_CD and rec.get("split") and rec.get("aspect", 0.0) >= ELONGATED_CLOSE:
            return "UNCERTAIN", reasons + ["close_fragment_tape_not_attributable"]  # F2: a cut fragment's tape is a neighbour's
        return "CONFIDENT_DEFECT", ["tape_frac_high"] + reasons
    if rec.get("merged") or rec.get("split") or rec.get("partial"):
        return "UNCERTAIN", reasons
    if score <= thr["good"]:
        return "CONFIDENT_GOOD", ["tape_frac_low"]
    return "UNCERTAIN", reasons + ["tape_frac_mid"]


def scan_full(top_rgb, close_rgb=None):
    """Full frame result: bricks (both views), hands, rejection counts. Raises HandsPresent."""
    top = segment_top(top_rgb)
    if top["hands"]:
        raise HandsPresent(f"hands_present: {len(top['hands'])} skin blob(s)")
    thr = load_thresholds()
    bricks = []
    for rec in top["bricks"]:
        verdict, reasons = triage(rec, thr)
        rec.pop("_mask", None)
        rec.update(verdict=verdict, reasons=reasons, confident=verdict != "UNCERTAIN")
        bricks.append(rec)
    close = {"bricks": [], "partial_at_border": 0, "rejected": {}}
    if close_rgb is not None:
        close = segment_close(close_rgb)
        for rec in close["bricks"]:
            rec.pop("_mask", None)
            verdict, reasons = triage(rec, thr)
            rec.update(verdict=verdict, reasons=reasons, confident=verdict != "UNCERTAIN")
            bricks.append(rec)
    for i, rec in enumerate(bricks):
        rec["id"] = f"b{i:02d}_{rec['view']}"
    return {"bricks": bricks, "hands": top["hands"], "plates_rejected": top["plates_rejected"],
            "partial_at_border": close["partial_at_border"], "rejected": close["rejected"],
            "threshold_loaded": thr is not None}


def scan(top_rgb, close_rgb=None):
    """List of brick dicts (both views). Raises HandsPresent when hands are in the top view."""
    return scan_full(top_rgb, close_rgb)["bricks"]


def find_target(bricks, colour="red"):
    """Brick whose colour matches the team's text (e.g. 'move the red brick'); pink counts as red only if asked."""
    cands = [b for b in bricks if b["colour"] == colour]
    cands.sort(key=lambda b: -b["area"])
    return cands[0] if cands else None


class Inspector:
    """LOOP-CONTRACT section 3 interface for one brick or one frame (CPU; device argument kept for the contract)."""

    def __init__(self, model_dir=None, device="CPU"):
        self.model_dir = model_dir
        self.device = device

    def inspect_rgb(self, rgb, camera_key=None):
        view = "close" if camera_key and "243722064316" in camera_key else "top"
        seg = segment_top(rgb) if view == "top" else segment_close(rgb)
        thr = load_thresholds()
        if view == "top" and seg["hands"]:
            return {"label": "UNCERTAIN", "score": None, "threshold": None, "reasons": ["hands_present"],
                    "geometry": {}, "latency_ms": None, "device": self.device, "crop": None}
        if not seg["bricks"]:
            return {"label": "UNCERTAIN", "score": None, "threshold": None, "reasons": ["no_brick"],
                    "geometry": {}, "latency_ms": None, "device": self.device, "crop": None}
        b = max(seg["bricks"], key=lambda r: r["area"])
        verdict, reasons = triage(b, thr)
        x, y, w, hh = b["bbox"]
        crop = rgb[y:y + hh, x:x + w]
        label = {"CONFIDENT_DEFECT": "DEFECTIVE", "CONFIDENT_GOOD": "GOOD"}.get(verdict, "UNCERTAIN")
        if label == "UNCERTAIN" and b["shape"] == "not_2x2":
            reasons = reasons + ["not_2x2"]
        return {"label": label, "score": b.get("tape_score"), "threshold": (thr or {}).get("defect"),
                "reasons": reasons, "geometry": {"shape": b["shape"], "shape_conf": b["shape_conf"],
                                                 "aspect": b["aspect"], "centre_px": b["centre_px"]},
                "latency_ms": None, "device": self.device, "crop": crop}

    def locate_brick(self, rgb):
        seg = segment_top(rgb)
        if not seg["bricks"]:
            return None
        b = max(seg["bricks"], key=lambda r: r["area"])
        x, y, w, hh = b["bbox"]
        return [x, y, x + w, y + hh]
