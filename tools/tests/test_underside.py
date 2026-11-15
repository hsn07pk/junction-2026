"""Close camera and underside inspection offline: the close-camera model is fitted from jaw pixels paired with FK
positions (a virtual camera), a pose that shows the held brick's bottom to it is planned with the general IK, and
rendered bottoms (an open DUPLO bottom: lit rim, dark cavity, a tube; or tape over the opening) are judged with limits
fitted on clean bottoms only. The camera's place and the colours are test values of this virtual cell."""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import closecam as cc  # noqa: E402
import kin  # noqa: E402
import sim_scene as sim  # noqa: E402
import survey as sv  # noqa: E402
import underside as us  # noqa: E402


def _camera():
    """A close camera at the front of the table looking back across the work area, slightly down."""
    C = np.array([560.0, 40.0, kin.MAT_Z + 110.0])
    z = np.array([-1.0, -0.08, -0.22])
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, -1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # world -> camera rows
    return {"f": 560.0, "cx": 322.0, "cy": 238.0, "k1": -0.05, "rvec": cv2.Rodrigues(R)[0].ravel().tolist(),
            "centre_mm": C.tolist(), "image_size": [640, 480]}


def _render(M, q, colour, tape=None, seed=0):
    """Close frame of the held brick's bottom: lit rim, dark cavity with the tube, or tape over the opening."""
    sp = sv.spec()
    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), 0.02, np.float32)
    img += np.linspace(0, 0.02, 640)[None, :, None]
    c, n, corners = us.bottom(q, sp, kin.cell.get("tip_final_mm"))
    light = 0.35 + 0.6 * max(0.0, float(n @ np.array([0.3, 0.1, 0.95])))
    col = sim.to_linear(colour) * light

    def quad(scale):
        return np.round(cc.project(M, c + scale * (corners - c))).astype(np.int32)

    cv2.fillConvexPoly(img, quad(1.0), col.tolist())
    cv2.fillConvexPoly(img, quad(0.80), (col * 0.25).tolist())  # the cavity (walls 3 mm)
    tube = [c + 0.33 * (corners[0] - c) * math.cos(t) + 0.33 * (corners[1] - c) * math.sin(t)
            for t in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
    cv2.fillPoly(img, [np.round(cc.project(M, np.array(tube))).astype(np.int32)], (col * 0.5).tolist())
    if tape is not None:
        cv2.fillConvexPoly(img, quad(tape[1]), (sim.to_linear(tape[0]) * light).tolist())
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 0.004, img.shape).astype(np.float32)
    out = np.clip(np.round(sim.to_display(img)), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def test_closecam_fit_recovers_the_camera_from_jaw_pairs():
    true = _camera()
    q_ref = list(kin.cell.get("spot_hover_q"))
    P = np.array([kin.fk(q)[0] for q in cc.grid_poses(q_ref)])
    P = P[cc.in_front(true, P)]
    uv = cc.project(true, P) + np.random.default_rng(1).normal(0, 1.5, (len(P), 2))
    keep = (uv[:, 0] > 0) & (uv[:, 0] < 640) & (uv[:, 1] > 0) & (uv[:, 1] < 480)
    P, uv = P[keep], uv[keep]
    assert len(P) >= 12, len(P)
    test = np.arange(len(P)) % 4 == 0
    M = cc.fit(P[~test], uv[~test], (640, 480))
    err = np.linalg.norm(cc.project(M, P[test]) - cc.project(true, P[test]), axis=1)
    assert np.median(err) < 2.0 and err.max() < 6.0, err
    assert np.linalg.norm(np.array(M["centre_mm"]) - true["centre_mm"]) < 25.0, M["centre_mm"]


def test_underside_plan_and_judge_bottoms():
    """A pose that faces the bottom to the camera exists; clean bottoms of colours the fit never saw pass, tape over
    the opening (white, black, same colour as the brick) fails."""
    M, sp, cfg = _camera(), sv.spec(), us.config()
    q_ref = list(kin.cell.get("spot_hover_q"))
    pl = us.plan(M, q_ref, sp, cfg, kin.cell.get("tip_final_mm"))
    assert pl is not None, "no presentation pose"
    q, info = pl
    assert info["cos"] >= cfg["min_cos"] and info["size_px"] > 40, info

    def record(colour, tape=None, seed=0):
        img = _render(M, q, colour, tape, seed)
        _, _, corners = us.bottom(q, sp, kin.cell.get("tip_final_mm"))
        found = us.find_bottom(img, cc.project(M, corners))
        assert found["ok"] and found["iou"] >= cfg["min_iou"], found
        return us.measure(img, found["quad"])

    fit_set = [{"truth": "GOOD", "m": record(c, seed=i)} for i, c in enumerate(((40, 120, 230), (215, 215, 210),
                                                                                  (40, 70, 120)))]
    um = us.fit(fit_set, cfg)
    cases = [((60, 160, 40), None, "CLEAN"), ((40, 40, 200), None, "CLEAN"), ((150, 150, 150), None, "CLEAN"),
             ((40, 40, 200), ((235, 235, 235), 0.85), "DEFECT"), ((215, 215, 210), ((30, 30, 30), 0.85), "DEFECT"),
             ((200, 110, 30), ((200, 110, 30), 0.85), "DEFECT"), ((60, 160, 40), ((235, 235, 235), 1.0), "DEFECT")]
    wrong = []
    for i, (colour, tape, want) in enumerate(cases):
        m = record(colour, tape, seed=10 + i)
        if us.verdict(m, um) != want:
            wrong.append((colour, tape, want, m))
    assert not wrong, (um, wrong)
