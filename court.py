"""Squash court geometry, calibration (pixel -> court metres) and T-recovery analysis.

Court coordinates: x runs across the court (0 = left wall, 6.4 = right wall, seen from the back wall
looking at the front wall); y runs from the front wall (0) to the back wall (9.75). Units: metres.
"""
import math

import cv2
import numpy as np

COURT_W = 6.4
COURT_L = 9.75
SHORT_LINE_Y = 5.49
T_POINT = (COURT_W / 2, SHORT_LINE_Y)
BOX = 1.6  # service box size

# id -> (description, (x, y) in metres)
LANDMARKS = {
    "fl": ("Front-left corner (front wall x left wall)", (0.0, 0.0)),
    "fr": ("Front-right corner (front wall x right wall)", (COURT_W, 0.0)),
    "bl": ("Back-left corner (back wall x left wall)", (0.0, COURT_L)),
    "br": ("Back-right corner (back wall x right wall)", (COURT_W, COURT_L)),
    "sl": ("Short line x left wall", (0.0, SHORT_LINE_Y)),
    "sr": ("Short line x right wall", (COURT_W, SHORT_LINE_Y)),
    "t": ("T (short line x half-court line)", T_POINT),
    "hb": ("Half-court line x back wall", (COURT_W / 2, COURT_L)),
}
DEFAULT_LANDMARKS = ["fl", "fr", "br", "bl"]

T_RADIUS = 1.0        # m: "on the T"
OUT_RADIUS = 2.0      # m: leaving this counts as an excursion that needs a recovery
MAX_RECOVERY_S = 5.0  # slower than this is treated as a pause between points, not a recovery
MAX_GAP_S = 1.0       # tracking gap that invalidates an excursion
MIN_SAMPLES = 15


def landmark_court_points(ids):
    return np.array([LANDMARKS[i][1] for i in ids], dtype=np.float64)


def compute_homography(pixel_pts, court_pts):
    """Pixel -> court homography. Returns (H, rms_error_m) or (None, None) if degenerate."""
    px = np.asarray(pixel_pts, dtype=np.float64)
    ct = np.asarray(court_pts, dtype=np.float64)
    if len(px) < 4 or len(px) != len(ct):
        return None, None
    # reject (near-)collinear point sets: a degenerate configuration cannot define a plane mapping
    c = ct - ct.mean(axis=0)
    if np.linalg.svd(c, compute_uv=False)[-1] < 1e-6:
        return None, None
    p = px - px.mean(axis=0)
    if np.linalg.svd(p, compute_uv=False)[-1] < 1e-6 * max(1.0, np.abs(px).max()):
        return None, None
    H, _ = cv2.findHomography(px, ct, 0)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        return None, None
    proj = to_court(H, px)
    rms = float(np.sqrt(np.mean(np.sum((proj - ct) ** 2, axis=1))))
    return H, rms


def to_court(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def to_pixels(H, court_pts):
    return to_court(np.linalg.inv(H), court_pts)


def draw_court_overlay(img_rgb, H, color=(255, 75, 75)):
    """Draw the court lines on an image using the calibration, so the user can check the fit."""
    img = img_rgb.copy()

    def seg(a, b):
        p = to_pixels(H, [a, b])
        cv2.line(img, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), color, 2)

    W, L, S = COURT_W, COURT_L, SHORT_LINE_Y
    for a, b in [((0, 0), (W, 0)), ((W, 0), (W, L)), ((W, L), (0, L)), ((0, L), (0, 0)),
                 ((0, S), (W, S)), ((W / 2, S), (W / 2, L)),
                 ((0, S + BOX), (BOX, S + BOX)), ((BOX, S), (BOX, S + BOX)),
                 ((W, S + BOX), (W - BOX, S + BOX)), ((W - BOX, S), (W - BOX, S + BOX))]:
        seg(a, b)
    t = to_pixels(H, [T_POINT])[0]
    cv2.circle(img, (int(t[0]), int(t[1])), 8, (255, 220, 0), -1)
    return img


def draw_topdown(path_xy, events=(), scale=55, margin=24):
    """Top-down court picture with the player's path and the excursion peaks."""
    w, h = int(COURT_W * scale) + 2 * margin, int(COURT_L * scale) + 2 * margin
    img = np.full((h, w, 3), 24, dtype=np.uint8)

    def px(x, y):
        return int(margin + x * scale), int(margin + y * scale)

    line = (200, 200, 200)
    cv2.rectangle(img, px(0, 0), px(COURT_W, COURT_L), line, 2)
    cv2.line(img, px(0, SHORT_LINE_Y), px(COURT_W, SHORT_LINE_Y), line, 1)
    cv2.line(img, px(COURT_W / 2, SHORT_LINE_Y), px(COURT_W / 2, COURT_L), line, 1)
    cv2.rectangle(img, px(0, SHORT_LINE_Y), px(BOX, SHORT_LINE_Y + BOX), line, 1)
    cv2.rectangle(img, px(COURT_W - BOX, SHORT_LINE_Y), px(COURT_W, SHORT_LINE_Y + BOX), line, 1)
    cv2.circle(img, px(*T_POINT), int(T_RADIUS * scale), (255, 220, 0), 1)
    path = np.asarray(path_xy, dtype=np.float64)
    pts = [px(*p) for p in path if np.isfinite(p).all()]
    for a, b in zip(pts[:-1], pts[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 6 * scale:  # don't draw tracker jumps
            cv2.line(img, a, b, (255, 75, 75), 1)
    for i, e in enumerate(events, start=1):
        c = px(*e["peak_xy"])
        cv2.circle(img, c, 6, (255, 220, 0), -1)
        cv2.putText(img, str(i), (c[0] + 8, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _smooth(p, k=3):
    if len(p) < k:
        return p
    pad = k // 2
    padded = np.pad(p, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, i], np.ones(k) / k, mode="valid") for i in range(2)], axis=1)


def analyze_positions(times, xy):
    """Compute T-zone and recovery metrics from court positions. Returns (metrics, events) or (None, [])."""
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(xy, dtype=np.float64)
    if len(t) == 0:
        return None, []
    margin = 0.8
    ok = (np.isfinite(p).all(axis=1) & (p[:, 0] > -margin) & (p[:, 0] < COURT_W + margin)
          & (p[:, 1] > -margin) & (p[:, 1] < COURT_L + margin))
    inside_ratio = float(ok.mean())
    t, p = t[ok], p[ok]
    if len(t) < MIN_SAMPLES:
        return None, []
    p = _smooth(p)
    d = np.hypot(p[:, 0] - T_POINT[0], p[:, 1] - T_POINT[1])

    events, long_pauses, incomplete = [], 0, 0
    n, i = len(d), 0
    while i < n:
        if d[i] <= OUT_RADIUS:
            i += 1
            continue
        j = i
        while j < n and d[j] > T_RADIUS:
            j += 1
        if i == 0:                       # clip starts outside the T: leaving moment unknown
            incomplete += 1
        elif j >= n:                     # never got back before the clip ended
            incomplete += 1
        elif np.any(np.diff(t[i - 1:j + 1]) > MAX_GAP_S):
            incomplete += 1              # tracking lost during the excursion
        else:
            peak = i + int(np.argmax(d[i:j]))
            rec = float(t[j] - t[peak])
            if rec > MAX_RECOVERY_S:
                long_pauses += 1
            else:
                events.append({
                    "peak_time_s": round(float(t[peak]), 1),
                    "peak_distance_m": round(float(d[peak]), 1),
                    "peak_xy": (float(p[peak, 0]), float(p[peak, 1])),
                    "recovery_s": round(rec, 2),
                    "recovery_speed_mps": round((float(d[peak]) - T_RADIUS) / rec, 1) if rec > 0 else None,
                })
        i = j + 1

    recs = [e["recovery_s"] for e in events]
    speeds = [e["recovery_speed_mps"] for e in events if e["recovery_speed_mps"] is not None]
    metrics = {
        "court_positions_used": int(len(t)),
        "court_inside_ratio": round(inside_ratio, 2),
        "time_within_T_zone_pct": round(float(np.mean(d <= T_RADIUS)) * 100, 0),
        "avg_distance_from_T_m": round(float(np.mean(d)), 1),
        "p95_distance_from_T_m": round(float(np.percentile(d, 95)), 1),
        "excursions_from_T": len(events),
        "excursions_skipped_incomplete": incomplete,
        "excursions_skipped_long_pause": long_pauses,
        "recovery_time_avg_s": round(float(np.mean(recs)), 2) if recs else None,
        "recovery_time_median_s": round(float(np.median(recs)), 2) if recs else None,
        "recovery_time_best_s": round(min(recs), 2) if recs else None,
        "recovery_time_worst_s": round(max(recs), 2) if recs else None,
        "recovery_speed_avg_mps": round(float(np.mean(speeds)), 1) if speeds else None,
        "recovery_definition": (
            f"time from the farthest point of a trip beyond {OUT_RADIUS:g} m from the T "
            f"until back within {T_RADIUS:g} m of the T"
        ),
        "events": [{k: v for k, v in e.items() if k != "peak_xy"} for e in events[:12]],
    }
    metrics["reliability"] = "low" if inside_ratio < 0.7 else "ok"
    return metrics, events
