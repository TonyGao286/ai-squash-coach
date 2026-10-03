"""Skeleton-based motion metrics for a squash clip (MediaPipe Pose, 2D).

All distances are normalised by the player's torso length so they are comparable
across camera distances. Angles/ratios are 2D projections: they depend on camera
angle and are indicative, not laboratory measurements.
"""
import math
import os

import numpy as np

import court

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker_full.task")

# MediaPipe pose landmark indices
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28
MIN_VIS = 0.5
MIN_FRAMES = 10
MAX_JUMP_TORSO = 2.5  # per sampled frame; larger = tracker jumped to another person


def _pt(frame, i):
    x, y, v = frame[i]
    return np.array([x, y]) if v >= MIN_VIS else None


def _angle(a, b, c):
    """Angle at b (degrees) formed by a-b-c."""
    if a is None or b is None or c is None:
        return None
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1, 1))))


def _mid(a, b):
    return None if a is None or b is None else (a + b) / 2


def _stats(values):
    return [v for v in values if v is not None]


def _pct(values, q):
    vals = _stats(values)
    return float(np.percentile(vals, q)) if vals else None


def compute_metrics(frames, times):
    """frames: list of 33x(x_px, y_px, visibility) arrays for the tracked player.
    times: matching timestamps in seconds. Returns (metrics dict, deepest_frame_index)."""
    n = len(frames)
    if n < MIN_FRAMES:
        return None, None

    torsos, shoulder_w = [], []
    knee_l, knee_r, lean, stance, centers, wr_l, wr_r, el_l, el_r = ([] for _ in range(9))
    sh_mids = []
    for f in frames:
        ls, rs, lh, rh = (_pt(f, i) for i in (L_SH, R_SH, L_HIP, R_HIP))
        sm, hm = _mid(ls, rs), _mid(lh, rh)
        sh_mids.append(sm)
        if sm is not None and hm is not None:
            torsos.append(float(np.linalg.norm(sm - hm)))
            d = sm - hm  # pointing up in the image => negative y
            lean.append(math.degrees(math.atan2(abs(d[0]), max(-d[1], 1e-6))))
            centers.append(hm)
        else:
            lean.append(None)
            centers.append(None)
        if ls is not None and rs is not None:
            shoulder_w.append(float(np.linalg.norm(ls - rs)))
        la, ra = _pt(f, L_ANK), _pt(f, R_ANK)
        stance.append(None if la is None or ra is None else float(np.linalg.norm(la - ra)))
        knee_l.append(_angle(_pt(f, L_HIP), _pt(f, L_KNEE), la))
        knee_r.append(_angle(_pt(f, R_HIP), _pt(f, R_KNEE), ra))
        wr_l.append(_pt(f, L_WR))
        wr_r.append(_pt(f, R_WR))
        el_l.append(_pt(f, L_EL))
        el_r.append(_pt(f, R_EL))

    if len(torsos) < MIN_FRAMES:
        return None, None
    torso = float(np.median(torsos))
    sw = float(np.median(shoulder_w)) if shoulder_w else None
    t = np.asarray(times, dtype=float)

    def speeds(points):
        out = []
        for i in range(1, n):
            a, b = points[i - 1], points[i]
            dt = t[i] - t[i - 1]
            d = None if a is None or b is None or dt <= 0 else float(np.linalg.norm(b - a)) / torso
            out.append(None if d is None or d > MAX_JUMP_TORSO else d / dt)
        return out

    # dominant (racket) arm = the wrist that travels more
    def path(points):
        return sum(s for s in speeds(points) if s is not None)

    # tracking switches show up as huge frame-to-frame jumps; ignore those samples
    right_dom = path(wr_r) >= path(wr_l)
    wr, sp_pts = (wr_r, wr_r) if right_dom else (wr_l, wr_l)
    wrist_speed = _stats(speeds(sp_pts))
    elev = []
    for w, sm in zip(wr, sh_mids):
        elev.append(None if w is None or sm is None else float((sm[1] - w[1]) / torso))

    # lead knee at each frame = the more bent one
    knee_min_per_frame = [min(v for v in (a, b) if v is not None) if (a is not None or b is not None) else None
                          for a, b in zip(knee_l, knee_r)]
    knee_vals = _stats(knee_min_per_frame)
    deepest = None
    if knee_vals:
        deepest = int(np.nanargmin([v if v is not None else np.nan for v in knee_min_per_frame]))

    center_speed = speeds(centers)
    center_path = sum(s * (t[i + 1] - t[i]) for i, s in enumerate(center_speed) if s is not None)
    stance_vals = _stats(stance)
    lean_vals = _stats(lean)

    def r(x, d=1):
        return None if x is None else round(float(x), d)

    metrics = {
        "clip_seconds_analysed": r(t[-1] - t[0]),
        "frames_with_player": n,
        "racket_arm": "right" if right_dom else "left",
        "lead_knee_angle_deepest_deg": r(_pct(knee_vals, 5)),
        "lead_knee_angle_avg_deg": r(np.mean(knee_vals)) if knee_vals else None,
        "torso_lean_avg_deg": r(np.mean(lean_vals)) if lean_vals else None,
        "torso_lean_peak_deg": r(_pct(lean_vals, 95)),
        "stance_width_avg_x_shoulder_width": r(np.mean(stance_vals) / sw, 2) if stance_vals and sw else None,
        "stance_width_peak_x_shoulder_width": r(_pct(stance_vals, 95) / sw, 2) if stance_vals and sw else None,
        "racket_wrist_peak_speed_torso_per_s": r(_pct(wrist_speed, 95), 2),
        "racket_wrist_avg_speed_torso_per_s": r(np.mean(wrist_speed), 2) if wrist_speed else None,
        "racket_wrist_peak_height_above_shoulders_torso": r(_pct(elev, 95), 2),
        "distance_covered_torso_lengths": r(center_path, 1),
        "avg_movement_speed_torso_per_s": r(np.mean(_stats(center_speed)), 2) if _stats(center_speed) else None,
    }
    return metrics, deepest


def _select_player(poses, prev_center, width, height, start_side="largest"):
    """Pick one pose from several detections: continue the previous player; on the first frame use
    start_side (largest / left / right)."""
    best, best_score = None, None
    for lm in poses:
        pts = np.array([[p.x * width, p.y * height, p.visibility] for p in lm])
        vis = pts[pts[:, 2] >= MIN_VIS]
        if len(vis) < 8:
            continue
        center = vis[:, :2].mean(axis=0)
        size = float(np.ptp(vis[:, 1])) + float(np.ptp(vis[:, 0]))
        if prev_center is not None:
            score = -np.linalg.norm(center - prev_center)
        elif start_side == "left":
            score = -center[0]
        elif start_side == "right":
            score = center[0]
        else:
            score = size
        if best_score is None or score > best_score:
            best, best_score = (pts, center), score
    return best


def metrics_torso_px(frames):
    vals = []
    for f in frames:
        sm, hm = _mid(_pt(f, L_SH), _pt(f, R_SH)), _mid(_pt(f, L_HIP), _pt(f, R_HIP))
        if sm is not None and hm is not None:
            vals.append(float(np.linalg.norm(sm - hm)))
    return float(np.median(vals)) if vals else 1.0


def analyze_clip(video_path, start_side="largest", calibration=None, target_fps=15, max_seconds=60, max_width=640, model_path=MODEL_PATH):
    """Run pose estimation on a clip. calibration = {"landmarks": [ids], "points_norm": [[x, y], ...]} with
    pixel positions normalised to 0-1 of the frame; when given, court-position metrics are added. Returns dict with metrics, quality info and a keyframe image
    (RGB ndarray with skeleton drawn at the deepest lunge), or None if the player was not found."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / target_fps)))
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frames, times, images, sampled = [], [], [], 0
    frame_size = None
    prev_center = None
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            idx = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                ts = idx / fps
                if ts > max_seconds:
                    break
                if idx % step == 0:
                    sampled += 1
                    h, w = bgr.shape[:2]
                    if w > max_width:
                        bgr = cv2.resize(bgr, (max_width, int(h * max_width / w)))
                        h, w = bgr.shape[:2]
                    frame_size = (w, h)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(ts * 1000))
                    chosen = _select_player(result.pose_landmarks, prev_center, w, h, start_side) if result.pose_landmarks else None
                    if chosen:
                        pts, prev_center = chosen
                        frames.append(pts)
                        times.append(ts)
                        images.append(rgb)
                idx += 1
    finally:
        cap.release()

    if sampled == 0:
        return None
    metrics, deepest = compute_metrics(frames, times)
    if metrics is None:
        return None
    metrics["player_detected_ratio"] = round(len(frames) / sampled, 2)
    metrics["tracking_jumps"] = int(sum(
        1 for a, b in zip(frames[:-1], frames[1:])
        if np.linalg.norm(a[:, :2].mean(axis=0) - b[:, :2].mean(axis=0)) > metrics_torso_px(frames) * MAX_JUMP_TORSO))

    metrics["reliability"] = (
        "low" if metrics["player_detected_ratio"] < 0.8 or metrics["tracking_jumps"] >= 3 else "ok"
    )

    court_map = None
    if calibration and frame_size:
        w, h = frame_size
        H, rms = court.compute_homography(
            np.asarray(calibration["points_norm"], dtype=np.float64) * [w, h],
            court.landmark_court_points(calibration["landmarks"]),
        )
        if H is None:
            metrics["court_recovery"] = {"error": "calibration points are degenerate"}
        else:
            foot = []
            for f in frames:
                feet = [f[i][:2] for i in (L_ANK, R_ANK) if f[i][2] >= MIN_VIS]
                foot.append(np.mean(feet, axis=0) if feet else [np.nan, np.nan])
            xy = court.to_court(H, np.asarray(foot, dtype=np.float64))
            cm, events = court.analyze_positions(times, xy)
            if cm is None:
                metrics["court_recovery"] = {"error": "not enough on-court positions to measure recovery"}
            else:
                cm["calibration_error_m"] = round(rms, 2)
                if rms > 0.5:
                    cm["reliability"] = "low"
                metrics["court_recovery"] = cm
                court_map = court.draw_topdown(xy[np.isfinite(xy).all(axis=1)], events)

    keyframe = None
    if deepest is not None:
        img = images[deepest].copy()
        pts = frames[deepest]
        for a, b in [(L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR), (L_SH, L_HIP),
                     (R_SH, R_HIP), (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANK), (R_HIP, R_KNEE),
                     (R_KNEE, R_ANK)]:
            if pts[a][2] >= MIN_VIS and pts[b][2] >= MIN_VIS:
                cv2.line(img, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), (255, 75, 75), 3)
        for i in (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK):
            if pts[i][2] >= MIN_VIS:
                cv2.circle(img, (int(pts[i][0]), int(pts[i][1])), 5, (255, 255, 255), -1)
        keyframe = img
    return {"metrics": metrics, "keyframe": keyframe, "keyframe_time": round(times[deepest], 1) if deepest is not None else None,
            "court_map": court_map}
