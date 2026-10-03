import streamlit as st
from google import genai
from google.genai import types
import tempfile
import os
import time
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
import mimetypes
import json
import re

import history
import pose_metrics
import court

MODEL_NAME = "gemini-2.5-flash"
MAX_UPLOAD_MB = 100
PROCESSING_TIMEOUT_S = 300

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Squash Coach", page_icon="🎾", layout="wide") # 改为 wide 布局，让左右分栏更美观

# --- 2. API Key Setup (Secure Mode) ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("GOOGLE_API_KEY")

# --- 3. Sidebar: Clean & Professional ---
with st.sidebar:
    st.header("⚙️ System Status")
    if not api_key:
        api_key = st.text_input("Enter Google API Key (Developer Mode)", type="password")
    
    if api_key:
        st.success("🟢 API key configured")
    else:
        st.warning("🟠 API key required")
    st.info("💡 Powered by Gemini Vision AI. \n\nDeveloper: Tony Gao")

# --- 4. Benchmarks & prompt building ---
BENCHMARK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks.json")


@st.cache_data
def load_benchmarks():
    try:
        with open(BENCHMARK_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


BENCHMARKS = load_benchmarks()

OUTPUT_SCHEMA = """{
  "visual_confirmation": "player clothing colors and the exact shot/drill seen in THIS video",
  "overall_score": <integer 1-10>,
  "summary": "2-3 sentence overall assessment",
  "strengths": ["specific strength observed in this clip", ...],
  "issues": [
    {"area": "Racket preparation | Footwork | Balance & posture | Recovery | Other",
     "observation": "what happens in this clip, with approximate timestamp",
     "benchmark_gap": "how it differs from the benchmark (or 'n/a' if no benchmark)",
     "fix": "one concrete correction"}
  ],
  "drills": [{"name": "drill name", "how": "how to do it", "goal": "what it fixes"}]
}"""


def build_prompt(coach_instruction, benchmarks, metrics=None):
    parts = [
        "Act as an elite squash biomechanics analyst. Base your analysis STRICTLY on the "
        "visual evidence in this specific video. Do not use generic squash cliches. "
        "If something cannot be judged from the footage, say so instead of guessing.",
        coach_instruction.strip(),
    ]
    if benchmarks:
        parts.append("Compare the player against these professional benchmarks:")
        for b in benchmarks:
            points = "\n".join(f"- {p}" for p in b["key_points"])
            parts.append(f"Benchmark - {b['player']}: {b['title']}\n{b['focus']}\n{points}")
    if metrics:
        shown = {k: v for k, v in metrics.items() if v is not None}
        parts.append(
            "Skeleton-tracking measurements for the player (MediaPipe pose, 2D, camera-angle dependent; "
            "distances in torso lengths; angles in degrees; 'deepest' and 'peak' values are 5th/95th "
            "percentiles). Use them as supporting evidence and quote the numbers in your observations, "
            "but trust the video when they disagree"
            + (" - tracking reliability is LOW (the tracker may have mixed up players), so treat them with caution"
               if metrics.get("reliability") == "low" else "")
            + ":\n" + json.dumps(shown, indent=1)
        )
        if "court_recovery" in shown and "error" not in shown["court_recovery"]:
            parts.append(
                "'court_recovery' comes from a calibrated court map: it measures how fast the player gets back "
                "to the T after being pulled away. Comment on T discipline and recovery speed using those "
                "numbers (professionals typically recover in roughly 1-2 s), and cite specific events."
            )
    parts.append(
        "Respond with ONLY valid JSON (no markdown fences) matching this shape:\n" + OUTPUT_SCHEMA
    )
    return "\n\n".join(parts)


def parse_report(text):
    """Parse model output as JSON; return None if it isn't a valid report."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(cleaned)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


METRIC_LABELS = [
    ("lead_knee_angle_deepest_deg", "Deepest knee bend", "°"),
    ("torso_lean_peak_deg", "Peak torso lean", "°"),
    ("stance_width_peak_x_shoulder_width", "Widest stance", "× shoulders"),
    ("racket_wrist_peak_speed_torso_per_s", "Racket-hand peak speed", "torso/s"),
    ("distance_covered_torso_lengths", "Distance covered", "torso lengths"),
    ("avg_movement_speed_torso_per_s", "Avg movement speed", "torso/s"),
]


def render_metrics(pose):
    m = pose["metrics"]
    st.markdown("### 🦴 Measured Movement")
    if m.get("reliability") == "low":
        st.warning(
            "Tracking confidence is low (player lost or swapped with the opponent). "
            "Treat these numbers as rough; a clip with one player clearly visible works best."
        )
    cols = st.columns(3)
    for i, (key, label, unit) in enumerate(METRIC_LABELS):
        if m.get(key) is not None:
            cols[i % 3].metric(label, f"{m[key]:g} {unit}")
    cr = m.get("court_recovery")
    if isinstance(cr, dict):
        st.markdown("#### ⏱️ Recovery to the T")
        if "error" in cr:
            st.warning(f"Court timing unavailable: {cr['error']}.")
        else:
            if cr.get("reliability") == "low":
                st.warning("Court mapping looks unreliable (calibration error or player often off-court).")
            c = st.columns(4)
            c[0].metric("Avg recovery", f"{cr['recovery_time_avg_s']:g} s" if cr["recovery_time_avg_s"] is not None else "n/a")
            c[1].metric("Best / worst",
                        f"{cr['recovery_time_best_s']:g} / {cr['recovery_time_worst_s']:g} s" if cr["recovery_time_best_s"] is not None else "n/a")
            c[2].metric("Trips from T", cr["excursions_from_T"])
            c[3].metric("Time on the T", f"{cr['time_within_T_zone_pct']:g}%")
            st.caption(cr["recovery_definition"].capitalize() + ". Long pauses (>5 s) are ignored.")
            if cr["events"]:
                st.dataframe(
                    [{"Time (s)": e["peak_time_s"], "Farthest from T (m)": e["peak_distance_m"],
                      "Recovery (s)": e["recovery_s"]} for e in cr["events"]],
                    hide_index=True,
                )
            if pose.get("court_map") is not None:
                st.image(pose["court_map"], caption="Player path (red), farthest points (numbered), T zone (yellow circle)", width=260)
    if pose.get("keyframe") is not None:
        st.image(
            pose["keyframe"],
            caption=f"Deepest-knee-bend frame at {pose['keyframe_time']}s - check the skeleton is on the right player",
            width=320,
        )


def render_report(report):
    st.markdown("### 📋 AI Scouting Report")
    score = report.get("overall_score")
    if isinstance(score, (int, float)):
        st.metric("Overall score", f"{score}/10")
    if report.get("summary"):
        st.markdown(report["summary"])
    if report.get("visual_confirmation"):
        st.caption(f"👁️ Seen in video: {report['visual_confirmation']}")

    if report.get("strengths"):
        st.markdown("#### ✅ Strengths")
        for item in report["strengths"]:
            st.markdown(f"- {item}")

    if report.get("issues"):
        st.markdown("#### 🔧 Things to fix")
        for issue in report["issues"]:
            if not isinstance(issue, dict):
                continue
            with st.expander(issue.get("area", "Issue"), expanded=True):
                st.markdown(f"**Observation:** {issue.get('observation', '')}")
                gap = issue.get("benchmark_gap")
                if gap and gap.lower() != "n/a":
                    st.markdown(f"**vs. benchmark:** {gap}")
                st.markdown(f"**Fix:** {issue.get('fix', '')}")

    if report.get("drills"):
        st.markdown("#### 🏋️ Recommended drills")
        for d in report["drills"]:
            if isinstance(d, dict):
                st.markdown(f"- **{d.get('name', '')}** - {d.get('how', '')} _(goal: {d.get('goal', '')})_")


# --- 5. Core logic ---
def _state_name(f):
    state = getattr(f, "state", None)
    return getattr(state, "name", None) or str(state or "ACTIVE")


def analyze_video(video_path, prompt, key, mime_type):
    client = genai.Client(api_key=key)
    status_text = st.empty()
    status_text.info("🚀 Uploading footage to AI engine...")

    video_file = client.files.upload(file=video_path, config=types.UploadFileConfig(mime_type=mime_type))
    try:
        deadline = time.monotonic() + PROCESSING_TIMEOUT_S
        while _state_name(video_file) == "PROCESSING":
            if time.monotonic() > deadline:
                status_text.empty()
                st.error("❌ Video processing timed out. Try a shorter clip.")
                return None
            status_text.info("⏳ Processing video, AI is analyzing court movement...")
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        if _state_name(video_file) != "ACTIVE":
            status_text.empty()
            st.error(f"❌ Video processing failed (state: {_state_name(video_file)}).")
            return None

        status_text.info("🧠 Generating tactical and technical feedback...")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[video_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.3
            ),
        )
        text = response.text
        if not text:
            # Blocked or empty response
            status_text.empty()
            st.error("❌ The AI returned no content (possibly blocked by safety filters).")
            return None
        status_text.success("✅ Analysis Complete!")
        return text
    finally:
        # Don't leave user footage on Google's servers
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass


def show_local_video(path):
    if os.path.exists(path):
        st.video(path)
    else:
        st.warning(f"Video file '{path}' not found in repository.")


@st.cache_data(show_spinner=False, max_entries=8)
def grab_frame(file_id, seconds, _data, suffix, max_width=720):
    """Return (RGB frame at `seconds`, clip duration in seconds)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(_data)
        path = f.name
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, bgr = cap.read()
        cap.release()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if not ok:
        return None, duration
    h, w = bgr.shape[:2]
    if w > max_width:
        bgr = cv2.resize(bgr, (max_width, int(h * max_width / w)))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), duration


def court_calibrator(uploaded_file):
    """Click-to-calibrate UI. Returns {"landmarks", "points_norm"} when complete, else None."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    options = list(court.LANDMARKS)
    ids = st.multiselect(
        "Court points visible in the video (pick at least 4, then click them in this order)",
        options=options,
        default=court.DEFAULT_LANDMARKS,
        format_func=lambda k: court.LANDMARKS[k][0],
    )
    if len(ids) < 4:
        st.info("Select at least 4 points.")
        return None

    _, duration = grab_frame(uploaded_file.file_id, 0.0, uploaded_file.getvalue(), suffix)
    seconds = st.slider("Frame to click on (pick one where the floor lines are clear)",
                        0.0, max(float(duration), 0.1), 0.0, 0.1)
    frame, _ = grab_frame(uploaded_file.file_id, seconds, uploaded_file.getvalue(), suffix)
    if frame is None:
        st.error("Could not read that frame.")
        return None

    sig = (uploaded_file.file_id, tuple(ids), seconds)
    state = st.session_state.setdefault("calib", {})
    if state.get("sig") != sig:
        state.clear()
        state.update(sig=sig, pts=[], resets=state.get("resets", 0) + 1)
    pts = state["pts"]
    h, w = frame.shape[:2]

    marked = frame.copy()
    for n, (x, y) in enumerate(pts, start=1):
        c = (int(x * w), int(y * h))
        cv2.circle(marked, c, 6, (255, 75, 75), -1)
        cv2.putText(marked, str(n), (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    done = len(pts) == len(ids)
    H = rms = None
    if done:
        H, rms = court.compute_homography(np.array(pts) * [w, h], court.landmark_court_points(ids))
        if H is None:
            st.error("These points do not define a court (too close to a line). Reset and re-click.")
        else:
            marked = court.draw_court_overlay(marked, H)
            if len(ids) == 4:
                st.info("Check the red court lines sit on the real floor lines. If they drift, reset and click more carefully.")
            else:
                (st.success if rms <= 0.3 else st.warning)(
                    f"Calibration fit error: {rms:.2f} m. Check the red court lines sit on the real floor lines"
                    + ("" if rms <= 0.3 else " - if not, reset and click more carefully.")
                )
    else:
        st.markdown(f"**Click point {len(pts) + 1}/{len(ids)}:** {court.LANDMARKS[ids[len(pts)]][0]}")

    click = streamlit_image_coordinates(
        marked, width=min(w, 480), key=f"calib_img_{state['resets']}_{len(pts)}_{done}",
        cursor="crosshair",
    )
    if click and not done:
        pts.append([click["x"] / click["width"], click["y"] / click["height"]])
        st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("Undo last point", disabled=not pts):
        pts.pop()
        state["resets"] += 1
        st.rerun()
    if c2.button("Reset points", disabled=not pts):
        pts.clear()
        state["resets"] += 1
        st.rerun()
    return {"landmarks": ids, "points_norm": [list(p) for p in pts]} if done and H is not None else None


# --- 6. Main UI with Tabs ---
st.title("🎾 Next-Gen Squash AI Coach")
st.markdown("Upload your practice footage, or explore AI tactical breakdowns of PSA professionals.")

tab_solo, tab_hist, tab_pro = st.tabs(
    ["📹 Solo Training (Analyze My Video)", "📈 History & Progress", "🏆 Pro Case Studies (PSA)"]
)

# ==== Tab 1: upload & analyze ====
with tab_solo:
    st.markdown("### Upload Your Footage")
    uploaded_file = st.file_uploader(
        f"Upload video clip (Recommended length: < 30 seconds, max {MAX_UPLOAD_MB} MB)",
        type=['mp4', 'mov', 'avi'],
    )

    if uploaded_file is not None:
        if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"❌ File is too large (max {MAX_UPLOAD_MB} MB).")
        elif not api_key:
            st.error("❌ API Key not detected. Please check system configurations.")
        else:
            col_video, _ = st.columns([1, 1])
            with col_video:
                st.video(uploaded_file)

            titles = {b["id"]: f"{b['player']} - {b['title']}" for b in BENCHMARKS}
            selected_ids = st.multiselect(
                "Compare against pro benchmarks",
                options=list(titles),
                default=list(titles),
                format_func=titles.get,
            )

            note = st.text_input("Session note (optional, saved to history)", placeholder="e.g. Tuesday drill, backhand boast")

            use_pose = st.checkbox(
                "Add skeleton motion metrics (knee bend, stance, swing speed, movement)", value=True,
                help="Runs pose tracking locally first (~10-20 s) and gives the AI measured numbers.",
            )
            start_side = st.radio(
                "Which player is Tony (if two players are in frame)?",
                ["largest", "left", "right"],
                format_func={"largest": "Closest to camera", "left": "Left side", "right": "Right side"}.get,
                horizontal=True,
                disabled=not use_pose,
            )

            use_court = st.checkbox(
                "Calibrate the court (enables recovery-to-T timing)", value=False, disabled=not use_pose,
                help="Click 4 known floor points once so pixels can be converted to metres on court.",
            )
            calibration = None
            if use_court and use_pose:
                with st.container(border=True):
                    calibration = court_calibrator(uploaded_file)
                if calibration is None:
                    st.caption("Finish the calibration above to include recovery timing (analysis still runs without it).")

            instruction = st.text_area(
                "Coach's Instruction (optional focus)",
                value="Focus on racket preparation timing, footwork pattern and balance during the follow-through.",
                height=100,
            )

            if st.button("Start AI Analysis", type="primary"):
                chosen = [b for b in BENCHMARKS if b["id"] in selected_ids]
                file_extension = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                mime_type = (
                    uploaded_file.type
                    or mimetypes.guess_type(uploaded_file.name)[0]
                    or "video/mp4"
                )

                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tfile:
                    tfile.write(uploaded_file.getvalue())
                    temp_filename = tfile.name

                try:
                    metrics = None
                    if use_pose:
                        with st.spinner("🦴 Tracking skeleton and measuring movement..."):
                            try:
                                pose = pose_metrics.analyze_clip(
                                    temp_filename, start_side=start_side, calibration=calibration
                                )
                            except Exception as e:
                                pose = None
                                st.warning(f"Skeleton tracking unavailable, continuing without it ({e}).")
                        if pose:
                            metrics = pose["metrics"]
                            render_metrics(pose)
                        elif pose is None:
                            st.warning("Could not track a player in this clip; continuing without metrics.")
                    prompt = build_prompt(instruction, chosen, metrics)
                    result = analyze_video(temp_filename, prompt, api_key, mime_type)
                    if result:
                        report = parse_report(result)
                        if report:
                            render_report(report)
                            try:
                                history.add_record(
                                    report,
                                    filename=uploaded_file.name,
                                    note=note.strip(),
                                    benchmarks=[b["id"] for b in chosen],
                                    metrics=metrics,
                                )
                                st.caption("💾 Saved to History tab.")
                            except OSError as e:
                                st.warning(f"Could not save to history: {e}")
                        else:
                            st.markdown("### 📋 AI Scouting Report")
                            st.markdown(result)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    try:
                        os.remove(temp_filename)
                    except OSError:
                        pass

# ==== Tab 2: history & progress ====
def format_ts(ts):
    return ts.replace("T", " ")[:16]


with tab_hist:
    st.markdown("### 📈 Progress Over Time")
    records = history.load_history()

    with st.expander("Backup / restore history"):
        st.caption(
            "History is stored in a local file. On hosted deployments that file may be reset "
            "when the app restarts, so download a backup now and then."
        )
        st.download_button(
            "Download history (JSON)",
            data=json.dumps(records, ensure_ascii=False, indent=2),
            file_name="squash_history.json",
            mime="application/json",
            disabled=not records,
        )
        imported = st.file_uploader("Restore from backup", type=["json"], key="history_import")
        if imported is not None and st.button("Import backup"):
            try:
                added = history.merge_records(json.loads(imported.getvalue().decode("utf-8")))
                st.success(f"Imported {added} new session(s).")
                st.rerun()
            except (ValueError, OSError) as e:
                st.error(f"Could not import: {e}")

    if not records:
        st.info("No sessions yet. Run an analysis in the Solo Training tab and it will appear here.")
    else:
        scored = [
            {"Date": format_ts(r["timestamp"]), "Score": r["report"]["overall_score"]}
            for r in records
            if isinstance(r["report"].get("overall_score"), (int, float))
        ]
        if len(scored) >= 2:
            st.line_chart(scored, x="Date", y="Score")
            delta = scored[-1]["Score"] - scored[0]["Score"]
            st.metric("Change since first session", f"{scored[-1]['Score']}/10", f"{delta:+g}")
        elif scored:
            st.caption("Run at least two sessions to see a progress chart.")

        st.markdown("#### Sessions")
        for r in reversed(records):
            rep = r["report"]
            score = rep.get("overall_score")
            label = f"{format_ts(r['timestamp'])} - {r.get('note') or r.get('filename') or 'Session'}"
            if score is not None:
                label += f"  ({score}/10)"
            with st.expander(label):
                if rep.get("summary"):
                    st.markdown(rep["summary"])
                for issue in rep.get("issues") or []:
                    if isinstance(issue, dict):
                        st.markdown(f"- **{issue.get('area', '')}:** {issue.get('fix', '')}")
                m = r.get("metrics")
                if isinstance(m, dict):
                    shown = [f"{label}: {m[k]:g} {unit}" for k, label, unit in METRIC_LABELS if m.get(k) is not None]
                    cr = m.get("court_recovery")
                    if isinstance(cr, dict) and cr.get("recovery_time_avg_s") is not None:
                        shown.append(f"Avg recovery to T: {cr['recovery_time_avg_s']:g} s")
                    if shown:
                        st.caption("🦴 " + " · ".join(shown))
                if st.button("Delete this session", key=f"del_{r['id']}"):
                    try:
                        history.delete_record(r["id"])
                        st.rerun()
                    except OSError as e:
                        st.error(f"Could not delete: {e}")

# ==== Tab 3: pro benchmark library ====
with tab_pro:
    st.markdown("### 🧠 Tactical Breakdown: Paul Coll (Former World #1)")
    st.info(
        "These curated breakdowns are the reference benchmarks used when analyzing your own video."
    )
    if not BENCHMARKS:
        st.warning("No benchmarks found (benchmarks.json is missing or invalid).")
    for i, b in enumerate(BENCHMARKS, start=1):
        st.markdown(f"#### Case {i}: {b['title']}")
        col_v, col_t = st.columns([1, 1.2])
        with col_v:
            show_local_video(b["video"])
        with col_t:
            st.markdown("**🎯 Focus:**")
            st.code(b["prompt"], language="text")
            st.markdown("**💡 Key points:**")
            st.success("\n".join(f"* {p}" for p in b["key_points"]))
        if i < len(BENCHMARKS):
            st.divider()
