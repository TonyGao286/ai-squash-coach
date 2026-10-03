import streamlit as st
from google import genai
from google.genai import types
import tempfile
import os
import time
import mimetypes
import json
import re

import history

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


def build_prompt(coach_instruction, benchmarks):
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

            instruction = st.text_area(
                "Coach's Instruction (optional focus)",
                value="Focus on racket preparation timing, footwork pattern and balance during the follow-through.",
                height=100,
            )

            if st.button("Start AI Analysis", type="primary"):
                chosen = [b for b in BENCHMARKS if b["id"] in selected_ids]
                prompt = build_prompt(instruction, chosen)
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
