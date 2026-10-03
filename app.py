import streamlit as st
import google.generativeai as genai
import tempfile
import os
import time
import mimetypes

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

# --- 4. Core Logic Function ---
def analyze_video(video_path, prompt, key, mime_type):
    genai.configure(api_key=key)
    status_text = st.empty()
    status_text.info("🚀 Uploading footage to AI engine...")

    video_file = genai.upload_file(path=video_path, mime_type=mime_type)
    try:
        deadline = time.monotonic() + PROCESSING_TIMEOUT_S
        while video_file.state.name == "PROCESSING":
            if time.monotonic() > deadline:
                status_text.empty()
                st.error("❌ Video processing timed out. Try a shorter clip.")
                return None
            status_text.info("⏳ Processing video, AI is analyzing court movement...")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name != "ACTIVE":
            status_text.empty()
            st.error(f"❌ Video processing failed (state: {video_file.state.name}).")
            return None

        status_text.info("🧠 Generating tactical and technical feedback...")
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content([video_file, prompt])
        try:
            text = response.text
        except ValueError:
            # Blocked or empty response
            status_text.empty()
            st.error("❌ The AI returned no content (possibly blocked by safety filters).")
            return None
        status_text.success("✅ Analysis Complete!")
        return text
    finally:
        # Don't leave user footage on Google's servers
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass


def show_local_video(path):
    if os.path.exists(path):
        st.video(path)
    else:
        st.warning(f"Video file '{path}' not found in repository.")

# --- 5. Main UI with Tabs ---
st.title("🎾 Next-Gen Squash AI Coach")
st.markdown("Upload your practice footage, or explore AI tactical breakdowns of PSA professionals.")

# 创建两个极其现代的选项卡
tab_solo, tab_pro = st.tabs(["📹 Solo Training (Analyze My Video)", "🏆 Pro Case Studies (PSA)"])

# ==== 选项卡 1：原本的上传分析功能 ====
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
            # 限制个人视频的显示宽度
            col_video, _ = st.columns([1, 1])
            with col_video:
                st.video(uploaded_file)
            
            default_prompt = """
            Act as an elite squash biomechanics analyst. You MUST base your analysis STRICTLY on the visual evidence in this specific video. Do not use generic squash cliches.

            Step 1: Visual Confirmation (Prove you watched the video)
            Briefly describe the player's clothing colors and the specific type of shot/drill they are performing in this exact footage.

            Step 2: Biomechanical Critique
            Based ONLY on the movement shown, provide 2-3 specific observations regarding:
            - The exact timing of their racket preparation relative to the ball's bounce.
            - Their specific footwork pattern (e.g., crossover step, lunge stability, or shuffle).
            - Their balance and posture during the follow-through.

            Highlight what is uniquely good or what specifically needs correction in THIS video clip.
            """
            
            prompt = st.text_area("Coach's Instruction (Prompt)", value=default_prompt, height=180)

            if st.button("Start AI Analysis", type="primary"):
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
                        st.markdown("### 📋 AI Scouting Report")
                        st.markdown(result)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    try:
                        os.remove(temp_filename)
                    except OSError:
                        pass

# ==== 选项卡 2：职业球员战术解析展厅 ====
with tab_pro:
    st.markdown("### 🧠 AI Tactical Breakdown: Paul Coll (Former World #1)")
    st.info("How does AI decode the movement and technique of 'Superman' on the PSA tour?")
    
    # --- 案例一：比赛飞扑回中 ---
    st.markdown("#### Case 1: The 'Superman' Recovery (British Open)")
    col1, col2 = st.columns([1, 1.2]) # 左边视频，右边文字
    
    with col1:
        show_local_video("coll_match.mp4")
            
    with col2:
        st.markdown("**🎯 Prompt to Gemini Vision:**")
        st.code("Analyze the player in black (Paul Coll). Focus on his recovery path to the T-zone after the extreme lunge/dive in the front court.", language="text")
        st.markdown("**💡 AI Output & Tactical Takeaway:**")
        st.success("""
        * **Incredible Resilience:** After the desperate retrieve, Coll instantly pushes off the floor using his core and front lunging leg.
        * **Visual Discipline:** His eyes remain fixed on the front wall and his opponent, never dropping his head.
        * **Efficiency:** Notice the explosive crossover step. He is back dominating the T-zone before the opponent can strike.
        * **Takeaway for Tony:** Never admire your own shot. The point continues until the ball bounces twice. Immediate T-recovery is non-negotiable.
        """)
        
    st.divider() # 分割线
    
    # --- 案例二：前场极速截击 ---
    st.markdown("#### Case 2: Front-Court Volley Drill (Extreme Reaction)")
    col3, col4 = st.columns([1, 1.2])
    
    with col3:
        show_local_video("coll_volley.mp4")
            
    with col4:
        st.markdown("**🎯 Prompt to Gemini Vision:**")
        st.code("Analyze the rapid front-wall volley drill. Focus on racket preparation, backswing length, and wrist stability.", language="text")
        st.markdown("**💡 AI Output & Technical Takeaway:**")
        st.success("""
        * **Shortened Backswing:** To cope with the rapid pace, the backswing is virtually eliminated. The racket head stays up and in front of the body at all times.
        * **Locked Wrist:** The wrist remains completely stable. Power is generated purely from rapid forearm rotation and slight body weight transfer.
        * **Target Fixation:** Outstanding hand-eye coordination with zero wasted movement.
        * **Takeaway for Tony:** On aggressive front-court volleys, shorten the swing, lock the wrist, and keep the racket preparation extremely early.

        """)


