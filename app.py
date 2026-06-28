import streamlit as st
import cv2
# import mediapipe as mp
import numpy as np
import sqlite3
from datetime import date
import matplotlib.pyplot as plt

# ── Setup ──────────────────────────────────────

import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# mp_pose = mp.solutions.pose
# mp_draw = mp.solutions.drawing_utils
IS_CLOUD = os.environ.get("HOME") == "/home/adminuser"

if not IS_CLOUD:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
# ── Database ────────────────────────────────────
def init_db():
    conn = sqlite3.connect("yogamirror.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (date TEXT, pose TEXT, duration INTEGER, accuracy REAL)''')
    conn.commit()
    conn.close()

def save_session(pose, duration, accuracy):
    conn = sqlite3.connect("yogamirror.db")
    c = conn.cursor()
    c.execute("INSERT INTO sessions VALUES (?,?,?,?)",
              (str(date.today()), pose, duration, accuracy))
    conn.commit()
    conn.close()
    print(f"✅ Session saved: {pose}, {duration}s")  # yeh add karo

def get_history():
    conn = sqlite3.connect("yogamirror.db")
    c = conn.cursor()
    c.execute("SELECT * FROM sessions ORDER BY date DESC")
    data = c.fetchall()
    conn.close()
    return data

# ── Angle Calculator ────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return round(angle, 2)

# ── AI Feedback ─────────────────────────────────
# def get_ai_feedback(pose_name, landmarks, concern):
#     lm = landmarks.landmark

#     def angle(a, b, c):
#         return calculate_angle(
#             [lm[a].x, lm[a].y],
#             [lm[b].x, lm[b].y],
#             [lm[c].x, lm[c].y]
#         )

#     angles = {
#         "left_knee":      angle(23, 25, 27),
#         "right_knee":     angle(24, 26, 28),
#         "left_shoulder":  angle(11, 13, 15),
#         "right_shoulder": angle(12, 14, 16),
#         "left_hip":       angle(11, 23, 25),
#         "right_hip":      angle(12, 24, 26),
#     }

#     prompt = f"""
#     User is doing {pose_name}.
#     Health concern: {concern}

#     Current body angles:
#     - Left knee: {angles['left_knee']}°
#     - Right knee: {angles['right_knee']}°
#     - Left shoulder: {angles['left_shoulder']}°
#     - Right shoulder: {angles['right_shoulder']}°
#     - Left hip: {angles['left_hip']}°
#     - Right hip: {angles['right_hip']}°

#     You are a yoga therapy expert.
#     Based on correct alignment for {pose_name},
#     give 1 short correction in simple Hindi.
#     Max 10 words only. No explanation.
#     """

#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content

def get_ai_feedback(pose_name, landmarks, concern):
    lm = landmarks.landmark

    def angle(a, b, c):
        return calculate_angle(
            [lm[a].x, lm[a].y],
            [lm[b].x, lm[b].y],
            [lm[c].x, lm[c].y]
        )

    angles = {
        "left_knee":      angle(23, 25, 27),
        "right_knee":     angle(24, 26, 28),
        "left_shoulder":  angle(11, 13, 15),
        "right_shoulder": angle(12, 14, 16),
        "left_hip":       angle(11, 23, 25),
        "right_hip":      angle(12, 24, 26),
    }

    # ── Validation ──────────────────────────────
    # Agar saare angles 0 ya 180 hain matlab body
    # properly detect nahi hui — AI call mat karo
    valid_angles = [v for v in angles.values() 
                   if 10 < v < 170]
    
    if len(valid_angles) < 3:
        return "Thoda door ho jao — poora body dikhao 🙏"

    prompt = f"""
You are a clinical Yoga Therapist with 20+ years experience.

User is performing: {pose_name}
User health concern: {concern}

Current measured body angles:
- Left knee:      {angles['left_knee']}°
- Right knee:     {angles['right_knee']}°
- Left shoulder:  {angles['left_shoulder']}°
- Right shoulder: {angles['right_shoulder']}°
- Left hip:       {angles['left_hip']}°
- Right hip:      {angles['right_hip']}°

Your task:
1. Analyze these angles for {pose_name}
2. Find the MOST IMPORTANT correction needed
3. Give exactly 1 correction in simple Hindi
4. Max 8 words
5. Be specific — mention which body part

Example good corrections:
- "Dono haath upar seedhe rakho"
- "Ghutna 90 degree pe laao"
- "Kamar seedhi karo aage mat jhuko"

Do NOT give generic advice.
Do NOT explain why.
Only 1 correction. Max 8 words.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": "You are a yoga therapist. Give only 1 short Hindi correction. Max 8 words. No explanation."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content
# ── 7 Day Plan ──────────────────────────────────

def get_plan(concern, level):
    prompt = f"""
You are an International Certified Yoga Therapist with 20+ years of clinical experience in Yoga Therapy, Naturopathy, and Rehabilitation.

You create evidence-based, clinically safe, and professionally structured yoga therapy programs.

==========================
USER PROFILE
==========================

Therapeutic Goal: {concern}
Experience Level: {level}

==========================
PROGRAM RULES
==========================

- Create EXACTLY 7 days.
- Every day must be EXACTLY 60 minutes.
- Difficulty increases gradually day by day.
- Follow correct yoga class sequence always.
- Choose poses strictly according to goal and level.
- Never prescribe advanced poses to beginners.
- Always provide a modification for each pose.

==========================
CLASS STRUCTURE
==========================

Warm-up & Joint Mobilization (8-10 min)
↓
Standing Asanas (15-20 min)
↓
Balancing Asanas (8-10 min)
↓
Forward Bend / Hip Opening (5-8 min)
↓
Back Bend (5-8 min)
↓
Twisting Pose (3-5 min)
↓
Pranayama & Bandha (8-10 min)
↓
Shavasana / Yoga Nidra (5-7 min ONLY)

Never exceed 7 minutes for Shavasana.

==========================
THERAPEUTIC PROTOCOLS
==========================

For Stress / Anxiety:
- Activate Parasympathetic Nervous System
- Include: Nadi Shodhana, Bhramari, Yoga Nidra
- Avoid: Fast-paced sequences, Kumbhaka for beginners
- Focus: Long exhalation (1:2 ratio inhale:exhale)

For Back Pain:
- Strengthen core first — Marjaryasana, Bitilasana
- Avoid: Deep backbends in acute phase, Paschimottanasana
- Include: Setu Bandhasana, Balasana, Supported Shavasana
- Focus: Spinal decompression and core stability

For Poor Sleep:
- Evening practice only — no stimulating poses
- Include: Viparita Karani, Supta Baddha Konasana, Yoga Nidra
- Avoid: Surya Namaskar, Strong Backbends, Kapalabhati
- Focus: Nervous system downregulation

For General Fitness:
- Balance strength and flexibility equally
- Include: Surya Namaskar A and B
- Progress: Week 1 static holds, Week 2 dynamic flow
- Focus: Full body conditioning

==========================
POSE SELECTION RULES
==========================

For Flexibility Goals:
- Tadasana, Trikonasana, Uttanasana
- Paschimottanasana, Janu Shirshasana

For Weight Loss:
- Surya Namaskar mandatory
- Dynamic Vinyasa sequences

For Back Pain:
- Avoid deep backbends
- Core strengthening priority

For Knee Pain:
- Avoid deep squats and Vajrasana
- Supported poses only

For Pregnancy:
- Only certified pregnancy-safe yoga
- No inversions, no supine after 20 weeks

NEVER use for Beginners:
- Shirshasana
- Pincha Mayurasana
- Hanumanasana
- Mayurasana

==========================
SAFETY PROTOCOLS
==========================

- Every pose must have a beginner modification
- Flag contraindications clearly in table
- Avoid prescribing for acute injuries
- Add note: consult doctor if pain increases
- Modifications must be genuinely simpler versions

==========================
OUTPUT FORMAT
==========================

Return ONLY Markdown. No extra text before Day 1.

For EVERY DAY generate exactly this structure:

---

## 📅 Day [N] — [Theme of the Day]

*Focus: [One line about what this day targets therapeutically]*

| Yoga Pose | Sanskrit Name | Duration | Rounds | Modification for Beginners | Contraindication |
|-----------|--------------|----------|--------|---------------------------|------------------|

> 💡 **Therapist Note:** [One clinical insight about today's practice — why this sequence helps the goal]

---

Rules:
- Duration column total = approximately 60 minutes per day.
- Modification = genuinely simpler version, not just "skip it".
- Contraindication = under 5 words.
- Theme must reflect therapeutic progression.
- Therapist Note must be clinically meaningful — not generic.

==========================
QUALITY CHECK
==========================

Before returning answer verify:

✓ Total duration ~60 minutes per day
✓ Pose order follows yoga teaching standards
✓ Difficulty increases day by day
✓ Poses match therapeutic goal
✓ Every pose has modification
✓ Therapist note is clinically relevant

Generate the complete 7-day yoga therapy program now.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": """You are a clinical Yoga Therapist with expertise in:
- Yoga Therapy for chronic conditions
- Naturopathy and integrative medicine  
- Evidence-based yoga research
- Therapeutic yoga sequencing

You create safe, realistic, and clinically informed yoga programs.
You never recommend poses that could harm the user.
You always prioritize safety over complexity."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# ── Main App ────────────────────────────────────
init_db()

st.set_page_config(page_title="YogaMirror", page_icon="🧘",layout="wide")
st.title("🧘 YogaMirror")
st.markdown("""
    <style>
    .stImage img {
        width: 100% !important;
        height: 80vh !important;
        object-fit: cover;
    }
    </style>
""", unsafe_allow_html=True)
st.caption("AI Yoga Assistant for Beginners")

# Sidebar
st.sidebar.header("Apne baare mein batao")
name = st.sidebar.text_input("Naam", "Yogi")
# ---------------- Health Goal ----------------

category = st.sidebar.selectbox(
    "🎯 Select Goal",
    [
        "Fitness",
        "Pain Relief",
        "Medical Conditions",
        "Mental Wellness",
        "Special Yoga",
        "Other"
    ]
)

if category == "Fitness":

    concern = st.sidebar.selectbox(
        "Choose Goal",
        [
            "General Fitness",
            "Weight Loss",
            "Weight Gain",
            "Flexibility",
            "Core Strength",
            "Balance Improvement",
            "Energy Boost"
        ]
    )

elif category == "Pain Relief":

    concern = st.sidebar.selectbox(
        "Choose Problem",
        [
            "Back Pain",
            "Lower Back Pain",
            "Neck Pain",
            "Shoulder Pain",
            "Knee Pain",
            "Sciatica",
            "Poor Posture"
        ]
    )

elif category == "Medical Conditions":

    concern = st.sidebar.selectbox(
        "Choose Condition",
        [
            "Diabetes",
            "High Blood Pressure",
            "PCOS / PCOD",
            "Thyroid",
            "Arthritis",
            "Digestive Issues",
            "Constipation",
            "Migraine"
        ]
    )

elif category == "Mental Wellness":

    concern = st.sidebar.selectbox(
        "Choose Goal",
        [
            "Stress",
            "Anxiety",
            "Depression",
            "Better Sleep",
            "Meditation",
            "Breathing Practice"
        ]
    )

elif category == "Special Yoga":

    concern = st.sidebar.selectbox(
        "Choose Goal",
        [
            "Pregnancy Yoga",
            "Post Pregnancy Recovery",
            "Senior Citizen Wellness",
            "Sports Recovery"
        ]
    )

else:

    concern = st.sidebar.text_input(
        "Enter Your Goal",
        placeholder="Example: Frozen Shoulder"
    )# ---------------- Health Goal ----------------

if concern == "Other":
    custom_concern = st.text_input(
        "Enter your concern",
        placeholder="Example: Cervical Spondylosis"
    )

    if custom_concern.strip():
        concern = custom_concern.strip()
level = st.sidebar.selectbox("Experience?", [
    "Bilkul beginner",
    "Thoda jaanta hoon",
    "Regular practice"
])

# Tabs
tab1, tab2, tab3 = st.tabs(["📋 My Plan", "📷 Practice", "📈 Progress"])

# ── TAB 1: PLAN ─────────────────────────────────
with tab1:
    st.subheader(f"Namaste {name}! 🙏")
    if st.button("Mera 7-Day Plan Banao"):
        with st.spinner("Plan ban raha hai..."):
            plan = get_plan(concern, level)
            st.success("Tumhara plan ready hai!")
            st.markdown(plan)

# ── TAB 2: PRACTICE ─────────────────────────────
# with tab2:
#     pose_choice = st.selectbox("Kaunsa pose karna hai?", [
#         "Tadasana",
#         "Vrikshasana",
#         "Balasana",
#         "Virabhadrasana I",
#         "Virabhadrasana II",
#         "Trikonasana",
#         "Bhujangasana",
#         "Adho Mukha Svanasana",
#         "Shavasana",
#         "Natarajasana"
#     ])

#     if st.button("Practice Shuru Karo"):
#         stframe = st.empty()
#         feedback_box = st.empty()
#         stop = st.button("Band Karo")

#         cap = cv2.VideoCapture(0)
#         frame_count = 0
#         feedback = "Pose karo..."
#         hold_time = 0

#         with mp_pose.Pose() as pose:
#             while cap.isOpened() and not stop:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = pose.process(rgb)

#                 if results.pose_landmarks:
#                     mp_draw.draw_landmarks(
#                         frame, results.pose_landmarks,
#                         mp_pose.POSE_CONNECTIONS)

#                     lm = results.pose_landmarks.landmark
#                     hip = [lm[23].x, lm[23].y]
#                     knee = [lm[25].x, lm[25].y]
#                     ankle = [lm[27].x, lm[27].y]
#                     knee_angle = calculate_angle(hip, knee, ankle)
#                     hold_time += 1

#                     frame_count += 1
#                     if frame_count % 60 == 0:
#                         feedback = get_ai_feedback( pose_choice, results.pose_landmarks, concern)

#                     cv2.putText(frame, f"AI: {feedback}",
#                                (20, 50),
#                                cv2.FONT_HERSHEY_SIMPLEX,
#                                0.7, (0, 255, 0), 2)

#                 stframe.image(frame, channels="BGR", use_container_width=True)
#                 feedback_box.info(f"💬 {feedback}")

#         cap.release()
#         seconds = hold_time // 30
#         save_session(pose_choice, seconds, 85.0)
#         st.success(f"Session save! {seconds} seconds hold kiya.")
with tab2:
    if IS_CLOUD:
        st.subheader("📷 Live Pose Practice")
        st.info("""
        🖥️ **Local Mode Required**
        
        Webcam feature sirf local machine pe kaam karta hai.
        
        **Local pe run karne ke liye:**
                git clone https://github.com/TUMHARA_USERNAME/yogamirror
                cd yogamirror
                pip install -r requirements.txt
                streamlit run app.py
                """)
        st.markdown("**Features jo locally kaam karte hain:**")
        st.markdown("""
        - 📸 Real-time body landmark detection (33 points)
        - 📐 Joint angle calculation
        - 🤖 AI correction in Hindi
        - 💾 Session progress saving
        """)
    else:
        pose_options = sorted([
            "Adho Mukha Svanasana",
            "Ananda Balasana",
            "Ardha Chandrasana",
            "Ardha Matsyendrasana",
            "Ashwa Sanchalanasana",
            "Baddha Konasana",
            "Bakasana",
            "Balasana",
            "Bhujangasana",
            "Bitilasana",
            "Chakrasana",
            "Chaturanga Dandasana",
            "Dandasana",
            "Dhanurasana",
            "Garudasana",
            "Gomukhasana",
            "Halasana",
            "Hanumanasana",
            "Janu Sirsasana",
            "Kapotasana",
            "Kurmasana",
            "Malasana",
            "Marjaryasana",
            "Matsyasana",
            "Mayurasana",
            "Natarajasana",
            "Naukasana",
            "Padahastasana",
            "Padmasana",
            "Parighasana",
            "Parivrtta Trikonasana",
            "Parsvakonasana",
            "Parsvottanasana",
            "Paschimottanasana",
            "Pavanamuktasana",
            "Pincha Mayurasana",
            "Plank Pose",
            "Purvottanasana",
            "Salabhasana",
            "Sarvangasana",
            "Shavasana",
            "Setu Bandhasana",
            "Simhasana",
            "Sukhasana",
            "Surya Namaskar",
            "Tadasana",
            "Trikonasana",
            "Urdhva Dhanurasana",
            "Ustrasana",
            "Utkatasana",
            "Uttanasana",
            "Uttitha Parsvakonasana",
            "Utthita Trikonasana",
            "Vajrasana",
            "Vasisthasana",
            "Veerabhadrasana I",
            "Veerabhadrasana II",
            "Veerabhadrasana III",
            "Viparita Karani",
            "Virasana",
            "Vrikshasana",
            "Other"
        ])
        pose_choice = st.selectbox(
            "🧘 Select Asana",
            pose_options
        )

        selected_pose = pose_choice

        if pose_choice == "Other":
            custom_pose = st.text_input(
                "Enter Asana Name",
                placeholder="Example: Yoga Mudrasana"
            )

            if custom_pose.strip():
                selected_pose = custom_pose.strip()
            

        col1, col2 = st.columns(2)
        with col1:
            start = st.button("▶️ Practice Shuru Karo")
        with col2:
            stop = st.button("⏹️ Band Karo")

        if start:
            st.session_state["running"] = True
            st.session_state["hold_time"] = 0
            st.session_state["frame_count"] = 0
            st.session_state["feedback"] = "Pose karo..."

        if stop and st.session_state.get("running"):
            seconds = st.session_state.get("hold_time", 0) // 30
            save_session(selected_pose, seconds, 85.0)
            st.session_state["running"] = False
            st.success(f"✅ Session save! {seconds} seconds hold kiya.")

        if st.session_state.get("running"):
            stframe = st.empty()
            feedback_box = st.empty()

            cap = cv2.VideoCapture(0)

            with mp_pose.Pose() as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)

                    if results.pose_landmarks:
                        mp_draw.draw_landmarks(
                            frame, results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS)

                        st.session_state["hold_time"] += 1
                        st.session_state["frame_count"] += 1

                        if st.session_state["frame_count"] % 60 == 0:
                            st.session_state["feedback"] = get_ai_feedback(
                                selected_pose, results.pose_landmarks, concern)

                        cv2.putText(frame,
                                f"AI: {st.session_state['feedback']}",
                                (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                        
                        stframe.image(frame, channels="BGR",
                                    use_container_width=True)
                        feedback_box.info(
                            f"💬 {st.session_state.get('feedback', 'Pose karo...')}")

            cap.release()

# ── TAB 3: PROGRESS ─────────────────────────────
# mp_pose = mp.solutions.pose
# mp_draw = mp.solutions.drawing_utils
# with tab3:
#     st.subheader("Tumhari Progress 📈")

#     if st.button("Refresh 🔄"):
#         st.rerun()

#     history = get_history()
#     st.write(f"Total sessions saved: {len(history)}")

#     if history:
#         dates = [row[0] for row in history]
#         durations = [row[2] for row in history]

#         fig, ax = plt.subplots()
#         ax.bar(dates, durations, color='#7EB8A4')
#         ax.set_xlabel("Date")
#         ax.set_ylabel("Seconds")
#         ax.set_title("Pose Hold Time Progress")
#         plt.xticks(rotation=45)
#         st.pyplot(fig)

#         st.dataframe({
#             "Date": dates,
#             "Pose": [row[1] for row in history],
#             "Duration (sec)": durations
#         })
#     else:
#         st.info("Abhi koi session nahi hai. Practice tab pe jaao!")

# ── TAB 3: PROGRESS ─────────────────────────────
with tab3:

    st.title("📈 Your Yoga Journey")

    if st.button("🔄 Refresh"):
        st.rerun()

    history = get_history()

    if history:

        dates = [row[0] for row in history]
        poses = [row[1] for row in history]
        durations = [row[2] for row in history]
        accuracies = [row[3] for row in history]

        total_sessions = len(history)
        total_time = sum(durations)
        avg_accuracy = round(sum(accuracies) / len(accuracies), 1)

        favourite_pose = max(set(poses), key=poses.count)

        # -------------------- TOP CARDS --------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "🧘 Sessions",
            total_sessions
        )

        c2.metric(
            "⏱ Total Time",
            f"{total_time} sec"
        )

        c3.metric(
            "⭐ Avg Accuracy",
            f"{avg_accuracy}%"
        )

        c4.metric(
            "🥇 Favourite",
            favourite_pose
        )

        st.divider()

        # -------------------- CHARTS --------------------

        left, right = st.columns(2)

        with left:

            st.subheader("📊 Practice Duration")

            fig, ax = plt.subplots(figsize=(7,4))

            ax.plot(
                range(len(durations)),
                durations,
                marker="o",
                linewidth=3
            )

            ax.set_xlabel("Session")
            ax.set_ylabel("Seconds")

            st.pyplot(fig)

        with right:

            st.subheader("🏆 Most Practiced Asanas")

            from collections import Counter

            pose_counts = Counter(poses)

            fig2, ax2 = plt.subplots(figsize=(7,4))

            ax2.barh(
                pose_counts.keys(),
                pose_counts.values()
            )

            st.pyplot(fig2)

        st.divider()

        # -------------------- RECENT SESSIONS --------------------

        st.subheader("📅 Recent Sessions")

        import pandas as pd

        df = pd.DataFrame({
            "📅 Date": dates,
            "🧘 Pose": poses,
            "⏱ Duration (sec)": durations,
            "⭐ Accuracy": accuracies
        })

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # -------------------- ACHIEVEMENTS --------------------

        st.subheader("🏅 Achievements")

        a1, a2, a3 = st.columns(3)

        if total_sessions >= 1:
            a1.success("🎉 First Session Completed")

        if total_sessions >= 10:
            a2.success("🔥 10 Sessions Completed")

        if len(set(poses)) >= 5:
            a3.success("🧘 5 Different Asanas Practiced")

    else:

        st.info("No sessions found. Practice your first pose! 🧘")