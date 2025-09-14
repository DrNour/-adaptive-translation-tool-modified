import streamlit as st
from difflib import SequenceMatcher
import time, random
from langdetect import detect

# Optional packages
try:
    import sacrebleu
    sacrebleu_available = True
except ModuleNotFoundError:
    sacrebleu_available = False
    st.warning("sacrebleu not installed: BLEU/chrF/TER scoring disabled.")

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd_available = True
except ModuleNotFoundError:
    pd_available = False
    st.warning("pandas/seaborn/matplotlib not installed: Dashboard charts disabled.")

# =========================
# Streamlit Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool (Prize Version)")

# =========================
# Gamification / Leaderboard
# =========================
if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []
if "streak" not in st.session_state:
    st.session_state.streak = 0

def update_score(username, points):
    st.session_state.score += points
    if username not in st.session_state.leaderboard:
        st.session_state.leaderboard[username] = 0
    st.session_state.leaderboard[username] += points

# =========================
# Language Detection
# =========================
def detect_language(text):
    try:
        lang = detect(text)
        if lang.startswith("ar"):
            return "arabic"
        else:
            return "english"
    except:
        return "unknown"

# =========================
# Error Highlighting
# =========================
def highlight_diff(student, reference):
    matcher = SequenceMatcher(None, reference.split(), student.split())
    highlighted = ""
    feedback = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        stu_words = " ".join(student.split()[j1:j2])
        ref_words = " ".join(reference.split()[i1:i2])
        if tag == "equal":
            highlighted += f"<span style='color:green'>{stu_words} </span>"
        elif tag == "replace":
            highlighted += f"<span style='color:red'>{stu_words} </span>"
            feedback.append(f"Replace '{stu_words}' with '{ref_words}'")
        elif tag == "insert":
            highlighted += f"<span style='color:orange'>{stu_words} </span>"
            feedback.append(f"Extra words: '{stu_words}'")
        elif tag == "delete":
            highlighted += f"<span style='color:blue'>{ref_words} </span>"
            feedback.append(f"Missing: '{ref_words}'")
    return highlighted, feedback

# =========================
# Edit Distance (difflib)
# =========================
def edit_distance(a, b):
    matcher = SequenceMatcher(None, a, b)
    ratio = matcher.ratio()
    return int((1 - ratio) * max(len(a), len(b)))

# =========================
# Tabs
# =========================
username = st.text_input("Enter your name:")

tab1, tab2, tab3, tab4 = st.tabs(["Translate & Post-Edit", "Challenges", "Leaderboard", "Instructor Dashboard"])

# =========================
# Tab 1: Translate & Post-Edit
# =========================
with tab1:
    st.subheader("🔎 Translate or Post-Edit MT Output")
    source_text = st.text_area("Source Text")
    reference_translation = st.text_area("Reference Translation (Human Translation)")
    student_translation = st.text_area("Your Translation", height=150)

    challenge_time_limit = 180  # seconds
    start_time = time.time()
    
    if st.button("Evaluate Translation"):
        highlighted, fb = highlight_diff(student_translation, reference_translation)
        st.markdown(highlighted, unsafe_allow_html=True)
        
        st.subheader("💡 Feedback:")
        for f in fb:
            st.warning(f)

        # Scores
        if sacrebleu_available:
            bleu_score = sacrebleu.corpus_bleu([student_translation], [[reference_translation]]).score
            chrf_score = sacrebleu.corpus_chrf([student_translation], [[reference_translation]]).score
            ter_score = sacrebleu.corpus_ter([student_translation], [[reference_translation]]).score
            st.write(f"BLEU: {bleu_score:.2f}, chrF: {chrf_score:.2f}, TER: {ter_score:.2f}")

        dist = edit_distance(student_translation, reference_translation)
        st.write(f"Edit Distance (approx.): {dist}")

        elapsed_time = time.time() - start_time
        st.write(f"Time Taken: {elapsed_time:.2f} seconds")

        # Gamification points
        points = 10 + int(random.random()*10)
        if elapsed_time <= challenge_time_limit:
            bonus_points = 5
            st.success(f"Challenge completed in {elapsed_time:.1f}s! Bonus points: {bonus_points}")
            points += bonus_points
        update_score(username, points)
        st.success(f"Points earned: {points}")

        # Streak
        if dist < 5:  # successful small edit distance
            st.session_state.streak += 1
            streak_points = st.session_state.streak * 2
            update_score(username, streak_points)
            st.success(f"🔥 Current streak: {st.session_state.streak} exercises! Bonus points: {streak_points}")
        else:
            st.session_state.streak = 0

        # Store feedback
        st.session_state.feedback_history.append((username, [{"edit_distance": dist}]))

        # =========================
        # Suggested Exercises
        # =========================
        st.subheader("📝 Suggested Exercises Based on Feedback:")
        for idx, f in enumerate(fb):
            st.write(f"**Exercise {idx+1}:** {f}")
            corrected = st.text_input(f"Your attempt for Exercise {idx+1}:", key=f"exercise_{idx}")
            if st.button(f"Submit Exercise {idx+1}", key=f"submit_{idx}"):
                if corrected.strip():
                    st.success("Exercise submitted! Your correction will be evaluated.")
                    if sacrebleu_available:
                        new_bleu = sacrebleu.corpus_bleu([corrected], [[reference_translation]]).score
                        st.write(f"Updated BLEU after exercise: {new_bleu:.2f}")
                    new_dist = edit_distance(corrected, reference_translation)
                    st.write(f"Updated Edit Distance: {new_dist}")
                    # Extra points
                    extra_points = 5 + int(random.random()*5)
                    update_score(username, extra_points)
                    st.success(f"Extra points earned: {extra_points}")

# =========================
# Tab 2: Challenges (optional daily challenges)
# =========================
with tab2:
    st.subheader("🎯 Daily / Timer Challenges")
    st.info(f"Complete translation/post-edit tasks within {challenge_time_limit} seconds to earn bonus points!")
    st.write("You can attempt exercises in Tab 1 and earn streak/bonus points.")

# =========================
# Tab 3: Leaderboard
# =========================
with tab3:
    st.subheader("🏆 Leaderboard")
    if pd_available:
        leaderboard_df = pd.DataFrame(list(st.session_state.leaderboard.items()), columns=["Student", "Points"])
        st.dataframe(leaderboard_df.sort_values(by="Points", ascending=False))
    else:
        st.write(st.session_state.leaderboard)

# =========================
# Tab 4: Instructor Dashboard
# =========================
with tab4:
    st.subheader("📊 Instructor Dashboard")
    if not pd_available or len(st.session_state.feedback_history) == 0:
        st.info("Dashboard unavailable or no data yet.")
    else:
        records = []
        for student, fb_list in st.session_state.feedback_history:
            for fb in fb_list:
                records.append({
                    "Student": student,
                    "Edit Distance": fb.get("edit_distance", 0)
                })
        df = pd.DataFrame(records)
        avg_student = df.groupby("Student").mean().reset_index()
        st.write("Average Edit Distance per Student:")
        st.dataframe(avg_student)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x="Student", y="Edit Distance", data=avg_student, ax=ax)
        ax.set_title("Average Edit Distance per Student")
        plt.tight_layout()
        st.pyplot(fig)
