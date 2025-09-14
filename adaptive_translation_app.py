import streamlit as st
from difflib import SequenceMatcher
import time, random

# Optional packages
try:
    import sacrebleu
    sacrebleu_available = True
except ModuleNotFoundError:
    sacrebleu_available = False
    st.warning("sacrebleu not installed: BLEU/chrF/TER scoring disabled.")

# Session state
if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "streak" not in st.session_state:
    st.session_state.streak = 0

st.title("Adaptive Translation Tool – Step 3 (Gamification + Post-Editing)")

username = st.text_input("Enter your name:")

source_text = st.text_area("Source Text")
reference_translation = st.text_area("Reference Translation")
student_translation = st.text_area("Your Translation")

challenge_time_limit = 180  # seconds
start_time = time.time()

# Highlight differences
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

def edit_distance(a, b):
    matcher = SequenceMatcher(None, a, b)
    ratio = matcher.ratio()
    return int((1 - ratio) * max(len(a), len(b)))

if st.button("Evaluate Translation"):
    if not student_translation.strip() or not reference_translation.strip():
        st.warning("Please provide both your translation and reference translation.")
    else:
        # Highlight & feedback
        highlighted, fb = highlight_diff(student_translation, reference_translation)
        st.markdown(highlighted, unsafe_allow_html=True)

        st.subheader("Feedback:")
        for f in fb:
            st.warning(f)

        # Scores
        dist = edit_distance(student_translation, reference_translation)
        st.write(f"Edit Distance: {dist}")

        if sacrebleu_available:
            try:
                bleu = sacrebleu.corpus_bleu([student_translation], [[reference_translation]]).score
                st.write(f"BLEU: {bleu:.2f}")
            except:
                st.warning("BLEU calculation skipped.")

        # Timer & points
        elapsed_time = time.time() - start_time
        points = max(0, 10 - dist)
        if elapsed_time <= challenge_time_limit:
            bonus = 5
            points += bonus
            st.success(f"Bonus points for fast submission: {bonus}")
        st.session_state.score += points
        st.success(f"Points earned: {points}")
        st.info(f"Total Score: {st.session_state.score}")

        # Streaks
        if dist < 5:
            st.session_state.streak += 1
            streak_bonus = st.session_state.streak * 2
            st.session_state.score += streak_bonus
            st.success(f"🔥 Current streak: {st.session_state.streak} exercises! Bonus points: {streak_bonus}")
        else:
            st.session_state.streak = 0

        # Update leaderboard
        if username:
            st.session_state.leaderboard[username] = st.session_state.score

        # Post-Editing Exercises
        st.subheader("📝 Suggested Exercises:")
        for idx, f in enumerate(fb):
            st.write(f"Exercise {idx+1}: {f}")
            corrected = st.text_input(f"Your correction:", key=f"exercise_{idx}")
            if st.button(f"Submit Exercise {idx+1}", key=f"submit_{idx}"):
                if corrected.strip():
                    st.success("Exercise submitted!")
                    new_dist = edit_distance(corrected, reference_translation)
                    st.write(f"Updated Edit Distance: {new_dist}")
                    extra_points = max(0, 5 - new_dist)
                    st.session_state.score += extra_points
                    st.success(f"Extra points earned: {extra_points}")
                    if username:
                        st.session_state.leaderboard[username] = st.session_state.score

# Leaderboard display
st.subheader("🏆 Leaderboard")
for user, score in sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True):
    st.write(f"{user}: {score} points")
