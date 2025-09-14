import streamlit as st
from difflib import SequenceMatcher

# Session state
if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}

st.title("Adaptive Translation Tool – Minimal Safe Version")

username = st.text_input("Enter your name:")

source_text = st.text_area("Source Text")
reference_translation = st.text_area("Reference Translation")
student_translation = st.text_area("Your Translation")

# Function to calculate edit distance
def edit_distance(a, b):
    matcher = SequenceMatcher(None, a, b)
    ratio = matcher.ratio()
    return int((1 - ratio) * max(len(a), len(b)))

if st.button("Evaluate Translation"):
    if not username.strip():
        st.warning("Please enter your name.")
    elif not student_translation.strip() or not reference_translation.strip():
        st.warning("Please provide both your translation and reference translation.")
    else:
        # Edit distance
        dist = edit_distance(student_translation, reference_translation)
        st.write(f"Edit Distance: {dist}")

        # Simple points system
        points = max(0, 10 - dist)
        st.session_state.score += points
        st.success(f"Points earned: {points}")
        st.info(f"Total Score: {st.session_state.score}")

        # Update leaderboard
        st.session_state.leaderboard[username] = st.session_state.score

# Display leaderboard
st.subheader("🏆 Leaderboard")
for user, score in sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True):
    st.write(f"{user}: {score} points")
