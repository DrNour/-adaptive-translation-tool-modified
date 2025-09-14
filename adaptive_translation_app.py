import streamlit as st
from difflib import SequenceMatcher
import sacrebleu
import Levenshtein
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import MarianMTModel, MarianTokenizer
from collections import Counter

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")
st.write("English ↔ Arabic translation with MT, post-editing, gamification, and instructor analytics.")

# =========================
# Load MT Model
# =========================
@st.cache_resource
def load_mt_model():
    model_name = "Helsinki-NLP/opus-mt-en-ar"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_mt_model()

def translate_mt(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# Gamification / Leaderboard
# =========================
if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

def update_score(username, points):
    st.session_state.score += points
    if username not in st.session_state.leaderboard:
        st.session_state.leaderboard[username] = 0
    st.session_state.leaderboard[username] += points

# =========================
# User Input
# =========================
username = st.text_input("Enter your name to start:")

tab1, tab2, tab3, tab4 = st.tabs(["Translate & Post-Edit", "Challenges", "Leaderboard", "Instructor Dashboard"])

# =========================
# Error Highlighting Function
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
# Tab 1: Translate & Post-Edit
# =========================
with tab1:
    st.subheader("🔎 Translate or Post-Edit MT Output")
    source_text = st.text_area("Source Text")
    reference_translation = st.text_area("Reference Translation (Human Translation)")
    
    if st.button("Generate MT Output"):
        if source_text:
            mt_output = translate_mt(source_text)
            st.session_state.mt_output = mt_output
            st.success("Machine Translation Generated!")
    
    student_translation = st.text_area("Edit MT Output Here", value=st.session_state.get("mt_output", ""), height=150)
    start_time = time.time()
    
    if st.button("Evaluate Translation"):
        if source_text and student_translation and reference_translation:
            elapsed_time = time.time() - start_time
            highlighted, fb = highlight_diff(student_translation, reference_translation)
            
            # Display highlighted text
            st.markdown(highlighted, unsafe_allow_html=True)
            
            # Feedback
            st.subheader("💡 Feedback:")
            for f in fb:
                st.warning(f)
            
            # Scoring
            bleu_score = sacrebleu.corpus_bleu([student_translation], [[reference_translation]]).score
            chrf_score = sacrebleu.corpus_chrf([student_translation], [[reference_translation]]).score
            ter_score = sacrebleu.corpus_ter([student_translation], [[reference_translation]]).score
            edit_dist = Levenshtein.distance(student_translation, reference_translation)
            st.write(f"BLEU: {bleu_score:.2f}, chrF: {chrf_score:.2f}, TER: {ter_score:.2f}, Edit Distance: {edit_dist}")
            st.write(f"Time Taken: {elapsed_time:.2f} seconds")
            
            # Points
            points = int(chrf_score + bleu_score - edit_dist + max(0, 50 - int(elapsed_time)))
            update_score(username, points)
            st.success(f"Points earned: {points}")
            
            # Record feedback for instructor
            st.session_state.feedback_history.append(fb)
            
            # Suggested exercises
            st.subheader("📝 Suggested Exercises:")
            if any("Replace" in f for f in fb):
                st.info("Try replacing the highlighted words with more accurate or idiomatic expressions.")
            if any("Extra words" in f for f in fb):
                st.info("Remove unnecessary words to make your sentence concise.")
            if any("Missing" in f for f in fb):
                st.info("Add the missing words to match the reference meaning.")

# =========================
# Tab 2: Challenges
# =========================
with tab2:
    st.subheader("⏱️ Timer Challenge Mode")
    challenges = [
        ("I love you.", "أنا أحبك."),
        ("Knowledge is power.", "المعرفة قوة."),
        ("The weather is nice today.", "الطقس جميل اليوم.")
    ]
    
    if st.button("Start Challenge"):
        challenge = random.choice(challenges)
        st.session_state.challenge = challenge
        st.write(f"Translate this: **{challenge[0]}**")
    
    if "challenge" in st.session_state:
        user_ans = st.text_area("Your Translation (Challenge Mode)", key="challenge_box")
        if st.button("Submit Challenge"):
            elapsed = time.time() - start_time
            highlighted, fb = highlight_diff(user_ans, st.session_state.challenge[1])
            st.markdown(highlighted, unsafe_allow_html=True)
            
            chrf_score = sacrebleu.corpus_chrf([user_ans], [[st.session_state.challenge[1]]]).score
            edit_dist = Levenshtein.distance(user_ans, st.session_state.challenge[1])
            st.write(f"chrF Score: {chrf_score:.2f}, Edit Distance: {edit_dist}")
            st.write(f"Time Taken: {elapsed:.2f} sec")
            
            points = int(chrf_score - edit_dist + max(0, 50 - int(elapsed)))
            update_score(username, points)
            st.success(f"Points earned: {points}")
            
            st.subheader("📝 Suggested Exercises:")
            if any("Replace" in f for f in fb):
                st.info("Replace highlighted words for accuracy/idiomatic usage.")
            if any("Extra words" in f for f in fb):
                st.info("Remove unnecessary words for conciseness.")
            if any("Missing" in f for f in fb):
                st.info("Add missing words to preserve meaning.")

# =========================
# Tab 3: Leaderboard
# =========================
with tab3:
    st.subheader("🏆 Leaderboard")
    if st.session_state.leaderboard:
        sorted_lb = sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True)
        for rank, (user, points) in enumerate(sorted_lb, start=1):
            st.write(f"{rank}. **{user}** - {points} points")
    else:
        st.info("No scores yet. Start translating to enter the leaderboard!")

# =========================
# Tab 4: Instructor Dashboard
# =========================
with tab4:
    st.subheader("📊 Instructor Dashboard")
    
    if st.session_state.leaderboard:
        df = pd.DataFrame([
            {"Student": user, "Points": points} 
            for user, points in st.session_state.leaderboard.items()
        ])
        st.dataframe(df)
        st.bar_chart(df.set_index("Student")["Points"])
        
        # Error Analysis
        feedback_list = st.session_state.feedback_history
        all_errors = [f for sublist in feedback_list for f in sublist]
        if all_errors:
            counter = Counter(all_errors)
            error_df = pd.DataFrame(counter.items(), columns=["Error", "Count"]).sort_values(by="Count", ascending=False)
            st.subheader("Common Errors Across Class")
            st.table(error_df.head(10))
            
            # Heatmap
            plt.figure(figsize=(10,6))
            sns.barplot(data=error_df.head(10), x="Count", y="Error")
            st.pyplot(plt)
    else:
        st.info("No student activity yet. Have students complete exercises first.")
