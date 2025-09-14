# --- Required Libraries ---
# pip install streamlit transformers sentence-transformers keybert torch numpy pandas

import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import pandas as pd
import re
import random
import time
import os

# --- Paths ---
LEADERBOARD_FILE = "leaderboard.csv"

# --- Load Models (cached for Cloud) ---
@st.cache_resource
def load_models():
    nli_model = pipeline("text-classification", model="joeddav/xlm-roberta-base-xnli")
    sts_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L6-v2')
    kw_model = KeyBERT()
    return nli_model, sts_model, kw_model

nli_model, sts_model, kw_model = load_models()

# --- Session State ---
if 'points' not in st.session_state:
    st.session_state.points = 0
if 'badges' not in st.session_state:
    st.session_state.badges = []
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'timer' not in st.session_state:
    st.session_state.timer = 300  # 5 mins

# --- Helper Functions ---
MAX_TOKENS = 1024

def truncate_text(text, max_words=500):
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

def stylistic_score(text):
    sentences = re.split(r'[.!?]', text)
    avg_len = sum(len(s.split()) for s in sentences)/max(len(sentences),1)
    punctuation_count = len(re.findall(r'[!?,;:]', text))
    return avg_len, punctuation_count

phrase_antonyms = {
    "i love you": ["i hate you", "i dislike you"],
    "good morning": ["bad morning", "sad morning"],
    "happy birthday": ["sad birthday", "unhappy birthday"]
}

def highlight_contradictions(source_text, translation_text):
    highlights = {}
    src_lower = source_text.lower()
    trans_lower = translation_text.lower()
    for phrase, options in phrase_antonyms.items():
        for opp in options:
            if phrase in src_lower and opp in trans_lower:
                highlights[opp] = phrase
    return highlights

def evaluate_translation(source_text, translation_text, top_n_keywords=10):
    results = {}
    source_trunc = truncate_text(source_text)
    translation_trunc = truncate_text(translation_text)
    
    # NLI Contradiction Check
    try:
        nli_input = f"{source_trunc} </s> {translation_trunc}"
        nli_result = nli_model(nli_input)[0]
        results['nli_label'] = nli_result['label']
        results['nli_score'] = round(nli_result['score'],3)
        results['contradiction_highlights'] = highlight_contradictions(source_text, translation_text)
        if nli_result['label'] == 'CONTRADICTION':
            results['warning'] = "⚠ Translation contradicts source!"
    except:
        results['nli_label'] = 'ERROR'
        results['nli_score'] = 0.0
    
    # Semantic similarity
    source_emb = sts_model.encode([source_text])
    trans_emb = sts_model.encode([translation_text])
    similarity = util.cos_sim(source_emb, trans_emb)[0][0].item()
    results['semantic_similarity'] = round(similarity,3)
    if similarity < 0.7:
        results['semantic_warning'] = "⚠ Low semantic similarity."
    
    # Literary / poetic keywords
    source_keywords = [kw for kw,_ in kw_model.extract_keywords(source_text, top_n=top_n_keywords)]
    trans_keywords = [kw for kw,_ in kw_model.extract_keywords(translation_text, top_n=top_n_keywords)]
    missing_keywords = [kw for kw in source_keywords if kw not in trans_keywords]
    results['missing_keywords'] = missing_keywords
    results['literary_highlights'] = missing_keywords
    if missing_keywords:
        results['literary_warning'] = f"⚠ Potential literary loss: {missing_keywords}"
    
    # Stylistic check
    src_len, src_punct = stylistic_score(source_text)
    trans_len, trans_punct = stylistic_score(translation_text)
    if abs(src_len-trans_len)>3 or abs(src_punct-trans_punct)>2:
        results['stylistic_warning'] = "⚠ Stylistic mismatch"

    return results

def highlight_translation(text, red_phrases, orange_phrases, yellow_phrases=[]):
    highlighted_text = text
    for phrase in red_phrases:
        highlighted_text = re.sub(re.escape(phrase), f'<span style="color:red;font-weight:bold;">{phrase}</span>', highlighted_text, flags=re.IGNORECASE)
    for phrase in orange_phrases:
        highlighted_text = re.sub(re.escape(phrase), f'<span style="color:orange;font-weight:bold;">{phrase}</span>', highlighted_text, flags=re.IGNORECASE)
    for phrase in yellow_phrases:
        highlighted_text = re.sub(re.escape(phrase), f'<span style="color:yellow;font-weight:bold;">{phrase}</span>', highlighted_text, flags=re.IGNORECASE)
    return highlighted_text

def award_points(results):
    points = 0
    if 'contradiction_highlights' in results and results['contradiction_highlights']:
        points += 10 * len(results['contradiction_highlights'])
    if 'literary_highlights' in results and results['literary_highlights']:
        points += 5 * len(results['literary_highlights'])
    if 'stylistic_warning' in results:
        points += 2
    st.session_state.points += points
    return points

def check_badges():
    badges = []
    if st.session_state.points >= 50 and "Poetry Master" not in st.session_state.badges:
        st.session_state.badges.append("Poetry Master")
        badges.append("Poetry Master")
    if st.session_state.points >= 100 and "Translation Expert" not in st.session_state.badges:
        st.session_state.badges.append("Translation Expert")
        badges.append("Translation Expert")
    return badges

def update_leaderboard(student_name, points):
    if os.path.exists(LEADERBOARD_FILE):
        df = pd.read_csv(LEADERBOARD_FILE)
    else:
        df = pd.DataFrame(columns=['Student','Points'])
    
    if student_name in df['Student'].values:
        df.loc[df['Student']==student_name, 'Points'] = max(points, df.loc[df['Student']==student_name, 'Points'].values[0])
    else:
        df = pd.concat([df, pd.DataFrame({'Student':[student_name], 'Points':[points]})], ignore_index=True)
    
    df.to_csv(LEADERBOARD_FILE, index=False)
    return df.sort_values('Points', ascending=False)

# --- Streamlit UI ---
st.title("Adaptive Translation Tool – Gamified Student Platform 🎮")

student_name = st.text_input("Enter your name")
exercise_mode = st.checkbox("Exercise Mode", value=True)

if exercise_mode:
    exercises = [
        {"source":"With love's light wings did I o'erperch these walls; For stony limits cannot hold love out…",
         "reference":"تسلقت هذه الجدران الحجرية كأنني أطير بجناحين ينساب من بينهما حبٌ متدفق، فلا تجد الجدران سبيلاً لمنعه."},
        {"source":"I am happy to see you today!","reference":"أنا سعيد لرؤيتك اليوم!"}
    ]
    selected = random.choice(exercises)
    source_text = st.text_area("Source Text", selected['source'])
    reference_text = selected['reference']
else:
    source_text = st.text_area("Source Text", "")
    reference_text = None

translation_text = st.text_area("Your Translation", "")

# Timer display (non-blocking)
if st.button("Start Timer Challenge"):
    st.session_state.start_time = time.time()
    st.session_state.timer = 300
    st.success("Timer started: 5 minutes!")

if st.session_state.start_time:
    elapsed = int(time.time() - st.session_state.start_time)
    remaining = max(0, st.session_state.timer - elapsed)
    mins, secs = divmod(remaining, 60)
    st.info(f"Time left: {mins:02d}:{secs:02d}")
    st.progress((300 - remaining)/300)

if st.button("Evaluate Translation") and student_name.strip():
    if not source_text.strip() or not translation_text.strip():
        st.warning("Provide both source and translation.")
    else:
        st.session_state.attempts += 1
        results = evaluate_translation(source_text, translation_text)

        st.subheader("Evaluation Results:")
        for key, value in results.items():
            if isinstance(value,list) and len(value)==0:
                continue
            st.write(f"**{key}:** {value}")

        red_phrases = list(results.get('contradiction_highlights', {}).keys())
        orange_phrases = results.get('literary_highlights', [])
        yellow_phrases = []

        highlighted_text = highlight_translation(translation_text, red_phrases, orange_phrases, yellow_phrases)
        if red_phrases or orange_phrases or yellow_phrases:
            st.subheader("In-Line Highlights 🔴🟠🟡")
            st.markdown(highlighted_text, unsafe_allow_html=True)

        st.subheader("Post-Editing Suggestions ✏️")
        for word in red_phrases:
            st.write(f"**{word}** might contradict the source. Suggested fixes: {phrase_antonyms.get(word, 'Check meaning')}")
        for word in orange_phrases:
            st.write(f"**{word}** might be missing literary imagery. Try preserving metaphors or key expressions.")

        # Gamification
        earned_points = award_points(results)
        st.success(f"Points earned this attempt: {earned_points} | Total Points: {st.session_state.points}")

        earned_badges = check_badges()
        if earned_badges:
            st.balloons()
            st.success(f"New Badge Unlocked: {', '.join(earned_badges)} 🎉")

        st.info(f"Attempts this session: {st.session_state.attempts}")

        # Leaderboard
        leaderboard = update_leaderboard(student_name, st.session_state.points)
        st.subheader("Leaderboard 🏆")
        st.dataframe(leaderboard)
