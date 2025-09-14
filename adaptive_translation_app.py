import streamlit as st
import requests
import random
import time

# =========================
# Hugging Face API Setup
# =========================
HF_TOKEN = "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"  # your token
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

API_URL_NLI = "https://api-inference.huggingface.co/models/microsoft/deberta-base-mnli"
API_URL_SIM = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def query(payload, api_url):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

# =========================
# Evaluation Functions
# =========================
def evaluate_translation(source, translation):
    """Evaluate translation using NLI + semantic similarity"""
    # NLI Check
    nli_input = {"inputs": {"premise": source, "hypothesis": translation}}
    nli_result = query(nli_input, API_URL_NLI)

    # Similarity Score
    sim_input = {"inputs": {"source_sentence": source, "sentences": [translation]}}
    sim_result = query(sim_input, API_URL_SIM)

    similarity = sim_result[0] if isinstance(sim_result, list) else 0.0
    label = nli_result[0]["label"] if isinstance(nli_result, list) else "UNKNOWN"

    return {
        "similarity": round(similarity, 3),
        "nli_label": label,
        "feedback": generate_feedback(source, translation, similarity, label)
    }

def generate_feedback(source, translation, similarity, label):
    """Student-friendly feedback with post-editing suggestions"""
    feedback = []
    if label == "CONTRADICTION":
        feedback.append("⚠️ The meaning contradicts the source. Double-check word choices.")
    if similarity < 0.6:
        feedback.append("🔄 Low similarity. Revise structure and preserve key ideas.")
    if 0.6 <= similarity < 0.85:
        feedback.append("✨ Good attempt, but polish idiomatic style and cohesion.")
    if similarity >= 0.85 and label == "ENTAILMENT":
        feedback.append("✅ Strong translation! Accurate and faithful to the source.")
    return feedback

# =========================
# Gamification
# =========================
if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}

def update_score(username, points):
    st.session_state.score += points
    if username not in st.session_state.leaderboard:
        st.session_state.leaderboard[username] = 0
    st.session_state.leaderboard[username] += points

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")

st.title("🌍 Adaptive Translation & Post-Editing Tool")
st.write("Bidirectional English ↔ Arabic translation evaluation with gamification, feedback, and leaderboard.")

username = st.text_input("Enter your name to start:")

tab1, tab2, tab3 = st.tabs(["Translate & Evaluate", "Challenges", "Leaderboard"])

with tab1:
    st.subheader("🔎 Translate & Get Feedback")
    source_text = st.text_area("Source Text")
    translation_text = st.text_area("Your Translation")

    if st.button("Evaluate Translation"):
        if source_text and translation_text:
            with st.spinner("Evaluating..."):
                results = evaluate_translation(source_text, translation_text)
            st.success(f"Similarity Score: {results['similarity']}")
            st.info(f"NLI Label: {results['nli_label']}")
            for fb in results["feedback"]:
                st.warning(fb)
            update_score(username, int(results["similarity"] * 100))

with tab2:
    st.subheader("⏱️ Timer Challenge Mode")
    challenges = [
        ("I love you.", "أنا أحبك."),
        ("Knowledge is power.", "المعرفة قوة."),
        ("The weather is nice today.", "الطقس جميل اليوم.")
    ]
    if st.button("Start Challenge"):
        challenge = random.choice(challenges)
        st.write(f"Translate this: **{challenge[0]}**")
        start_time = time.time()
        user_ans = st.text_area("Your Translation (Challenge Mode)")
        if st.button("Submit Challenge"):
            elapsed = time.time() - start_time
            results = evaluate_translation(challenge[0], user_ans)
            st.write(f"⏳ Time Taken: {elapsed:.2f} sec")
            st.write(f"Similarity: {results['similarity']}")
            for fb in results["feedback"]:
                st.warning(fb)
            bonus = max(0, 50 - int(elapsed))
            update_score(username, int(results["similarity"] * 100) + bonus)

with tab3:
    st.subheader("🏆 Leaderboard")
    if st.session_state.leaderboard:
        sorted_lb = sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True)
        for rank, (user, points) in enumerate(sorted_lb, start=1):
            st.write(f"{rank}. **{user}** - {points} points")
    else:
        st.info("No scores yet. Start translating to enter the leaderboard!")
