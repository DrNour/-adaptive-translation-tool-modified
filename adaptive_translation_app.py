import streamlit as st
from difflib import SequenceMatcher
import time, random
from langdetect import detect
import torch

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

# Transformers for semantic and fluency
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    # Cross-lingual NLI
    nli_model = pipeline("text-classification", model="joeddav/xlm-roberta-large-xnli")
    # English fluency
    en_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    en_model = AutoModelForCausalLM.from_pretrained("gpt2")
    # Arabic fluency
    ar_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/ara-gpt2")
    ar_model = AutoModelForCausalLM.from_pretrained("aubmindlab/ara-gpt2")
    semantic_fluency_available = True
except:
    semantic_fluency_available = False
    st.warning("Semantic accuracy / fluency scoring unavailable (transformers missing).")

# =========================
# Streamlit Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

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
    ratio = matcher.ratio()  # Similarity ratio 0-1
    return int((1 - ratio) * max(len(a), len(b)))

# =========================
# Semantic Accuracy
# =========================
def semantic_score(source, translation):
    if not semantic_fluency_available:
        return None
    result = nli_model(f"{source} </s></s> {translation}")[0]
    if result['label'] == 'ENTAILMENT':
        score = result['score'] * 100
    elif result['label'] == 'CONTRADICTION':
        score = 0
    else:
        score = result['score'] * 50
    return score

# =========================
# Fluency Scoring
# =========================
def fluency_score(text):
    if not semantic_fluency_available:
        return None
    lang = detect_language(text)
    if lang == "english":
        tokenizer, model = en_tokenizer, en_model
    elif lang == "arabic":
        tokenizer, model = ar_tokenizer, ar_model
    else:
        return None
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    score = 100 / (1 + loss.item())
    return score

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

        # Semantic & Fluency
        if semantic_fluency_available:
            sem_score = semantic_score(source_text, student_translation)
            flu_score = fluency_score(student_translation)
            st.write(f"Semantic Accuracy: {sem_score:.2f}")
            st.write(f"Fluency: {flu_score:.2f}")

        elapsed_time = time.time() - start_time
        st.write(f"Time Taken: {elapsed_time:.2f} seconds")

        # Gamification points
        points = 10 + int(random.random()*10)
        update_score(username, points)
        st.success(f"Points earned: {points}")

        # Store feedback
        st.session_state.feedback_history.append((username, [{"semantic": sem_score, "fluency": flu_score}]))

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
                    "Semantic Accuracy": fb.get("semantic", 0),
                    "Fluency": fb.get("fluency", 0)
                })
        df = pd.DataFrame(records)
        avg_student = df.groupby("Student").mean().reset_index()
        st.write("Average Semantic Accuracy & Fluency per Student:")
        st.dataframe(avg_student)
        class_avg = df[["Semantic Accuracy", "Fluency"]].mean()
        st.write(f"Class Average Semantic Accuracy: {class_avg['Semantic Accuracy']:.2f}")
        st.write(f"Class Average Fluency: {class_avg['Fluency']:.2f}")
        fig, ax = plt.subplots(1, 2, figsize=(12,5))
        sns.barplot(x="Student", y="Semantic Accuracy", data=avg_student, ax=ax[0])
        ax[0].set_title("Semantic Accuracy per Student")
        sns.barplot(x="Student", y="Fluency", data=avg_student, ax=ax[1])
        ax[1].set_title("Fluency per Student")
        plt.tight_layout()
        st.pyplot(fig)
