import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import re

# --- Load Models ---
@st.cache_resource
def load_models():
    nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")
    sts_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    kw_model = KeyBERT()
    return nli_model, sts_model, kw_model

nli_model, sts_model, kw_model = load_models()

# --- Helper Functions ---
def stylistic_score(text):
    sentences = re.split(r'[.!?]', text)
    avg_len = sum(len(s.split()) for s in sentences)/max(len(sentences),1)
    punctuation_count = len(re.findall(r'[!?,;:]', text))
    return avg_len, punctuation_count

# Phrase-level contradiction dictionary
phrase_antonyms = {
    "i love you": "i hate you",
    "good morning": "bad morning",
    "happy birthday": "sad birthday"
}

def highlight_contradictions(source_text, translation_text):
    highlights = {}
    src_lower = source_text.lower()
    trans_lower = translation_text.lower()
    for phrase, opp in phrase_antonyms.items():
        if phrase in src_lower and opp in trans_lower:
            highlights[opp] = phrase
    return highlights

def evaluate_translation(source_text, translation_text, top_n_keywords=10):
    results = {}

    # NLI Contradiction Check
    nli_input = f"{source_text} </s> {translation_text}"
    nli_result = nli_model(nli_input)[0]
    results['nli_label'] = nli_result['label']
    results['nli_score'] = round(nli_result['score'],3)
    results['contradiction_highlights'] = highlight_contradictions(source_text, translation_text)
    if nli_result['label'] == 'CONTRADICTION':
        results['warning'] = "⚠ Translation contradicts source!"

    # Semantic Similarity
    source_emb = sts_model.encode([source_text])
    trans_emb = sts_model.encode([translation_text])
    similarity = util.cos_sim(source_emb, trans_emb)[0][0].item()
    results['semantic_similarity'] = round(similarity,3)
    if similarity < 0.7:
        results['semantic_warning'] = "⚠ Low semantic similarity - meaning may be lost."

    # Literary/Poetic Keywords
    source_keywords = [kw for kw,_ in kw_model.extract_keywords(source_text, top_n=top_n_keywords)]
    trans_keywords = [kw for kw,_ in kw_model.extract_keywords(translation_text, top_n=top_n_keywords)]
    missing_keywords = [kw for kw in source_keywords if kw not in trans_keywords]
    results['missing_keywords'] = missing_keywords
    results['literary_highlights'] = missing_keywords
    if missing_keywords:
        results['literary_warning'] = f"⚠ Potential loss of imagery/poetic elements: {missing_keywords}"

    # Stylistic Proxy
    src_len, src_punct = stylistic_score(source_text)
    trans_len, trans_punct = stylistic_score(translation_text)
    if abs(src_len-trans_len)>3 or abs(src_punct-trans_punct)>2:
        results['stylistic_warning'] = "⚠ Potential stylistic/rhythm mismatch"

    return results

# --- Highlighting Function ---
def highlight_translation(text, red_phrases, orange_phrases):
    highlighted_text = text
    # Red for contradictions
    for phrase in red_phrases:
        highlighted_text = re.sub(re.escape(phrase), f'<span style="color:red;font-weight:bold;">{phrase}</span>', highlighted_text, flags=re.IGNORECASE)
    # Orange for missing literary expressions
    for phrase in orange_phrases:
        highlighted_text = re.sub(re.escape(phrase), f'<span style="color:orange;font-weight:bold;">{phrase}</span>', highlighted_text, flags=re.IGNORECASE)
    return highlighted_text

# --- Streamlit UI ---
st.title("Adaptive Translation Tool v5 🌐")
st.write("Detect semantic errors, contradictions, and missing literary expressions with phrase-level highlighting.")

source_text = st.text_area("Source Text", "")
translation_text = st.text_area("Student / Machine Translation", "")

if st.button("Evaluate Translation"):
    if not source_text.strip() or not translation_text.strip():
        st.warning("Please provide both source and translation texts.")
    else:
        results = evaluate_translation(source_text, translation_text)

        st.subheader("Evaluation Results:")
        for key, value in results.items():
            if isinstance(value, list) and len(value)==0:
                continue
            st.write(f"**{key}:** {value}")

        # In-line phrase highlighting
        red_phrases = list(results.get('contradiction_highlights', {}).keys())
        orange_phrases = results.get('literary_highlights', [])
        highlighted_text = highlight_translation(translation_text, red_phrases, orange_phrases)

        if red_phrases or orange_phrases:
            st.subheader("In-Line Highlights 🔴🟠")
            st.markdown(highlighted_text, unsafe_allow_html=True)
