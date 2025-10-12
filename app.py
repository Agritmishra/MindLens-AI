# app.py
import streamlit as st

st.set_page_config(
    page_title="MindLens – Your Reflection Companion",
    page_icon="🧠",
    layout="centered",
)

import textwrap
from insight_engine import InsightEngine

MODEL_CLASSIFIER = "MODEL_CLASSIFIER = "Agritmishra/emotion-tiny-distilbert"
MODEL_SUMMARIZER = None 

# Create engine,cached resource so it doesn't re-load on every rerun
@st.cache_resource
def get_engine():
    return InsightEngine(classifier_model=MODEL_CLASSIFIER, summarizer_model=MODEL_SUMMARIZER)


engine = get_engine()

# Page UI
st.markdown(
    """
    <style>
    body {background: #ffffff;}
    .main {max-width:900px;margin:0 auto;}
    .title {font-size:28px;font-weight:700;text-align:center;margin-bottom:4px;}
    .subtitle {color:#555;text-align:center;margin-bottom:18px;}
    .card {background:#fff;padding:1rem;border-radius:12px;box-shadow:0 6px 18px rgba(0,0,0,0.06);margin-bottom:1rem}
    .big-input {background:#f6f7f9;border-radius:10px;padding:10px;}
    .badge {display:inline-block;padding:6px 10px;border-radius:8px;background:#eef7ff;color:#0b5394;font-weight:600}
    .result-box {background:#fff;border-radius:12px;padding:18px;margin-top:12px}
    .error {background:#ffe8e8;border-left:4px solid #ff6b6b;padding:14px;border-radius:8px;}
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<div class='title'>🧠 MindLens — Your Reflection Companion</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Write how you're feeling and get a short insight, gentle reflection prompts and micro-actions.</div>", unsafe_allow_html=True)

with st.form("reflection_form"):
    text = st.text_area("💬 Write your thoughts:", height=220, placeholder="Example: I've been working hard and feel tired but also proud...", key="input_text")
    submitted = st.form_submit_button("✨ Analyze My Thoughts")

# small note about model availability
if engine.classifier is None and engine.zero_shot is None:
    st.warning("Model classifiers not found — MindLens will use lightweight heuristics. (Upload a fine-tuned model for better accuracy.)")

# Analyze when user submits
if submitted:
    # ensure text not empty
    if not text or not text.strip():
        st.error("Please write a few sentences so MindLens can analyze them.")
    else:
        with st.spinner("Analyzing — one moment..."):
            result = engine.analyze(text)

        # give output in a white card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;align-items:center;gap:12px;'><div style='font-size:20px'>🔮</div><div style='font-weight:700'>Summary</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:8px'>{result.get('summary','')}</div>", unsafe_allow_html=True)

        mood = result.get("mood", "neutral")
        mood_conf = result.get("mood_confidence", 0.0)
        balance = result.get("balance_score", "N/A")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex;gap:12px;align-items:center'><div class='badge'>Mood: {mood.title()}</div>"
                    f"<div style='color:#666'>Confidence: {round(mood_conf*100,1) if isinstance(mood_conf,float) else mood_conf}%</div>"
                    f"<div style='margin-left:auto;color:#333'>Balance Score: <b>{balance}</b>/100</div></div>", unsafe_allow_html=True)

        # Reflection prompts
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("### 🪞 Reflection prompts")
        for p in result.get("reflection_prompts", []):
            st.markdown(f"- {p}")
        st.markdown("### 🌱 Micro-actions")
        for a in result.get("micro_actions", []):
            st.markdown(f"- {a}")

        st.markdown("### 💡 Practical suggestions (concise)")
        for s in result.get("suggestions", []):
            st.markdown(f"- {s}")

        st.markdown("</div></div>", unsafe_allow_html=True) 

# helpful note
st.markdown("<div style='text-align:center;color:#777;margin-top:18px;font-size:13px;'> states like happiness/joyfulness may show in optimistic/neutral category </div>", unsafe_allow_html=True)
