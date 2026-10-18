"""Streamlit demo: SST-2 sentiment from a GaLore-tuned and an AdamW-tuned RoBERTa-base.

Run from the repo root:  streamlit run app/app.py
Expects the saved models in models/adamw and models/galore_r128.
"""
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = {
    "GaLore (rank 128)": "models/galore_r128",
    "AdamW baseline": "models/adamw",
}
LABELS = {0: "Negative", 1: "Positive"}

st.set_page_config(page_title="GaLore SST-2 demo", layout="centered")
st.title("Sentiment demo: GaLore vs AdamW")
st.caption("RoBERTa-base fine-tuned on SST-2. Both models run locally on CPU.")


@st.cache_resource
def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tok, model


def predict(path, text):
    tok, model = load(path)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    idx = int(probs.argmax())
    return LABELS[idx], float(probs[idx])


text = st.text_area("Type a movie-review style sentence", "A warm, funny and beautifully acted film.")
if st.button("Classify") and text.strip():
    cols = st.columns(2)
    for col, (name, path) in zip(cols, MODELS.items()):
        with col:
            st.subheader(name)
            try:
                label, conf = predict(path, text)
                st.metric("Prediction", label)
                st.progress(conf, text=f"Confidence {conf:.1%}")
            except Exception as e:
                st.error(f"Could not load {path}: {e}")
