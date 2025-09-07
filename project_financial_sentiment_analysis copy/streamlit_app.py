import os
import streamlit as st
import pandas as pd
import requests
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config
MODEL_DIR = st.text_input("Model directory (local) or HF model id", value="notebooks/finbert_minty_model_v1")
MAX_LENGTH = 256

@st.cache_resource
def load_model_and_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tok, model

try:
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)
    model.to(DEVICE)
    st.title("Financial Sentiment Analyzer — Minty fine-tuned FinBERT")
    st.write("Model:", MODEL_DIR)
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {str(i): str(i) for i in range(model.config.num_labels)}
    # This line has been corrected
    st.write("ID to Label mapping:", id2label) 

except Exception as e:
    st.error(f"Failed to load model from {MODEL_DIR}: {e}")
    st.stop()


def score_texts(texts: list[str], tokenizer, model, T: float=1.0):
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        logits = logits / T
        probs = softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
    rows = []
    for t, p, pr in zip(texts, preds, probs):
        row = {"text": t, "pred_label": id2label[int(p)]}
        for i in range(pr.shape[0]):
            row[f"prob_{id2label[i]}"] = float(pr[i])
        rows.append(row)
    return pd.DataFrame(rows)

mode = st.sidebar.radio("Input mode", ["Manual text", "Alpha Vantage news"])

if mode == "Manual text":
    txt = st.text_area("Enter one or more headlines (one per line):", height=200)
    if st.button("Analyze"):
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if not lines:
            st.warning("Enter at least one non-empty line.")
        else:
            df = score_texts(lines, tokenizer, model, T=1.0)
            st.dataframe(df)

else:
    st.sidebar.write("Provide Alpha Vantage API key in .env as ALPHAVANTAGE_API_KEY")
    ticker = st.sidebar.text_input("Ticker(s):", value="AAPL")
    limit = st.sidebar.slider("Number of headlines", 5, 50, 10)
    if st.button("Fetch & Score"):
        AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
        if not AV_KEY:
            st.error("No Alpha Vantage API key found. Put ALPHAVANTAGE_API_KEY in .env")
        else:
            params = {"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": limit, "apikey": AV_KEY}
            resp = requests.get("https://www.alphavantage.co/query", params=params)
            data = resp.json()
            feed = data.get("feed", [])
            if not feed:
                st.warning("No feed returned. Check rate limits or ticker.")
                st.json(data)
            else:
                df_feed = pd.json_normalize(feed)
                df_feed["text"] = df_feed.get("title", "").fillna("") + ". " + df_feed.get("summary", "").fillna("")
                texts = df_feed["text"].tolist()
                df_scores = score_texts(texts, tokenizer, model, T=1.0)
                # Drop the 'text' column from the scores DataFrame before concatenating
                df_scores_no_text = df_scores.drop(columns=['text'])
                # Concatenate the original df_feed with the scores, which no longer has the redundant 'text' column
                out = pd.concat([df_feed.reset_index(drop=True), df_scores_no_text.reset_index(drop=True)], axis=1)
                st.dataframe(out)
                csv = out.to_csv(index=False)
                st.download_button("Download CSV", csv, file_name="news_scored.csv", mime="text/csv")