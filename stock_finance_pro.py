import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from openai import OpenAI

# -------------------- SETUP --------------------
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

nltk.download("vader_lexicon")
vader = SentimentIntensityAnalyzer()

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", None)
NEWS_KEY = st.secrets.get("NEWSAPI_KEY")

client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# -------------------- UI --------------------
st.title("📈 AI-Driven Stock Analysis Platform")

ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()

# -------------------- STOCK DATA --------------------
stock = yf.Ticker(ticker)

try:
    hist = stock.history(period="6mo")
    price = hist["Close"].iloc[-1]

    st.subheader(f"💰 Current Price: ${price:.2f}")

    st.line_chart(hist["Close"])

except Exception as e:
    st.error("Error loading stock data.")
    st.stop()

# -------------------- NEWS FETCH --------------------
st.subheader("📰 News Sentiment Analysis")

def fetch_news(ticker):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&"
        f"language=en&"
        f"sortBy=publishedAt&"
        f"pageSize=10&"
        f"apiKey={NEWS_KEY}"
    )
    return requests.get(url).json().get("articles", [])

articles = fetch_news(ticker)

if not articles:
    st.warning("No news articles found.")
    st.stop()

# -------------------- VADER SENTIMENT --------------------
sentiments = []

for article in articles:
    text = (article["title"] or "") + " " + (article["description"] or "")
    score = vader.polarity_scores(text)["compound"]
    sentiments.append(score)

avg_vader = np.mean(sentiments)

st.write(f"**VADER Average Sentiment:** `{avg_vader:.3f}`")

# -------------------- GPT SENTIMENT (OPTIONAL) --------------------
gpt_sentiment = None

if client:
    try:
        headlines = "\n".join([a["title"] for a in articles[:5]])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial sentiment analysis assistant."
                },
                {
                    "role": "user",
                    "content": f"""
Analyze the sentiment of these headlines about {ticker}.
Return a score from -1 (very negative) to +1 (very positive)
and briefly explain why.

{headlines}
"""
                }
            ]
        )

        gpt_sentiment = response.choices[0].message.content
        st.success("ChatGPT Sentiment Analysis:")
        st.write(gpt_sentiment)

    except Exception:
        st.warning("ChatGPT unavailable — using rule-based fallback.")

# -------------------- VALIDATION LOGIC --------------------
st.subheader("🔍 AI Validation & Oversight")

st.write("""
This system **does not rely on a single AI model**.

- **VADER** provides rule-based sentiment (objective baseline)
- **ChatGPT** provides contextual reasoning
- Discrepancies are flagged for human interpretation
""")

if avg_vader > 0.2:
    verdict = "Positive"
elif avg_vader < -0.2:
    verdict = "Negative"
else:
    verdict = "Neutral"

st.metric("Final Validated Sentiment", verdict)

# -------------------- AI Q&A --------------------
st.subheader("🤖 Ask AI About the Stock")

question = st.text_input("Ask a question")

if question:
    if client:
        try:
            answer = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial assistant. Use conservative language."
                    },
                    {
                        "role": "user",
                        "content": f"{question} (Stock: {ticker})"
                    }
                ]
            )
            st.write(answer.choices[0].message.content)

        except Exception:
            st.warning("AI unavailable — showing data-based fallback.")
            st.write(f"Current price of {ticker}: ${price:.2f}")

    else:
        st.warning("AI disabled — OpenAI key not found.")
        st.write(f"Current price of {ticker}: ${price:.2f}")
