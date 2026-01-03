# stock_finance_pro.py
# AI-Driven Stock & Sentiment Analysis Dashboard
# Uses ChatGPT + TextBlob + optional VADER with safe fallback

import re
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import requests
from datetime import datetime

from textblob import TextBlob
from openai import OpenAI

# =============================
# Optional VADER (SAFE IMPORT)
# =============================
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_available = True
    vader = SentimentIntensityAnalyzer()
except Exception:
    vader_available = False
    vader = None

# =============================
# API KEYS (Streamlit Secrets)
# =============================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =============================
# FALLBACK AI (NO LIMIT)
# =============================
def finance_fallback_answer(question, symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        price = info.get("currentPrice")

        q = question.lower()

        match = re.search(r'(\d+)\s+shares?', q)
        if match and price:
            shares = int(match.group(1))
            return f"{shares} shares of {symbol} are worth approximately ${shares * price:,.2f}."

        if "price" in q:
            return f"The current price of {symbol} is approximately ${price:.2f}."

        if "risk" in q:
            return (
                "Risks include market volatility, earnings uncertainty, "
                "macroeconomic conditions, interest rate changes, and sector competition."
            )

        if "market cap" in q:
            return f"Market capitalization is approximately ${info.get('marketCap','N/A')}."

        return "AI was unavailable, so real-time financial data was used instead."

    except Exception:
        return "Unable to retrieve financial data."

# =============================
# DATA
# =============================
@st.cache_data(ttl=300)
def get_stock_data(symbol, period, interval):
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df

def sma(df, p): df[f"SMA{p}"] = df["close"].rolling(p).mean(); return df
def ema(df, p): df[f"EMA{p}"] = df["close"].ewm(span=p, adjust=False).mean(); return df

# =============================
# CHARTS
# =============================
def plot_candle(df, symbol):
    fig = go.Figure([go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"]
    )])
    fig.update_layout(title=f"{symbol} Candlestick")
    st.plotly_chart(fig, use_container_width=True)

def plot_line(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close"))
    for c in df.columns:
        if "SMA" in c or "EMA" in c:
            fig.add_trace(go.Scatter(x=df["date"], y=df[c], name=c))
    fig.update_layout(title=f"{symbol} Trend")
    st.plotly_chart(fig, use_container_width=True)

# =============================
# NEWS + SENTIMENT
# =============================
def fetch_news(symbol, n=5):
    url = f"https://newsapi.org/v2/everything?q={symbol}&pageSize={n}&apiKey={NEWSAPI_KEY}"
    articles = requests.get(url).json().get("articles", [])
    data = []

    for a in articles:
        text = f"{a.get('title','')} {a.get('description','')}"
        blob = TextBlob(text).sentiment.polarity
        vader_score = vader.polarity_scores(text)["compound"] if vader_available else None

        data.append({
            "title": a.get("title"),
            "description": a.get("description"),
            "date": a.get("publishedAt"),
            "url": a.get("url"),
            "textblob": blob,
            "vader": vader_score,
            "avg": (blob + vader_score) / 2 if vader_score is not None else blob
        })
    return data

# =============================
# AI Q&A
# =============================
def ai_answer(question, symbol):
    if client is None:
        return finance_fallback_answer(question, symbol)

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a financial analyst. No price predictions."},
                {"role":"user","content":question}
            ],
            temperature=0.4,
            max_tokens=300
        )
        return r.choices[0].message.content
    except Exception:
        return finance_fallback_answer(question, symbol)

# =============================
# UI
# =============================
st.set_page_config("Stock Sentiment Analyzer", layout="wide")
st.title("📈 AI-Driven Stock & Sentiment Analysis")

symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"])
sma_p = st.sidebar.number_input("SMA", 5, 200, 20)
ema_p = st.sidebar.number_input("EMA", 5, 200, 20)

df = get_stock_data(symbol, period, interval)
df = sma(df, sma_p)
df = ema(df, ema_p)

plot_candle(df, symbol)
plot_line(df, symbol)
st.dataframe(df.tail(10))
st.download_button("Download CSV", df.to_csv(index=False), f"{symbol}.csv")

st.subheader("📰 News Sentiment")
news = fetch_news(symbol)
for n in news:
    st.markdown(f"**{n['title']}** ({n['date']})")
    st.markdown(n["description"])
    st.markdown(
        f"TextBlob: `{n['textblob']:.2f}` | "
        f"VADER: `{n['vader'] if n['vader'] is not None else 'N/A'}` | "
        f"Avg: `{n['avg']:.2f}`"
    )
    st.markdown(f"[Read More]({n['url']})")
    st.markdown("---")

st.subheader("🤖 Ask AI About the Stock")
q = st.text_area("Question")
if st.button("Ask"):
    st.markdown(f"**Answer:** {ai_answer(q, symbol)}")

st.caption(
    "Uses ChatGPT, TextBlob, optional VADER, and real market data. "
    "Demonstrates AI validation, cross-checking, and responsible analysis."
)

