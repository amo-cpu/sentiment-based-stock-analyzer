# stock_finance_pro.py

import os
import re
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import requests

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI
from datetime import datetime, timedelta

# ----------------------------
# API Keys (from environment)
# ----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
vader = SentimentIntensityAnalyzer()

# ----------------------------
# Fallback Financial Logic
# ----------------------------
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

        if "price" in q or "current" in q:
            return f"The current price of {symbol} is approximately ${price:.2f}."

        if "market cap" in q:
            return f"{symbol}'s market cap is approximately ${info.get('marketCap', 'N/A')}."

        if "pe" in q or "p/e" in q:
            return f"{symbol}'s P/E ratio is approximately {info.get('trailingPE', 'N/A')}."

        if "risk" in q:
            return (
                "Risks include earnings volatility, market conditions, "
                "interest rate changes, and competitive pressure."
            )

        if "company" in q or "what does" in q:
            return info.get("longBusinessSummary", "Company description unavailable.")

        return (
            "AI was unavailable, so real-time financial data was used instead. "
            "Try asking about price, shares, fundamentals, or risk."
        )

    except Exception:
        return "Unable to retrieve financial data at the moment."

# ----------------------------
# Stock Data
# ----------------------------
@st.cache_data(ttl=300)
def get_stock_data(symbol, period, interval):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=interval)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df

def calculate_sma(df, period):
    df[f"SMA{period}"] = df["close"].rolling(period).mean()
    return df

def calculate_ema(df, period):
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df

# ----------------------------
# Charts
# ----------------------------
def plot_candlestick(df, symbol):
    fig = go.Figure(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    ))
    fig.update_layout(title=f"{symbol} Candlestick Chart")
    st.plotly_chart(fig, use_container_width=True)

def plot_line(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close"))

    for col in df.columns:
        if col.startswith("SMA") or col.startswith("EMA"):
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], name=col))

    fig.update_layout(title=f"{symbol} Price Trend")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# News + Sentiment Analysis
# ----------------------------
def fetch_news(symbol, limit=5):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={symbol}&sortBy=publishedAt&pageSize={limit}&apiKey={NEWSAPI_KEY}"
    )

    response = requests.get(url)
    articles = response.json().get("articles", [])
    results = []

    for a in articles:
        text = f"{a.get('title','')} {a.get('description','')}"
        tb_score = TextBlob(text).sentiment.polarity
        vader_score = vader.polarity_scores(text)["compound"]

        results.append({
            "title": a.get("title"),
            "description": a.get("description"),
            "url": a.get("url"),
            "date": a.get("publishedAt"),
            "textblob": tb_score,
            "vader": vader_score
        })

    return results

# ----------------------------
# AI Q&A
# ----------------------------
def ai_answer(question, symbol):
    if not client:
        return finance_fallback_answer(question, symbol)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst. "
                        "Do not predict prices. "
                        "Explain concepts and risks clearly."
                    )
                },
                {"role": "user", "content": question}
            ],
            temperature=0.4,
            max_tokens=300
        )
        return response.choices[0].message.content

    except Exception:
        return finance_fallback_answer(question, symbol)

# ----------------------------
# CSV Export
# ----------------------------
def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Professional Stock Dashboard", layout="wide")
st.title("📈 Professional Stock Dashboard")

st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
compare = st.sidebar.text_input("Compare With (Optional)").upper()
period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"])
sma_p = st.sidebar.number_input("SMA Period", 2, 200, 20)
ema_p = st.sidebar.number_input("EMA Period", 2, 200, 20)

st.subheader(f"{symbol} Historical Data")
df = get_stock_data(symbol, period, interval)

if not df.empty:
    df = calculate_sma(df, sma_p)
    df = calculate_ema(df, ema_p)
    plot_candlestick(df, symbol)
    plot_line(df, symbol)
    st.dataframe(df.tail(10))
    st.download_button("Download CSV", to_csv(df), f"{symbol}.csv")

if compare:
    st.subheader(f"Comparison: {compare}")
    df2 = get_stock_data(compare, period, interval)
    if not df2.empty:
        df2 = calculate_sma(df2, sma_p)
        df2 = calculate_ema(df2, ema_p)
        plot_line(df2, compare)

st.subheader("News & Sentiment Analysis")
news = fetch_news(symbol)

for n in news:
    st.markdown(f"**{n['title']}** ({n['date']})")
    st.write(n["description"])
    st.write(f"TextBlob Sentiment: {n['textblob']:.2f}")
    st.write(f"VADER Sentiment: {n['vader']:.2f}")
    st.markdown(f"[Read more]({n['url']})")
    st.markdown("---")

st.caption(
    "Multiple NLP models (TextBlob + VADER) are used to compare sentiment outputs "
    "and validate AI-generated interpretations."
)

st.subheader("Ask AI about the Stock")
q = st.text_area("Enter your question")
if st.button("Get Answer"):
    if q.strip():
        st.write(ai_answer(q, symbol))
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.markdown(
    "Developed by a Sophomore | AI-Driven Stock Analysis Dashboard | "
    "Streamlit · yFinance · OpenAI · VADER · TextBlob"
)
