# stock_finance_pro.py
# Professional AI-Powered Stock & Sentiment Analysis Dashboard
# Uses ChatGPT + VADER + TextBlob + Real-Time & Historical Market Data

import re
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import requests
from datetime import datetime, timedelta

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI

# ============================
# API KEYS (from Streamlit Secrets)
# ============================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
vader = SentimentIntensityAnalyzer()

# ============================
# Utility & Fallback Logic
# ============================
def finance_fallback_answer(question, symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        price = info.get("currentPrice")

        q = question.lower()

        # Shares calculation
        match = re.search(r'(\d+)\s+shares?', q)
        if match and price:
            shares = int(match.group(1))
            return f"{shares} shares of {symbol} are worth approximately ${shares * price:,.2f}."

        if "price" in q:
            return f"The current price of {symbol} is approximately ${price:.2f}."

        if "risk" in q:
            return (
                "Investment risks include market volatility, earnings uncertainty, "
                "interest rate changes, macroeconomic conditions, and industry competition."
            )

        if "market cap" in q:
            return f"{symbol}'s market cap is approximately ${info.get('marketCap', 'N/A')}."

        return (
            "AI access was unavailable, so real-time financial data was used instead. "
            "Try asking about price, shares, risk, or company fundamentals."
        )

    except Exception:
        return "Unable to retrieve financial data at this time."

# ============================
# Data Fetching
# ============================
@st.cache_data(ttl=300)
def get_stock_data(symbol, period, interval):
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df

def calculate_sma(df, period):
    df[f"SMA{period}"] = df["close"].rolling(period).mean()
    return df

def calculate_ema(df, period):
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df

# ============================
# Charting
# ============================
def plot_candlestick(df, symbol):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]
        )
    ])
    fig.update_layout(title=f"{symbol} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def plot_line(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close"))
    for col in df.columns:
        if "SMA" in col or "EMA" in col:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], name=col))
    fig.update_layout(title=f"{symbol} Trend Analysis", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# ============================
# News + Sentiment (VADER + TextBlob)
# ============================
def fetch_news(symbol, page_size=5):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={symbol}&sortBy=publishedAt&pageSize={page_size}&apiKey={NEWSAPI_KEY}"
    )
    articles = requests.get(url).json().get("articles", [])

    news_data = []
    for a in articles:
        text = f"{a.get('title','')} {a.get('description','')}"
        vader_score = vader.polarity_scores(text)["compound"]
        blob_score = TextBlob(text).sentiment.polarity

        news_data.append({
            "title": a.get("title"),
            "description": a.get("description"),
            "url": a.get("url"),
            "date": a.get("publishedAt"),
            "vader": vader_score,
            "textblob": blob_score,
            "avg_sentiment": (vader_score + blob_score) / 2
        })

    return news_data

# ============================
# AI Q&A (ChatGPT + fallback)
# ============================
def ai_answer(question, symbol):
    if client is None:
        return finance_fallback_answer(question, symbol)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional financial analyst. "
                        "Use historical context and sentiment cautiously. "
                        "Do not provide price predictions."
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

# ============================
# Streamlit UI
# ============================
st.set_page_config("Professional Stock Dashboard", layout="wide")
st.title("📊 AI-Driven Stock & Sentiment Analysis Platform")

# Sidebar
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
compare = st.sidebar.text_input("Compare With (Optional)").upper()
period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"])
sma_p = st.sidebar.number_input("SMA Period", 5, 200, 20)
ema_p = st.sidebar.number_input("EMA Period", 5, 200, 20)

# Main Stock
df = get_stock_data(symbol, period, interval)
df = calculate_sma(df, sma_p)
df = calculate_ema(df, ema_p)

plot_candlestick(df, symbol)
plot_line(df, symbol)
st.dataframe(df.tail(10))
st.download_button("Download CSV", df.to_csv(index=False), f"{symbol}.csv")

# Comparison
if compare:
    df2 = get_stock_data(compare, period, interval)
    plot_line(df2, compare)

# News & Sentiment
st.subheader("📰 News Sentiment Analysis (VADER + TextBlob)")
news = fetch_news(symbol)

for n in news:
    st.markdown(f"**{n['title']}** ({n['date']})")
    st.markdown(n["description"])
    st.markdown(
        f"VADER: `{n['vader']:.2f}` | "
        f"TextBlob: `{n['textblob']:.2f}` | "
        f"Average: `{n['avg_sentiment']:.2f}`"
    )
    st.markdown(f"[Read More]({n['url']})")
    st.markdown("---")

# AI Q&A
st.subheader("🤖 Ask AI About the Stock")
question = st.text_area("Enter your question:")
if st.button("Get Answer"):
    st.markdown(f"**Answer:** {ai_answer(question, symbol)}")

# Footer
st.markdown("---")
st.caption(
    "Built with Streamlit • yFinance • NewsAPI • VADER • TextBlob • OpenAI | "
    "Demonstrates AI-assisted analysis with validation and human oversight"
)
