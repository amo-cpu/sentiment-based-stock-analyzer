# AI-Driven Stock & Sentiment Analysis Dashboard
# Uses: ChatGPT (OpenAI), VADER, TextBlob, yfinance, Streamlit
# Designed to demonstrate AI validation, cross-model sentiment analysis, and fallback logic

import re
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI

# =========================
# API KEYS (FROM SECRETS)
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
vader = SentimentIntensityAnalyzer()

# =========================
# DATA FUNCTIONS
# =========================
@st.cache_data(ttl=300)
def get_stock_data(symbol, period="5y", interval="1d"):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=interval)
    df.reset_index(inplace=True)
    df.columns = [c.lower() for c in df.columns]
    return df

def add_indicators(df, sma=20, ema=20):
    df["SMA"] = df["close"].rolling(sma).mean()
    df["EMA"] = df["close"].ewm(span=ema, adjust=False).mean()
    return df

# =========================
# PLOTTING
# =========================
def plot_stock(df, symbol):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(x=df["date"], y=df["SMA"], name="SMA"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["EMA"], name="EMA"))

    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# NEWS + SENTIMENT
# =========================
def fetch_news(symbol, limit=5):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={symbol}&pageSize={limit}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    res = requests.get(url).json()
    return res.get("articles", [])

def analyze_sentiment(text):
    tb = TextBlob(text).sentiment.polarity
    vd = vader.polarity_scores(text)["compound"]
    avg = (tb + vd) / 2
    return tb, vd, avg

# =========================
# AI FALLBACK LOGIC
# =========================
def finance_fallback_answer(question, symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    hist = stock.history(period="1d")

    price = info.get("currentPrice")
    if price is None and not hist.empty:
        price = hist["Close"].iloc[-1]

    q = question.lower()

    match = re.search(r'(\d+)\s+shares?', q)
    if match and price:
        shares = int(match.group(1))
        value = shares * price
        return f"{shares} shares of {symbol} are worth approximately ${value:,.2f}."

    if "price" in q and price:
        return f"The current price of {symbol} is approximately ${price:.2f}."

    if "risk" in q:
        return (
            "Key risks include market volatility, earnings uncertainty, "
            "macroeconomic conditions, interest rates, and industry competition."
        )

    if "market cap" in q:
        return f"{symbol}'s market cap is approximately ${info.get('marketCap','N/A')}."

    return (
        "AI services were unavailable, so this response was generated using "
        "real financial data and rule-based analysis."
    )

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
                        "You are a financial analyst. "
                        "Do not predict prices. Provide analytical, educational answers."
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

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="AI Stock & Sentiment Analyzer", layout="wide")
st.title("📈 AI-Driven Stock & Sentiment Analysis")

symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
period = st.sidebar.selectbox("Period", ["1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
sma = st.sidebar.slider("SMA", 5, 100, 20)
ema = st.sidebar.slider("EMA", 5, 100, 20)

# =========================
# STOCK DATA
# =========================
df = get_stock_data(symbol, period, interval)
df = add_indicators(df, sma, ema)

plot_stock(df, symbol)
st.dataframe(df.tail(10))

# =========================
# NEWS SENTIMENT
# =========================
st.subheader("📰 News Sentiment")
articles = fetch_news(symbol)

for a in articles:
    text = f"{a.get('title','')} {a.get('description','')}"
    tb, vd, avg = analyze_sentiment(text)

    st.markdown(f"**{a['title']}**")
    st.caption(a["publishedAt"])
    st.write(a.get("description", ""))

    st.write(f"TextBlob: {tb:.2f} | VADER: {vd:.2f} | Avg: {avg:.2f}")
    st.markdown(f"[Read More]({a['url']})")
    st.markdown("---")

# =========================
# AI Q&A
# =========================
st.subheader("🤖 Ask AI About the Stock")
question = st.text_input("Question")

if st.button("Ask"):
    if question.strip():
        answer = ai_answer(question, symbol)
        st.success(answer)
    else:
        st.warning("Please enter a question.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "Built by a Sophomore | Uses ChatGPT, VADER, TextBlob | "
    "Demonstrates AI validation & fallback engineering"
)


