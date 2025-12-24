import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
from textblob import TextBlob
from datetime import datetime
from transformers import pipeline

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Stock Finance Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- THEME ----------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.stButton>button { background-color:#1f77b4; color:white; }
</style>
""", unsafe_allow_html=True)

# ---------------- AI ----------------
@st.cache_resource
def load_hf():
    return pipeline("text-generation", model="distilgpt2")

hf_ai = load_hf()

def ask_ai(question):
    try:
        import openai
        openai.api_key = "YOUR_OPENAI_KEY"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":question}]
        )
        return response.choices[0].message.content
    except:
        return hf_ai(question, max_length=150)[0]["generated_text"]

# ---------------- HELPERS ----------------
def fetch_stock(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df.reset_index(inplace=True)
    return df

def add_indicators(df):
    df["SMA25"] = df["Close"].rolling(25).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean()))
    df["BB_UP"] = df["Close"].rolling(20).mean() + 2 * df["Close"].rolling(20).std()
    df["BB_DOWN"] = df["Close"].rolling(20).mean() - 2 * df["Close"].rolling(20).std()
    return df

def sentiment(text):
    return TextBlob(text).sentiment.polarity

# ---------------- SIDEBAR ----------------
st.sidebar.title("📈 Stock Finance Pro")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Portfolio", "Charts", "News & Sentiment", "AI Assistant", "Currency Converter"]
)

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.title("Market Overview")

    ticker = st.text_input("Enter ticker", "AAPL")
    df = fetch_stock(ticker)

    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"${df['Close'].iloc[-1]:.2f}")
    col2.metric("High", f"${df['High'].max():.2f}")
    col3.metric("Low", f"${df['Low'].min():.2f}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- PORTFOLIO ----------------
elif page == "Portfolio":
    st.title("Portfolio")

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    ticker = st.text_input("Add stock")
    if st.button("Add"):
        st.session_state.portfolio.append(ticker.upper())

    st.write("Your Portfolio:", st.session_state.portfolio)

# ---------------- CHARTS ----------------
elif page == "Charts":
    st.title("Technical Charts")

    ticker = st.text_input("Ticker", "AAPL")
    df = add_indicators(fetch_stock(ticker))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA25"], name="SMA25"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA50"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_UP"], name="BB Upper"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_DOWN"], name="BB Lower"))

    st.plotly_chart(fig, use_container_width=True)

# ---------------- NEWS ----------------
elif page == "News & Sentiment":
    st.title("News & Sentiment")

    query = st.text_input("Company", "Apple")
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey=YOUR_NEWS_API_KEY"
    data = requests.get(url).json()

    for a in data.get("articles", [])[:5]:
        score = sentiment(a["title"])
        st.write(f"**{a['title']}**")
        st.write("Sentiment:", "🟢 Positive" if score > 0 else "🔴 Negative")
        st.write(a["url"])
        st.divider()

# ---------------- AI ----------------
elif page == "AI Assistant":
    st.title("AI Finance Assistant")

    q = st.text_area("Ask anything about finance")
    if st.button("Ask"):
        st.write(ask_ai(q))

# ---------------- CURRENCY ----------------
elif page == "Currency Converter":
    st.title("Currency Converter")

    amt = st.number_input("Amount", 1.0)
    frm = st.text_input("From", "USD")
    to = st.text_input("To", "EUR")

    r = requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()
    result = amt * r["rates"][to.upper()] / r["rates"][frm.upper()]
    st.metric("Converted", f"{result:.2f} {to.upper()}")

