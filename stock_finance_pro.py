# stock_finance_pro.py
# Full professional stock dashboard for Streamlit
# Features: Real-time stock data, SMA/EMA, candlestick & line charts, AI Q&A, news + sentiment, multi-stock comparison, CSV export

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import requests
from textblob import TextBlob
from openai import OpenAI
from datetime import datetime, timedelta
import os
# Get keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
# ----------------------------
# Helper Functions
# ----------------------------
def finance_fallback_answer(question, symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        q = question.lower()
        if "price" in q or "current" in q:
            return f"The current price of {symbol} is ${info.get('currentPrice', 'N/A')}."
        if "market cap" in q:
            return f"{symbol} has a market capitalization of ${info.get('marketCap', 'N/A')}."
        if "pe" in q or "p/e" in q:
            return f"{symbol} has a P/E ratio of {info.get('trailingPE', 'N/A')}."
        if "what does" in q or "company" in q:
            return info.get("longBusinessSummary", "No company description available.")
        if "risk" in q:
            return (
                "Key risks include market volatility, earnings uncertainty, "
                "macroeconomic conditions, and sector-specific risks."
            )
        if "outlook" in q or "future" in q:
            return (
                "Outlook depends on earnings growth, macroeconomic conditions, "
                "interest rates, and overall market sentiment."
            )
        return (
            "This question requires deeper qualitative analysis. "
            "The AI assistant could not be reached, so real-time data was used instead."
        )
    except Exception:
        return "Unable to retrieve fallback financial data."
@st.cache_data(ttl=300)
def get_stock_data(symbol, period="6mo", interval="1d"):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        if "close" not in df.columns:
            raise KeyError("No recognized close price column found in the DataFrame.")
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def calculate_sma(df, period=20):
    df[f"SMA{period}"] = df["close"].rolling(window=period).mean()
    return df

def calculate_ema(df, period=20):
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df

def plot_candlestick(df, symbol):
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'],
                                         name=symbol)])
    fig.update_layout(title=f"{symbol} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def plot_line(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Close'))
    if f"SMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['SMA20'], mode='lines', name='SMA20'))
    if f"EMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['EMA20'], mode='lines', name='EMA20'))
    fig.update_layout(title=f"{symbol} Line Chart", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def fetch_news(symbol, page_size=5):
    url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&pageSize={page_size}&apiKey={NEWSAPI_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        articles = data.get("articles", [])
        news_list = []
        for art in articles:
            title = art.get("title")
            desc = art.get("description")
            url = art.get("url")
            date = art.get("publishedAt")
            sentiment = TextBlob((title or "") + " " + (desc or "")).sentiment.polarity
            news_list.append({"title": title, "description": desc, "url": url, "date": date, "sentiment": sentiment})
        return news_list
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def ai_answer(question, symbol):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional financial analyst. "
                        "Do not predict exact prices. "
                        "Provide educational and analytical answers."
                    )
                },
                {"role": "user", "content": question}
            ],
            max_tokens=300,
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception:
        return finance_fallback_answer(question, symbol)
def download_csv(df):
    return df.to_csv(index=False).encode('utf-8')
# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(page_title="Professional Stock Dashboard", layout="wide")
st.title("📈 Professional Stock Dashboard")

# Sidebar Inputs
st.sidebar.header("Configuration")
stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
comparison_symbol = st.sidebar.text_input("Compare With (Optional)").upper()
period = st.sidebar.selectbox("Data Period", ["1mo","3mo","6mo","1y","2y","5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)
sma_period = st.sidebar.number_input("SMA Period", min_value=2, max_value=200, value=20)
ema_period = st.sidebar.number_input("EMA Period", min_value=2, max_value=200, value=20)

# Main Stock Data
st.subheader(f"{stock_symbol} Stock Data")
df = get_stock_data(stock_symbol, period, interval)
if not df.empty:
    df = calculate_sma(df, sma_period)
    df = calculate_ema(df, ema_period)
    plot_candlestick(df, stock_symbol)
    plot_line(df, stock_symbol)
    st.dataframe(df.tail(10))
    csv_data = download_csv(df)
    st.download_button("Download CSV", csv_data, file_name=f"{stock_symbol}_data.csv", mime="text/csv")

# Comparison Stock
if comparison_symbol:
    st.subheader(f"Comparison: {comparison_symbol}")
    df2 = get_stock_data(comparison_symbol, period, interval)
    if not df2.empty:
        df2 = calculate_sma(df2, sma_period)
        df2 = calculate_ema(df2, ema_period)
        plot_line(df2, comparison_symbol)
        st.dataframe(df2.tail(10))
        csv_data2 = download_csv(df2)
        st.download_button("Download CSV", csv_data2, file_name=f"{comparison_symbol}_data.csv", mime="text/csv")

# News & Sentiment
st.subheader("Latest News & Sentiment")
news = fetch_news(stock_symbol)
if news:
    for n in news:
        st.markdown(f"**{n['title']}** ({n['date']})")
        st.markdown(f"{n['description']}")
        st.markdown(f"Sentiment Score: {n['sentiment']:.2f}")
        st.markdown(f"[Read More]({n['url']})")
        st.markdown("---")

# AI Q&A
st.subheader("Ask AI about the Stock")
question = st.text_area("Enter your question here:")
if st.button("Get Answer"):
    if question.strip() != "":
answer = ai_answer(question, stock_symbol)
st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.markdown("Developed by a Sophomore | Professional Stock Dashboard | Powered by Streamlit, yfinance, OpenAI, NewsAPI, Plotly")



