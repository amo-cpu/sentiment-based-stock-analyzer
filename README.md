Professional Stock Finance Dashboard

Live Demo:
https://sentiment-based-stock-analyzer-p6kc8qzqfukysyffrfwv4u.streamlit.app/

This project is an interactive stock market analysis dashboard built with Python and Streamlit. It integrates real-time and historical market data, technical indicators, news sentiment analysis, and an AI-assisted question-answering system with deterministic fallback logic to ensure reliability.

The dashboard is designed for educational and analytical use, demonstrating responsible applications of artificial intelligence in finance.

Features
Stock Market Analysis

Real-time and historical stock data via Yahoo Finance

Configurable time periods and data intervals

Candlestick and line chart visualizations

Simple Moving Average (SMA)

Exponential Moving Average (EMA)

News and Sentiment Analysis

Live financial news retrieval using NewsAPI

Sentiment polarity scoring using TextBlob

Article-level sentiment display

AI-Assisted Stock Q&A

Natural-language questions about stocks

Supports:

Share value calculations

Current price queries

Company overviews

Market risk explanations

Basic fundamentals

Uses OpenAI for analytical responses

Automatically falls back to real-time financial calculations if AI is unavailable

Additional Functionality

Multi-stock comparison

CSV export for offline analysis

Cached data retrieval for performance optimization

Tech Stack

Python

Streamlit

pandas

yfinance

Plotly

NewsAPI

TextBlob

OpenAI API

Project Structure
stock_finance_pro.py
requirements.txt
README.md

Installation and Setup
Clone the Repository
git clone https://github.com/yourusername/stock-finance-dashboard.git
cd stock-finance-dashboard

Install Dependencies
pip install -r requirements.txt

Set Environment Variables

The following environment variables are required:

OPENAI_API_KEY=your_openai_api_key
NEWSAPI_KEY=your_newsapi_key


For Streamlit Cloud deployment, add these values under Secrets.

Running the Application
streamlit run stock_finance_pro.py

AI Safety and Reliability

The AI assistant is explicitly configured to:

Avoid price prediction

Provide educational and analytical responses only

Fall back to deterministic financial calculations when AI services fail

This ensures transparency, reliability, and responsible AI usage.

Disclaimer

This project is for educational purposes only and does not constitute financial advice.

Author

Developed by a high school student with interests in artificial intelligence, finance, and applied data science.
