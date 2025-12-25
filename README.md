Professional Stock Finance Dashboard

This project is an interactive stock market analysis dashboard built with Python and Streamlit. It integrates real-time and historical market data, technical indicators, news sentiment analysis, and an AI-powered question-answering system with a deterministic fallback to ensure reliability.

The dashboard is designed for educational and analytical use, demonstrating how artificial intelligence and data science can be applied responsibly in finance.

Features
Stock Market Analysis

Real-time and historical stock data using Yahoo Finance

Configurable time periods and intervals

Candlestick and line charts

Simple Moving Average (SMA)

Exponential Moving Average (EMA)

News & Sentiment Analysis

Live financial news using NewsAPI

Sentiment scoring using TextBlob

Article-level sentiment visualization and summaries

AI-Powered Stock Q&A

Natural language questions about stocks

Answers questions such as:

Share value calculations

Current price

Company overview

Market risk

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

numpy

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
1. Clone the Repository
git clone https://github.com/yourusername/stock-finance-dashboard.git
cd stock-finance-dashboard

2. Install Dependencies
pip install -r requirements.txt

3. Set Environment Variables

Set the following environment variables:

OPENAI_API_KEY=your_openai_api_key
NEWSAPI_KEY=your_newsapi_key


For Streamlit Cloud, add these under Secrets.

Running the Application
streamlit run stock_finance_pro.py

AI Safety and Reliability

The AI assistant is configured to:

Avoid price prediction

Provide educational and analytical responses only

Fall back to deterministic finance calculations when AI is unavailable

This ensures that the application remains reliable and transparent.

Disclaimer

This project is for educational purposes only and does not constitute financial advice.

Author

Developed by a high school student with interests in artificial intelligence, finance, and applied data science.pro
