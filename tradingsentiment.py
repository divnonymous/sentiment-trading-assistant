# app_tiingo_final.py
import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide", page_title="Tiingo+Finnhub Advisor")
st.title("Advisor — Tiingo (history) + Finnhub (live & news) + VADER")

# --- Keys (prefer env vars)
TIINGO_KEY = st.sidebar.text_input("Tiingo API key (or set TIINGO_KEY env var)", value=os.getenv("ab89107fabdd3f0939e7c13ee85d5b9e6ba445ce",""), type="password")
FINNHUB_KEY = st.sidebar.text_input("Finnhub API key (or set FINNHUB_KEY env var)", value=os.getenv("d4bq5phr01qoua311bmgd4bq5phr01qoua311bn0",""), type="password")
TICKER = st.sidebar.text_input("Ticker (Tiingo symbol)", "AAPL").strip().upper()
LOOKBACK = st.sidebar.slider("History days", 30, 720, 180)
NEWS_DAYS = st.sidebar.slider("News lookback days", 1, 14, 7)
FETCH_NEWS = st.sidebar.checkbox("Fetch company news (Finnhub)", True)
REFRESH = st.sidebar.button("Refresh")

# --- small rate guard
if 'last' not in st.session_state:
    st.session_state['last'] = 0
if REFRESH:
    st.session_state['last'] = time.time()

# --- helpers
def tiingo_history(symbol, token, limit=720, max_retries=2):
    """Return (df, msg). df has ['date','close']."""
    if not token:
        return None, "No Tiingo key"
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {"token": token}
    attempt = 0
    while attempt <= max_retries:
        try:
            r = requests.get(url, params=params, timeout=12)
            if r.status_code == 200:
                data = r.json()
                if not isinstance(data, list) or len(data) == 0:
                    return None, f"Tiingo returned empty or unexpected: {data}"
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date']).dt.floor('D')
                df = df[['date','close']].sort_values('date').tail(limit).reset_index(drop=True)
                return df, "OK (Tiingo)"
            elif r.status_code in (429, 503):
                # rate-limited or service unavailable: backoff
                backoff = min(60, 2 ** attempt)
                time.sleep(backoff)
                attempt += 1
                continue
            else:
                return None, f"Tiingo HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:
            attempt += 1
            time.sleep(min(5, 2**attempt))
    return None, f"Tiingo failed after {max_retries+1} attempts"

def finnhub_quote(sym, key):
    if not key:
        return None, "No Finnhub key"
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={sym}&token={key}"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None, f"Finnhub HTTP {r.status_code}: {r.text[:300]}"
        j = r.json()
        return j.get('c'), "OK"
    except Exception as e:
        return None, f"Finnhub exception: {e}"

def finnhub_news(sym, key, days=7):
    if not key:
        return None, "No Finnhub key"
    try:
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=days)
        url = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={from_date}&to={to_date}&token={key}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None, f"Finnhub HTTP {r.status_code}: {r.text[:300]}"
        j = r.json()
        if isinstance(j, dict) and j.get('error'):
            return None, f"Finnhub error: {j}"
        headlines = [it.get('headline') or it.get('summary') for it in j if it.get('headline') or it.get('summary')]
        return headlines, "OK"
    except Exception as e:
        return None, f"Finnhub news exception: {e}"

# --- data fetch (cached manually via session_state to avoid re-calls on rerun)
cache_key = f"{TICKER}_{LOOKBACK}_{TIINGO_KEY}"
if 'hist_cache' not in st.session_state:
    st.session_state['hist_cache'] = {}
hist_df = st.session_state['hist_cache'].get(cache_key)

hist_msg = ""
if hist_df is None:
    hist_df, hist_msg = tiingo_history(TICKER, TIINGO_KEY, limit=LOOKBACK)
    # store short-lived cache
    st.session_state['hist_cache'][cache_key] = hist_df

# live quote
live_price, live_msg = finnhub_quote(TICKER, FINNHUB_KEY) if FINNHUB_KEY else (None, "No Finnhub key")

# news
headlines = []
news_msg = "Not fetched"
if FETCH_NEWS and FINNHUB_KEY:
    headlines, news_msg = finnhub_news(TICKER, FINNHUB_KEY, days=NEWS_DAYS)

# sentiment
analyzer = SentimentIntensityAnalyzer()
avg_sent = 0.0
scores = []
if headlines:
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    avg_sent = float(np.mean(scores)) if scores else 0.0

# indicators + rule
if hist_df is not None and not hist_df.empty:
    hist_df = hist_df.sort_values('date').drop_duplicates('date').tail(LOOKBACK).reset_index(drop=True)
    hist_df['sma5'] = hist_df['close'].rolling(5, min_periods=1).mean()
    hist_df['sma20'] = hist_df['close'].rolling(20, min_periods=1).mean()
    hist_df['ret5'] = hist_df['close'].pct_change(5)
    latest = hist_df.iloc[-1]
    sma5 = float(latest['sma5']); sma20 = float(latest['sma20'])
    ret5 = float(latest['ret5']) if not np.isnan(latest['ret5']) else 0.0
    trend = 1 if sma5 > sma20 else (-1 if sma5 < sma20 else 0)
    momentum = 1 if ret5 > 0.02 else (-1 if ret5 < -0.02 else 0)
else:
    sma5 = sma20 = ret5 = 0.0; trend = momentum = 0

sentiment_factor = 1 if avg_sent > 0.05 else (-1 if avg_sent < -0.05 else 0)
score = trend + momentum + sentiment_factor
if score >= 2: signal = "STRONG BUY"
elif score == 1: signal = "BUY"
elif score == 0: signal = "HOLD"
elif score == -1: signal = "SELL"
else: signal = "STRONG SELL"

# --- UI
left, right = st.columns([3,1])
with left:
    st.subheader(f"{TICKER} chart (Tiingo)")
    if hist_df is not None:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(hist_df['date'], hist_df['close'], label='Close')
        ax.plot(hist_df['date'], hist_df['sma5'], label='SMA5')
        ax.plot(hist_df['date'], hist_df['sma20'], label='SMA20')
        ax.set_title(f"{TICKER} price + SMAs")
        ax.legend(); plt.xticks(rotation=25)
        st.pyplot(fig)
        st.dataframe(hist_df.tail(20))
    else:
        st.info("No historical data (Tiingo key missing or failed). See sidebar messages.")

with right:
    st.subheader("Recommendation")
    if signal in ("STRONG BUY","BUY"): st.success(signal)
    elif signal=="HOLD": st.info(signal)
    else: st.error(signal)
    st.write(f"Score: {score}")
    st.write(f"Avg news sentiment: {avg_sent:.3f} (n={len(scores)})")
    if live_price is not None: st.write(f"Live price (Finnhub): ${live_price:.2f}")

st.sidebar.markdown("## Status")
st.sidebar.write("Tiingo status:", hist_msg)
st.sidebar.write("Finnhub quote status:", live_msg)
st.sidebar.write("Finnhub news status:", news_msg)
st.caption("Use env vars TIINGO_KEY and FINNHUB_KEY for safety; do NOT paste keys publicly.")
