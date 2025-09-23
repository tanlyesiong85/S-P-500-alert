import os, math, time, urllib.parse, requests, yaml, feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

# ------------------ ENV ------------------
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY")
WHATSAPP_PHONE = os.getenv("WHATSAPP_PHONE")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

# ------------------ UTIL ------------------
def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def pct(a, b):
    if b == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return (a - b) / b * 100.0

def send_whatsapp(phone, apikey, text):
    base = "https://api.callmebot.com/whatsapp.php"
    params = {"phone": phone, "text": text, "apikey": apikey}
    url = f"{base}?{urllib.parse.urlencode(params)}"
    try:
        r = requests.get(url, timeout=20)
        return r.status_code, (r.text[:500] if r.text else "")
    except Exception as e:
        return -1, f"Exception: {e}"

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dl_weekly(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.rename(columns=str.title)
    return df

# ------------------ TECHNICALS ------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def sma(series, window):
    return series.rolling(window).mean()

def technical_signals_weekly(df, name, cfg, asset_class):
    """Return list of (direction, headline, tech_context) where direction in {'bullish','bearish'}."""
    out = []
    mb = cfg["rules"]["data"]["min_bars"]
    if df is None or df.empty or len(df) < mb:
        return out

    close = df["Close"].dropna()
    last = close.iloc[-1]
    last_dt = df.index[-1].strftime("%Y-%m-%d")

    # RSI extremes
    if cfg["rules"]["rsi"]["enabled"]:
        r = rsi(close, period=cfg["rules"]["rsi"]["period"]).iloc[-1]
        if r <= cfg["rules"]["rsi"]["oversold"]:
            out.append(("bullish", f"{name}: Weekly RSI {r:.1f} (oversold) on {last_dt}", f"RSI<{cfg['rules']['rsi']['oversold']}"))
        elif r >= cfg["rules"]["rsi"]["overbought"]:
            out.append(("bearish", f"{name}: Weekly RSI {r:.1f} (overbought) on {last_dt}", f"RSI>{cfg['rules']['rsi']['overbought']}"))

    # SMA 50/200 cross
    if cfg["rules"]["sma_cross"]["enabled"]:
        s = sma(close, cfg["rules"]["sma_cross"]["short"])
        l = sma(close, cfg["rules"]["sma_cross"]["long"])
        if not np.isnan(s.iloc[-1]) and not np.isnan(l.iloc[-1]) and not np.isnan(s.iloc[-2]) and not np.isnan(l.iloc[-2]):
            prev = s.iloc[-2] - l.iloc[-2]
            curr = s.iloc[-1] - l.iloc[-1]
            if prev < 0 <= curr:
                out.append(("bullish", f"{name}: Golden Cross (SMA{cfg['rules']['sma_cross']['short']}↑SMA{cfg['rules']['sma_cross']['long']}) on {last_dt}", "Golden Cross"))
            elif prev > 0 >= curr:
                out.append(("bearish", f"{name}: Death Cross (SMA{cfg['rules']['sma_cross']['short']}↓SMA{cfg['rules']['sma_cross']['long']}) on {last_dt}", "Death Cross"))

    # Price vs 200-week SMA
    if cfg["rules"]["price_vs_sma200"]["enabled"]:
        s200 = sma(close, 200)
        if not np.isnan(s200.iloc[-1]):
            diff = pct(last, s200.iloc[-1])
            if abs(diff) <= cfg["rules"]["price_vs_sma200"]["near_pct"]:
                out.append(("bullish", f"{name}: Price within {diff:.1f}% of 200-week SMA on {last_dt}", "Near 200-wk SMA"))
            elif diff <= -cfg["rules"]["price_vs_sma200"]["break_pct"]:
                out.append(("bullish", f"{name}: Price {diff:.1f}% below 200-week SMA on {last_dt}", "Deep below 200-wk SMA"))
            elif diff >= cfg["rules"]["price_vs_sma200"]()
