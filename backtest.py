import os, csv, yaml, math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

# -------- helpers --------
def sma(s, w): return s.rolling(w).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def pct(a, b):
    if b == 0 or np.isnan(a) or np.isnan(b): return np.nan
    return (a - b) / b * 100.0

def load_cfg(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def dl_weekly(symbol, years=10):
    # yfinance weekly for ~10y
    period = f"{years}y"
    df = yf.download(symbol, period=period, interval="1wk", auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.rename(columns=str.title)
    return df

# -------- backtest (TECHNICAL ONLY) --------
def generate_signals_weekly(df, cfg):
    """
    Returns list of dicts:
    {date, signal, direction, rationale, close, rsi, sma50, sma200, dist_to_sma200_pct}
    """
    out = []
    mb = cfg["rules"]["data"]["min_bars"]
    if df is
