import os, sys, csv, math, urllib.parse, requests, yaml
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
START = os.getenv("BACKTEST_START", "2023-09-01")   # past 24 months default
END   = os.getenv("BACKTEST_END",   datetime.utcnow().strftime("%Y-%m-%d"))

def load_config():
    import yaml
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def fetch_weekly(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, interval="1wk", auto_adjust=True, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    df = df[["Open","High","Low","Close","Volume"]]
    return df

def sma(series, n):
    return series.rolling(n).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def load_events(cfg):
    e = cfg.get("events", {})
    if not e or not e.get("enabled") or e.get("mode") == "off":
        return pd.DataFrame(columns=["date","tag","stance","notes"])
    path = e.get("flags_file", "events_flags.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date","tag","stance","notes"])
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def nearest_event_stance(events_df, dt, window_weeks):
    if events_df.empty:
        return None, None, None
    lo = dt - timedelta(weeks=window_weeks)
    hi = dt + timedelta(weeks=window_weeks)
    window = events_df[(events_df["date"] >= lo) & (events_df["date"] <= hi)]
    if window.empty:
        return None, None, None
    # pick event closest in absolute time
    window = window.copy()
    window["dist"] = (window["date"] - dt).abs()
    row = window.sort_values("dist").iloc[0]
    return row["stance"], row["tag"], row.get("notes", "")

def explain(signal_type, vals, stance, tag):
    if signal_type == "RSI_OVERSOLD":
        why = f"RSI={vals['rsi']:.1f} (<{vals['thresh']}). Weekly oversold zone."
        align = "Aligned with dovish event" if stance == "dovish" else "No dovish event"
        return f"BUY setup: {why}. {align}: {tag or 'n/a'}."
    if signal_type == "RSI_OVERBOUGHT":
        why = f"RSI={vals['rsi']:.1f} (>{vals['thresh']}). Weekly overbought."
        align = "Aligned with hawkish event" if stance == "hawkish" else "No hawkish event"
        return f"RISK/REDUCE: {why}. {align}: {tag or 'n/a'}."
    if signal_type == "MACD_BULL":
        return "BUY setup: Weekly MACD bullish cross."
    if signal_type == "MACD_BEAR":
        return "RISK/REDUCE: Weekly MACD bearish cross."
    if signal_type == "GOLDEN_CROSS":
        return "BUY bias: Weekly SMA50 crossed ABOVE SMA200 (Golden Cross)."
    if signal_type == "DEATH_CROSS":
        return "RISK bias: Weekly SMA50 crossed BELOW SMA200 (Death Cross)."
    if signal_type == "PRICE_ABOVE_200W":
        return f"Strength: Close > {vals['pct']:.0f}% above 200-week SMA."
    if signal_type == "PRICE_BELOW_200W":
        return f"Opportunity/Risk: Close < {vals['pct']:.0f}% below 200-week SMA."
    return "Signal."

def backtest():
    cfg = load_config()
    t = cfg["tickers"][0]  # S&P 500 only
    df = fetch_weekly(t["symbol"], START, END)
    if df.empty:
        print("No data downloaded; check symbol/period.")
        return

    close = df["Close"].copy()
    out = []

    # Indicators
    rsi_on = cfg["rules"]["rsi"]["enabled"]
    macd_on = cfg["rules"]["macd"]["enabled"]
    sma_cross_on = cfg["rules"]["sma_cross"]["enabled"]
    p200_on = cfg["rules"]["price_vs_sma200"]["enabled"]

    # Precompute
    rsi_series = rsi(close, cfg["rules"]["rsi"]["period"]) if rsi_on else None
    macd_line = signal_line = None
    if macd_on:
        macd_line, signal_line, _ = macd(close,
            cfg["rules"]["macd"]["fast"],
            cfg["rules"]["macd"]["slow"],
            cfg["rules"]["macd"]["signal"])

    sma50 = sma(close, cfg["rules"]["sma_cross"]["short"]) if sma_cross_on else None
    sma200 = sma(close, cfg["rules"]["sma_cross"]["long"]) if sma_cross_on or p200_on else None

    events_df = load_events(cfg)
    econf = cfg.get("events", {})
    e_enabled = econf.get("enabled", False) and econf.get("mode") != "off"
    win_weeks = int(econf.get("window_weeks", 2))

    for i in range(2, len(df)):
        dt = df.index[i]  # week end
        msgs = []

        # RSI extremes
        if rsi_on and not np.isnan(rsi_series.iloc[i]):
            r = rsi_series.iloc[i]
            if r <= cfg["rules"]["rsi"]["oversold"]:
                stance, tag, _ = nearest_event_stance(events_df, dt, win_weeks) if e_enabled else (None, None, None)
                if not e_enabled or stance == "dovish":
                    msgs.append(("RSI_OVERSOLD", {"rsi": r, "thresh": cfg["rules"]["rsi"]["oversold"]}, stance, tag))
            elif r >= cfg["rules"]["rsi"]["overbought"]:
                stance, tag, _ = nearest_event_stance(events_df, dt, win_weeks) if e_enabled else (None, None, None)
                if not e_enabled or stance == "hawkish":
                    msgs.append(("RSI_OVERBOUGHT", {"rsi": r, "thresh": cfg["rules"]["rsi"]["overbought"]}, stance, tag))

        # MACD cross
        if macd_on and i >= 1:
            prev = macd_line.iloc[i-1] - signal_line.iloc[i-1]
            curr = macd_line.iloc[i]   - signal_line.iloc[i]
            if prev < 0 <= curr:
                msgs.append(("MACD_BULL", {}, None, None))
            elif prev > 0 >= curr:
                msgs.append(("MACD_BEAR", {}, None, None))

        # SMA cross
        if sma_cross_on and i >= 1:
            prev = sma50.iloc[i-1] - sma200.iloc[i-1]
            curr = sma50.iloc[i]   - sma200.iloc[i]
            if not np.isnan(prev) and not np.isnan(curr):
                if prev < 0 <= curr:
                    msgs.append(("GOLDEN_CROSS", {}, None, None))
                elif prev > 0 >= curr:
                    msgs.append(("DEATH_CROSS", {}, None, None))

        # Price vs 200-week SMA
        if p200_on and not np.isnan(sma200.iloc[i]):
            pct = (close.iloc[i] - sma200.iloc[i]) / sma200.iloc[i] * 100.0
            thr = float(cfg["rules"]["price_vs_sma200"]["pct"])
            if pct >= thr:
                msgs.append(("PRICE_ABOVE_200W", {"pct": thr}, None, None))
            elif pct <= -thr:
                msgs.append(("PRICE_BELOW_200W", {"pct": thr}, None, None))

        # collect
        for sig, vals, stance, tag in msgs:
            out.append({
                "date": dt.strftime("%Y-%m-%d"),
                "symbol": t["symbol"],
                "name": t["name"],
                "signal": sig,
                "rationale": explain(sig, vals, stance, tag)
            })

    # Print CSV-style to stdout (and also save to file)
    if not out:
        print("date,symbol,name,signal,rationale")
        print("# No signals in window.")
        return

    df_out = pd.DataFrame(out)
    df_out = df_out.sort_values("date")
    df_out.to_csv("backtest_signals.csv", index=False)
    print("date,symbol,name,signal,rationale")
    for _, row in df_out.iterrows():
        print(f"{row['date']},{row['symbol']},{row['name']},{row['signal']},{row['rationale']}")
    print("\nSaved: backtest_signals.csv")
    print(f"Total alerts: {len(df_out)}")
