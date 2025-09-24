import os
import time
import yaml
import numpy as np
import pandas as pd
import yfinance as yf

# ---- config / constants ------------------------------------------------------
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
TICKER = "SPY"                  # use SPY as proxy for S&P500
ASSET_NAME = "S&P 500"
LOCAL_CSV = "data/spy_history.csv"   # written by refresh_data.py
OUT_DIR = "out"
SUMMARY_TXT = os.path.join(OUT_DIR, "sp500_backtest_summary_with_events.txt")
TRIGGERS_CSV = os.path.join(OUT_DIR, "sp500_backtest_triggers_with_events.csv")
RAW_CSV = os.path.join(OUT_DIR, "spy_rawdata.csv")
ERROR_TXT = os.path.join(OUT_DIR, "error.txt")

# ---- small utils -------------------------------------------------------------
def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def sma(s, w): return s.rolling(w).mean()

def rsi(series, period=14):
    d = series.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.rolling(period).mean()
    al = loss.rolling(period).mean()
    rs = (ag / (al.replace(0, np.nan)))
    return 100 - (100 / (1 + rs))

def pct(a, b):
    if b == 0 or np.isnan(a) or np.isnan(b): return np.nan
    return (a - b) / b * 100.0

# ---- data access -------------------------------------------------------------
def fetch_yf(symbol, years, interval="1wk", retries=3, sleep=5):
    last_err = None
    for i in range(1, retries + 1):
        try:
            print(f">>> yfinance {symbol} {years}y {interval} [try {i}/{retries}]")
            df = yf.download(symbol, period=f"{years}y", interval=interval,
                             auto_adjust=True, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # yfinance weekly columns may come as 'Close' or 'close'
                cols = {c: c.title() for c in df.columns}
                df = df.rename(columns=cols)
                if "Close" in df.columns:
                    df.index.name = "Date"
                    return df[["Close"]]
            last_err = "empty dataframe or missing Close"
        except Exception as e:
            last_err = str(e)
        time.sleep(sleep)
    raise RuntimeError(last_err or "yfinance failed")

def load_prices(symbol, years, interval="1wk"):
    # 1) Prefer cached CSV to avoid flaky network on CI
    if os.path.exists(LOCAL_CSV):
        print(f">>> Using cached CSV: {LOCAL_CSV}")
        df = pd.read_csv(LOCAL_CSV, parse_dates=["Date"]).set_index("Date")
        # ensure only Close column present with right dtype
        df = df.rename(columns={"close": "Close"})
        df = df[["Close"]].astype(float)
        return df

    # 2) Fallback to yfinance live fetch if cache is missing
    return fetch_yf(symbol, years, interval)

# ---- signal generation -------------------------------------------------------
def gen_signals_weekly(df, cfg):
    out = []
    mb = cfg["rules"]["data"]["min_bars"]
    if df is None or df.empty or len(df) < mb: return out

    close = df["Close"].dropna()
    idx = close.index

    rsi14 = rsi(close, cfg["rules"]["rsi"]["period"]) if cfg["rules"]["rsi"]["enabled"] else pd.Series(index=idx)
    s50 = close.rolling(cfg["rules"]["sma_cross"]["short"]).mean() if cfg["rules"]["sma_cross"]["enabled"] else pd.Series(index=idx)
    s200 = close.rolling(cfg["rules"]["sma_cross"]["long"]).mean() if cfg["rules"]["sma_cross"]["enabled"] else pd.Series(index=idx)

    for i in range(len(idx)):
        dt = idx[i]
        base = {"asset": ASSET_NAME, "date": dt.strftime("%Y-%m-%d"), "close": float(close.iloc[i])}

        # RSI
        if cfg["rules"]["rsi"]["enabled"] and not np.isnan(rsi14.iloc[i]):
            r = float(rsi14.iloc[i])
            if r <= cfg["rules"]["rsi"]["oversold"]:
                out.append({**base, "signal":"RSI", "direction":"bullish",
                            "technical_rationale": f"weekly RSI {r:.1f} <= {cfg['rules']['rsi']['oversold']}"})
            elif r >= cfg["rules"]["rsi"]["overbought"]:
                out.append({**base, "signal":"RSI", "direction":"bearish",
                            "technical_rationale": f"weekly RSI {r:.1f} >= {cfg['rules']['rsi']['overbought']}"})

        # SMA cross
        if cfg["rules"]["sma_cross"]["enabled"] and all([not np.isnan(s50.iloc[i]), not np.isnan(s200.iloc[i])]):
            prev = s50.iloc[i-1] - s200.iloc[i-1] if i > 0 else np.nan
            curr = s50.iloc[i] - s200.iloc[i]
            if not np.isnan(prev):
                if prev < 0 <= curr:
                    out.append({**base, "signal":"SMA Cross", "direction":"bullish",
                                "technical_rationale":"Golden Cross: SMA50 crosses above SMA200"})
                elif prev > 0 >= curr:
                    out.append({**base, "signal":"SMA Cross", "direction":"bearish",
                                "technical_rationale":"Death Cross: SMA50 crosses below SMA200"})

        # Price vs 200w
        if cfg["rules"]["price_vs_sma200"]["enabled"] and not np.isnan(s200.iloc[i]):
            dist = pct(close.iloc[i], s200.iloc[i])
            if not np.isnan(dist):
                npct = cfg["rules"]["price_vs_sma200"]["near_pct"]
                bpct = cfg["rules"]["price_vs_sma200"]["break_pct"]
                if abs(dist) <= npct:
                    out.append({**base,"signal":"Price≈SMA200","direction":"bullish",
                                "technical_rationale": f"Price within ±{npct}% of 200-week MA ({dist:.2f}%)"})
                elif dist <= -bpct:
                    out.append({**base,"signal":"Price≪SMA200","direction":"bullish",
                                "technical_rationale": f"Price below 200-week by ≥{bpct}% ({dist:.2f}%)"})
                elif dist >= bpct:
                    out.append({**base,"signal":"Price≫SMA200","direction":"bearish",
                                "technical_rationale": f"Price above 200-week by ≥{bpct}% ({dist:.2f}%)"})

    return out

# ---- events & alignment ------------------------------------------------------
def load_events(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["date","asset","event_type","bias","description"])
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["bias"] = df["bias"].str.lower()
    return df

def align_events(tech_df, events_df, window_days):
    if tech_df.empty:
        return tech_df.assign(event_date=None, event_bias=None, event_type=None,
                              event_description=None, aligned=False, bias_match=None)
    if events_df.empty:
        return tech_df.assign(event_date=None, event_bias=None, event_type=None,
                              event_description=None, aligned=False, bias_match=None)

    tech = tech_df.copy()
    tech["date_dt"] = pd.to_datetime(tech["date"])
    window = pd.Timedelta(days=window_days)
    ev = events_df.copy().sort_values("date")

    rows = []
    for _, tr in tech.iterrows():
        tdate = tr["date_dt"]
        cand = ev[(ev["date"] >= tdate - window) & (ev["date"] <= tdate + window)]
        if cand.empty:
            rows.append((None,None,None,None,False,"no_event_in_window")); continue
        cand = cand.assign(diff=(cand["date"] - tdate).abs()).sort_values("diff")
        e = cand.iloc[0]
        aligned = (tr["direction"] == e["bias"])
        rows.append((e["date"].strftime("%Y-%m-%d"), e["bias"], e["event_type"], e["description"], aligned,
                     "bias_match" if aligned else "bias_mismatch"))
    cols = ["event_date","event_bias","event_type","event_description","aligned","alignment_reason"]
    tech_df = pd.concat([tech_df, pd.DataFrame(rows, index=tech.index, columns=cols)], axis=1)
    tech_df.drop(columns=["date_dt"], inplace=True)
    return tech_df

# ---- main --------------------------------------------------------------------
def main():
    cfg = load_cfg(CONFIG_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        df = load_prices(TICKER,
                         years=cfg["rules"]["data"]["years"],
                         interval=cfg["rules"]["data"]["interval"])
        # save raw for debugging/traceability
        df.reset_index().rename(columns={"index":"Date"}).to_csv(RAW_CSV, index=False)
    except Exception as e:
        open(ERROR_TXT, "w").write(f"Download/load failed for {TICKER}: {e}\n")
        print(">>> out/error.txt written."); return

    tech_df = pd.DataFrame(gen_signals_weekly(df, cfg))
    if tech_df.empty:
        open(SUMMARY_TXT, "w").write("No technical triggers found for the period.\n")
        print(">>> Saved raw data; no triggers."); return

    ev = load_events(cfg["events"]["equity_csv"])
    aligned = align_events(tech_df, ev, cfg["events"]["window_days"])
    aligned.to_csv(TRIGGERS_CSV, index=False)

    total = len(aligned)
    aligned_yes = int(aligned["aligned"].sum())
    by_sig = aligned.groupby(["signal","direction"]).size().reset_index(name="count")
    by_aligned = aligned.groupby(["signal","direction","aligned"]).size().reset_index(name="count")

    lines = [
        f"Backtest: {ASSET_NAME} via {TICKER}, weekly, last {cfg['rules']['data']['years']} years",
        f"Total technical triggers: {total}",
        f"Tech + event aligned (±{cfg['events']['window_days']}d): {aligned_yes}",
        f"Not aligned / no event: {total - aligned_yes}",
        "",
        "Breakdown by signal & direction:",
    ]
    for _, r in by_sig.iterrows():
        lines.append(f"- {r['signal']} | {r['direction']}: {int(r['count'])}")
    lines.append("")
    for _, r in by_aligned.iterrows():
        lines.append(f"- {r['signal']} | {r['direction']} | aligned={bool(r['aligned'])}: {int(r['count'])}")

    open(SUMMARY_TXT, "w").write("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
