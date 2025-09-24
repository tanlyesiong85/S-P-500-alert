import os
import time
import yaml
import numpy as np
import pandas as pd
import yfinance as yf

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
TICKER = "SPY"
ASSET_NAME = "S&P 500"
LOCAL_CSV = "data/spy_history.csv"

# ---- utils ----
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
    rs = ag / (al.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def pct(a, b):
    if b == 0 or np.isnan(a) or np.isnan(b): return np.nan
    return (a - b) / b * 100.0

# ---- data ----
def fetch_yf(symbol, years, interval="1wk", retries=3, sleep=5):
    err = None
    for i in range(1, retries+1):
        try:
            print(f">>> yfinance {symbol} {years}y {interval} [try {i}/{retries}]")
            df = yf.download(symbol, period=f"{years}y", interval=interval, auto_adjust=True, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                return df.rename(columns=str.title)[["Close"]]
            err = "empty or missing Close"
        except Exception as e:
            err = str(e)
        time.sleep(sleep)
    raise RuntimeError(err)

def load_prices(symbol, years, interval="1wk"):
    # 1) use cached CSV if present
    if os.path.exists(LOCAL_CSV):
        print(f">>> Using cached CSV: {LOCAL_CSV}")
        df = pd.read_csv(LOCAL_CSV, parse_dates=["Date"]).set_index("Date")
        return df[["Close"]]
    # 2) fallback to yfinance live fetch
    return fetch_yf(symbol, years, interval)

# ---- signals ----
def gen_signals_weekly(df, cfg):
    out = []
    mb = cfg["rules"]["data"]["min_bars"]
    if df is None or df.empty or len(df) < mb: return out
    close = df["Close"].dropna()
    idx = close.index
    rsi14 = rsi(close, cfg["rules"]["rsi"]["period"]) if cfg["rules"]["rsi"]["enabled"] else pd.Series(index=idx)
    s50   = close.rolling(cfg["rules"]["sma_cross"]["short"]).mean() if cfg["rules"]["sma_cross"]["enabled"] else pd.Series(index=idx)
    s200  = close.rolling(cfg["rules"]["sma_cross"]["long"]).mean() if (cfg["rules"]["sma_cross"]["enabled"] or cfg["rules"]["price_vs_sma200"]["enabled"]) else pd.Series(index=idx)

    for i in range(1, len(idx)):
        dt = idx[i]
        base = {"asset": ASSET_NAME, "date": dt.strftime("%Y-%m-%d"), "close": float(close.iloc[i])}
        # RSI
        if cfg["rules"]["rsi"]["enabled"] and not np.isnan(rsi14.iloc[i]):
            r = float(rsi14.iloc[i])
            if r <= cfg["rules"]["rsi"]["oversold"]:
                out.append({**base,"signal":"RSI","direction":"bullish","technical_rationale":f"Weekly RSI {r:.1f} ≤ {cfg['rules']['rsi']['oversold']} (oversold)"})
            elif r >= cfg["rules"]["rsi"]["overbought"]:
                out.append({**base,"signal":"RSI","direction":"bearish","technical_rationale":f"Weekly RSI {r:.1f} ≥ {cfg['rules']['rsi']['overbought']} (overbought)"})
        # SMA cross
        if cfg["rules"]["sma_cross"]["enabled"] and all([not np.isnan(s50.iloc[i]),not np.isnan(s200.iloc[i]),not np.isnan(s50.iloc[i-1]),not np.isnan(s200.iloc[i-1])]):
            prev = s50.iloc[i-1] - s200.iloc[i-1]
            curr = s50.iloc[i]   - s200.iloc[i]
            if prev < 0 <= curr:
                out.append({**base,"signal":"SMA Cross","direction":"bullish","technical_rationale":"Golden Cross: SMA50 crossed above SMA200"})
            elif prev > 0 >= curr:
                out.append({**base,"signal":"SMA Cross","direction":"bearish","technical_rationale":"Death Cross: SMA50 crossed below SMA200"})
        # Price vs 200w
        if cfg["rules"]["price_vs_sma200"]["enabled"] and not np.isnan(s200.iloc[i]):
            dist = pct(close.iloc[i], s200.iloc[i])
            if not np.isnan(dist):
                if abs(dist) <= cfg["rules"]["price_vs_sma200"]["near_pct"]:
                    out.append({**base,"signal":"Price~SMA200","direction":"bullish","technical_rationale":f"Price within {dist:.1f}% of 200-week SMA (mean-revert zone)"})
                elif dist <= -cfg["rules"]["price_vs_sma200"]["break_pct"]:
                    out.append({**base,"signal":"Price<<SMA200","direction":"bullish","technical_rationale":f"Price {dist:.1f}% below 200-week SMA (deep discount)"})
                elif dist >= cfg["rules"]["price_vs_sma200"]["break_pct"]:
                    out.append({**base,"signal":"Price>>SMA200","direction":"bearish","technical_rationale":f"Price {dist:.1f}% above 200-week SMA (extended)"})
    return out

# ---- events ----
def load_events(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["date","asset","event_type","bias","description"])
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["bias"] = df["bias"].str.lower()
    return df

def align_events(tech_df, events_df, window_days):
    if tech_df.empty:
        return tech_df.assign(event_date=None,event_bias=None,event_type=None,event_description=None,aligned=False,alignment_reason="no_tech")
    if events_df.empty:
        return tech_df.assign(event_date=None,event_bias=None,event_type=None,event_description=None,aligned=False,alignment_reason="no_events")
    tech_df = tech_df.copy()
    tech_df["date_dt"] = pd.to_datetime(tech_df["date"])
    window = pd.Timedelta(days=window_days)
    e = events_df.copy().sort_values("date")
    rows = []
    for _, tr in tech_df.iterrows():
        tdate = tr["date_dt"]
        cand = e[(e["date"] >= tdate - window) & (e["date"] <= tdate + window)]
        if cand.empty:
            rows.append((None,None,None,None,False,"no_event_in_window")); continue
        cand = cand.assign(diff=(cand["date"] - tdate).abs()).sort_values("diff")
        ev = cand.iloc[0]
        aligned = (tr["direction"] == ev["bias"])
        rows.append((ev["date"].strftime("%Y-%m-%d"),ev["bias"],ev["event_type"],ev["description"],aligned,"bias_match" if aligned else "bias_mismatch"))
    cols = ["event_date","event_bias","event_type","event_description","aligned","alignment_reason"]
    tech_df[cols] = pd.DataFrame(rows, index=tech_df.index)
    tech_df.drop(columns=["date_dt"], inplace=True)
    return tech_df

# ---- main ----
def main():
    cfg = load_cfg(CONFIG_PATH)
    os.makedirs("out", exist_ok=True)
    try:
        df = load_prices(TICKER, years=cfg["rules"]["data"]["years"], interval=cfg["rules"]["data"]["interval"])
    except Exception as e:
        open("out/error.txt","w").write(f"Download failed for {TICKER}: {e}\n")
        print(">>> out/error.txt written."); return

    tech_df = pd.DataFrame(gen_signals_weekly(df, cfg))
    if tech_df.empty:
        open("out/sp500_backtest_summary_with_events.txt","w").write("No technical triggers found for the period.\n")
        df.to_csv("out/spy_rawdata.csv"); print(">>> Saved raw data; no triggers."); return

    ev = load_events(cfg["events"]["equity_csv"])
    aligned = align_events(tech_df, ev, cfg["events"]["window_days"])
    aligned.to_csv("out/sp500_backtest_triggers_with_events.csv", index=False)

    total = len(aligned); aligned_yes = int(aligned["aligned"].sum())
    by_sig = aligned.groupby(["signal","direction"]).size().reset_index(name="count")
    by_aligned = aligned.groupby(["signal","direction","aligned"]).size().reset_index(name="count")

    lines = [
        f"Backtest: {ASSET_NAME} via {TICKER}, weekly, last {cfg['rules']['data']['years']} years",
        f"Total technical triggers: {total}",
        f"Tech + Event aligned (±{cfg['events']['window_days']}d): {aligned_yes}",
        f"Not aligned / no event: {total - aligned_yes}",
        "", "Breakdown by signal & direction:",
    ]
    for _, r in by_sig.iterrows():
        lines.append(f"- {r['signal']} / {r['direction']}: {int(r['count'])}")
    lines += ["", "Alignment breakdown (signal / direction / aligned):"]
    for _, r in by_aligned.iterrows():
        lines.append(f"- {r['signal']} / {r['direction']} / {bool(r['aligned'])}: {int(r['count'])}")

    open("out/sp500_backtest_summary_with_events.txt","w").write("\n".join(lines))
    print("\n".join(lines))
    print("Details CSV: out/sp500_backtest_triggers_with_events.csv")

if __name__ == "__main__":
    main()
