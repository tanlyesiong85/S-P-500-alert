import os
import time
import io
import yaml
import numpy as np
import pandas as pd
import yfinance as yf

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

TICKER = "SPY"          # ETF proxy for S&P 500
ASSET_NAME = "S&P 500"  # friendly label in outputs

# ------------ Utils ------------
def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

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

# ------------ Data sources ------------
def fetch_alpha_vantage_weekly(symbol: str) -> pd.DataFrame:
    """
    Use Alpha Vantage TIME_SERIES_WEEKLY_ADJUSTED (CSV).
    Needs env ALPHA_VANTAGE_KEY.
    Returns DataFrame indexed by Date with column 'Close' (adjusted_close).
    """
    api_key = os.getenv("ALPHA_VANTAGE_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ALPHA_VANTAGE_KEY not set")

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={symbol}&apikey={api_key}&datatype=csv"

    # pandas can read CSV from URL directly
    df = pd.read_csv(url)
    # If rate-limited or error, AlphaVantage returns JSON; guard against that:
    if "timestamp" not in df.columns:
        raise RuntimeError("Alpha Vantage returned non-CSV (likely rate limit or error).")

    df.rename(columns={"timestamp": "Date", "adjusted_close": "Close"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Close"]].sort_values("Date").set_index("Date")
    return df

def fetch_yfinance_weekly(symbol: str, years: int, interval="1wk", retries=3, sleep_sec=3) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, retries+1):
        try:
            print(f">>> yfinance: downloading {symbol} ({years}y, {interval}) [try {attempt}/{retries}]")
            df = yf.download(symbol, period=f"{years}y", interval=interval, auto_adjust=True, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.rename(columns=str.title)[["Close"]]
            last_err = "empty dataframe"
        except Exception as e:
            last_err = str(e)
        time.sleep(sleep_sec)
    raise RuntimeError(f"yfinance failed for {symbol}: {last_err}")

def load_price_weekly(symbol: str, years: int, interval="1wk") -> pd.DataFrame:
    """
    Try Alpha Vantage first (API), fallback to yfinance.
    """
    try:
        print(f">>> Alpha Vantage: fetching {symbol} weekly adjusted CSV…")
        df = fetch_alpha_vantage_weekly(symbol)
        print(f">>> Alpha Vantage OK: {len(df)} rows")
        return df
    except Exception as e:
        print(f">>> Alpha Vantage failed: {e}")
        print(f">>> Falling back to yfinance.")
        return fetch_yfinance_weekly(symbol, years, interval)

# ------------ Technical Signals (weekly) ------------
def gen_signals_weekly(df, cfg):
    out = []
    mb = cfg["rules"]["data"]["min_bars"]
    if df is None or df.empty or len(df) < mb:
        return out

    close = df["Close"].dropna()
    idx = close.index

    rsi14 = rsi(close, cfg["rules"]["rsi"]["period"]) if cfg["rules"]["rsi"]["enabled"] else pd.Series(index=idx)
    s50   = sma(close, cfg["rules"]["sma_cross"]["short"]) if cfg["rules"]["sma_cross"]["enabled"] else pd.Series(index=idx)
    s200  = sma(close, cfg["rules"]["sma_cross"]["long"]) if (cfg["rules"]["sma_cross"]["enabled"] or cfg["rules"]["price_vs_sma200"]["enabled"]) else pd.Series(index=idx)

    for i in range(1, len(idx)):
        dt = idx[i]
        base = {
            "asset": ASSET_NAME,
            "date": dt.strftime("%Y-%m-%d"),
            "close": float(close.iloc[i]) if not np.isnan(close.iloc[i]) else None,
        }

        # RSI extremes
        if cfg["rules"]["rsi"]["enabled"] and not np.isnan(rsi14.iloc[i]):
            r = float(rsi14.iloc[i])
            if r <= cfg["rules"]["rsi"]["oversold"]:
                out.append({**base, "signal":"RSI", "direction":"bullish", "technical_rationale": f"Weekly RSI {r:.1f} ≤ {cfg['rules']['rsi']['oversold']} (oversold)"})
            elif r >= cfg["rules"]["rsi"]["overbought"]:
                out.append({**base, "signal":"RSI", "direction":"bearish", "technical_rationale": f"Weekly RSI {r:.1f} ≥ {cfg['rules']['rsi']['overbought']} (overbought)"})

        # SMA50/200 cross
        if cfg["rules"]["sma_cross"]["enabled"] and all([
            i >= 1, not np.isnan(s50.iloc[i]), not np.isnan(s200.iloc[i]),
            not np.isnan(s50.iloc[i-1]), not np.isnan(s200.iloc[i-1])
        ]):
            prev = s50.iloc[i-1] - s200.iloc[i-1]
            curr = s50.iloc[i]   - s200.iloc[i]
            if prev < 0 <= curr:
                out.append({**base, "signal":"SMA Cross", "direction":"bullish", "technical_rationale":"Golden Cross: SMA50 crossed above SMA200"})
            elif prev > 0 >= curr:
                out.append({**base, "signal":"SMA Cross", "direction":"bearish", "technical_rationale":"Death Cross: SMA50 crossed below SMA200"})

        # Price vs 200-week SMA
        if cfg["rules"]["price_vs_sma200"]["enabled"] and not np.isnan(s200.iloc[i]):
            dist = pct(close.iloc[i], s200.iloc[i])
            if not np.isnan(dist):
                if abs(dist) <= cfg["rules"]["price_vs_sma200"]["near_pct"]:
                    out.append({**base, "signal":"Price~SMA200", "direction":"bullish", "technical_rationale": f"Price within {dist:.1f}% of 200-week SMA (mean-revert zone)"})
                elif dist <= -cfg["rules"]["price_vs_sma200"]["break_pct"]:
                    out.append({**base, "signal":"Price<<SMA200", "direction":"bullish", "technical_rationale": f"Price {dist:.1f}% below 200-week SMA (deep discount)"})
                elif dist >= cfg["rules"]["price_vs_sma200"]["break_pct"]:
                    out.append({**base, "signal":"Price>>SMA200", "direction":"bearish", "technical_rationale": f"Price {dist:.1f}% above 200-week SMA (extended)"})

    return out

# ------------ Event Layer ------------
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
                              event_description=None, aligned=False, alignment_reason="no_tech")
    if events_df.empty:
        return tech_df.assign(event_date=None, event_bias=None, event_type=None,
                              event_description=None, aligned=False, alignment_reason="no_events")

    tech_df = tech_df.copy()
    tech_df["date_dt"] = pd.to_datetime(tech_df["date"])
    window = pd.Timedelta(days=window_days)

    e = events_df.copy().sort_values("date")

    results = []
    for _, tr in tech_df.iterrows():
        tdate = tr["date_dt"]
        low, high = tdate - window, tdate + window
        cand = e[(e["date"] >= low) & (e["date"] <= high)]
        if cand.empty:
            results.append((None, None, None, None, False, "no_event_in_window"))
            continue
        cand = cand.assign(diff=(cand["date"] - tdate).abs()).sort_values("diff")
        ev = cand.iloc[0]
        aligned = (tr["direction"] == ev["bias"])
        results.append((ev["date"].strftime("%Y-%m-%d"), ev["bias"], ev["event_type"], ev["description"],
                        aligned, "bias_match" if aligned else "bias_mismatch"))

    cols = ["event_date","event_bias","event_type","event_description","aligned","alignment_reason"]
    tech_df[cols] = pd.DataFrame(results, index=tech_df.index)
    tech_df.drop(columns=["date_dt"], inplace=True)
    return tech_df

# ------------ Main ------------
def main():
    cfg = load_cfg(CONFIG_PATH)
    os.makedirs("out", exist_ok=True)

    # Data (API first, fallback to yfinance)
    try:
        df = load_price_weekly(TICKER, years=cfg["rules"]["data"]["years"], interval=cfg["rules"]["data"]["interval"])
    except Exception as e:
        with open("out/error.txt", "w") as f:
            f.write(f"Download failed for {TICKER}: {e}\n")
        print(">>> out/error.txt written.")
        return

    # Generate technical triggers
    tech = gen_signals_weekly(df, cfg)
    tech_df = pd.DataFrame(tech)
    if tech_df.empty:
        with open("out/sp500_backtest_summary_with_events.txt", "w") as f:
            f.write("No technical triggers found for the period.\n")
        df.to_csv("out/spy_rawdata.csv")
        print(">>> Saved raw data; no triggers.")
        return

    # Load events & align
    ev_eq = load_events(cfg["events"]["equity_csv"])
    aligned = align_events(tech_df, ev_eq, cfg["events"]["window_days"])

    # Outputs
    full_csv = "out/sp500_backtest_triggers_with_events.csv"
    aligned.to_csv(full_csv, index=False)

    total = len(aligned)
    aligned_yes = int(aligned["aligned"].sum())
    by_sig = aligned.groupby(["signal","direction"]).size().reset_index(name="count")
    by_aligned = aligned.groupby(["signal","direction","aligned"]).size().reset_index(name="count")

    lines = [
        f"Backtest: {ASSET_NAME} via {TICKER}, weekly, last {cfg['rules']['data']['years']} years",
        f"Total technical triggers: {total}",
        f"Tech + Event aligned (±{cfg['events']['window_days']}d & bias match): {aligned_yes}",
        f"Not aligned / no event: {total - aligned_yes}",
        "", "Breakdown by signal & direction:"
    ]
    for _, r in by_sig.iterrows():
        lines.append(f"- {r['signal']} / {r['direction']}: {int(r['count'])}")
    lines += ["", "Alignment breakdown (signal / direction / aligned):"]
    for _, r in by_aligned.iterrows():
        lines.append(f"- {r['signal']} / {r['direction']} / {bool(r['aligned'])}: {int(r['count'])}")

    with open("out/sp500_backtest_summary_with_events.txt", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"Details CSV: {full_csv}")

if __name__ == "__main__":
    main()
