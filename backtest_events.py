import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

# ------------ Utils ------------
def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def sma(s, w):
    return s.rolling(w).mean()

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

def dl_weekly(symbol, years, interval="1wk"):
    df = yf.download(symbol, period=f"{years}y", interval=interval, auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.rename(columns=str.title)
    return df

# ------------ Technical Signals (weekly) ------------
def gen_signals_weekly(df, cfg):
    """
    Yield dict rows of technical triggers with rationale.
    """
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
        row_base = {
            "date": dt.strftime("%Y-%m-%d"),
            "close": float(close.iloc[i]) if not np.isnan(close.iloc[i]) else None,
        }

        # RSI extremes
        if cfg["rules"]["rsi"]["enabled"] and not np.isnan(rsi14.iloc[i]):
            r = float(rsi14.iloc[i])
            if r <= cfg["rules"]["rsi"]["oversold"]:
                out.append({**row_base, "signal":"RSI", "direction":"bullish", "technical_rationale": f"Weekly RSI {r:.1f} ≤ {cfg['rules']['rsi']['oversold']} (oversold)"})
            elif r >= cfg["rules"]["rsi"]["overbought"]:
                out.append({**row_base, "signal":"RSI", "direction":"bearish", "technical_rationale": f"Weekly RSI {r:.1f} ≥ {cfg['rules']['rsi']['overbought']} (overbought)"})

        # SMA50/200 cross
        if cfg["rules"]["sma_cross"]["enabled"] and all([
            i >= 1, not np.isnan(s50.iloc[i]), not np.isnan(s200.iloc[i]),
            not np.isnan(s50.iloc[i-1]), not np.isnan(s200.iloc[i-1])
        ]):
            prev = s50.iloc[i-1] - s200.iloc[i-1]
            curr = s50.iloc[i]   - s200.iloc[i]
            if prev < 0 <= curr:
                out.append({**row_base, "signal":"SMA Cross", "direction":"bullish", "technical_rationale":"Golden Cross: SMA50 crossed above SMA200"})
            elif prev > 0 >= curr:
                out.append({**row_base, "signal":"SMA Cross", "direction":"bearish", "technical_rationale":"Death Cross: SMA50 crossed below SMA200"})

        # Price vs 200-week SMA
        if cfg["rules"]["price_vs_sma200"]["enabled"] and not np.isnan(s200.iloc[i]):
            dist = pct(close.iloc[i], s200.iloc[i])
            if not np.isnan(dist):
                if abs(dist) <= cfg["rules"]["price_vs_sma200"]["near_pct"]:
                    out.append({**row_base, "signal":"Price~SMA200", "direction":"bullish", "technical_rationale": f"Price within {dist:.1f}% of 200-week SMA (mean-revert zone)"})
                elif dist <= -cfg["rules"]["price_vs_sma200"]["break_pct"]:
                    out.append({**row_base, "signal":"Price<<SMA200", "direction":"bullish", "technical_rationale": f"Price {dist:.1f}% below 200-week SMA (deep discount)"})
                elif dist >= cfg["rules"]["price_vs_sma200"]["break_pct"]:
                    out.append({**row_base, "signal":"Price>>SMA200", "direction":"bearish", "technical_rationale": f"Price {dist:.1f}% above 200-week SMA (extended)"})

    return out

# ------------ Event Layer ------------
def load_events(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["date","asset","event_type","bias","description"])
    df = pd.read_csv(csv_path)
    # normalize
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df["bias"] = df["bias"].str.lower()
    return df

def align_events(tech_df, events_df, window_days):
    """
    For each technical trigger, find the nearest event within ±window_days.
    If multiple exist, take the one with the smallest absolute date diff.
    """
    if events_df.empty or tech_df.empty:
        tech_df["event_date"] = None
        tech_df["event_bias"] = None
        tech_df["event_type"] = None
        tech_df["event_description"] = None
        tech_df["aligned"] = False
        tech_df["alignment_reason"] = "no_events"
        return tech_df

    tech_df = tech_df.copy()
    tech_df["date_dt"] = pd.to_datetime(tech_df["date"])
    window = pd.Timedelta(days=window_days)

    e = events_df.copy()
    e = e.sort_values("date")
    # Build fast lookup by window
    results = []
    for _, tr in tech_df.iterrows():
        tdate = tr["date_dt"]
        low, high = tdate - window, tdate + window
        cand = e[(e["date"] >= low) & (e["date"] <= high)]
        if cand.empty:
            results.append((None, None, None, None, False, "no_event_in_window"))
            continue
        # nearest by abs diff
        cand = cand.assign(diff=(cand["date"] - tdate).abs()).sort_values("diff")
        ev = cand.iloc[0]
        # Alignment if event bias matches direction
        aligned = (tr["direction"] == ev["bias"])
        results.append((ev["date"].strftime("%Y-%m-%d"), ev["bias"], ev["event_type"], ev["description"], aligned, "bias_match" if aligned else "bias_mismatch"))

    cols = ["event_date","event_bias","event_type","event_description","aligned","alignment_reason"]
    tech_df[cols] = pd.DataFrame(results, index=tech_df.index)
    tech_df.drop(columns=["date_dt"], inplace=True)
    return tech_df

# ------------ Main ------------
def main():
    cfg = load_cfg(CONFIG_PATH)
    os.makedirs("out", exist_ok=True)

    # ---- Data & signals (S&P 500) ----
    df = dl_weekly("^GSPC", years=cfg["rules"]["data"]["years"], interval=cfg["rules"]["data"]["interval"])
    if df is None or df.empty:
        print("Download failed or empty for ^GSPC.")
        return
    tech = gen_signals_weekly(df, cfg)
    tech_df = pd.DataFrame(tech)

    # ---- Events: equities macro timeline ----
    ev_eq = load_events(cfg["events"]["equity_csv"])

    # ---- Align within ±window_days ----
    aligned = align_events(tech_df, ev_eq, cfg["events"]["window_days"])

    # ---- Outputs ----
    # 1) Full list
    full_csv = "out/sp500_backtest_triggers_with_events.csv"
    aligned.to_csv(full_csv, index=False)

    # 2) Summary counts
    total = len(aligned)
    tech_only = total
    aligned_yes = aligned["aligned"].sum()
    aligned_no  = tech_only - aligned_yes

    by_sig = aligned.groupby(["signal","direction"]).size().reset_index(name="count")
    by_aligned = aligned.groupby(["signal","direction","aligned"]).size().reset_index(name="count")

    summary_lines = [
        "Backtest: ^GSPC (S&P 500), weekly, last 10 years",
        f"Total technical triggers: {tech_only}",
        f"Tech + Event aligned (±{cfg['events']['window_days']}d & bias match): {aligned_yes}",
        f"Not aligned / no event: {aligned_no}",
        "",
        "Breakdown by signal & direction:",
    ]
    for _, r in by_sig.iterrows():
        summary_lines.append(f"- {r['signal']} / {r['direction']}: {int(r['count'])}")

    summary_lines.append("")
    summary_lines.append("Alignment breakdown (signal / direction / aligned):")
    for _, r in by_aligned.iterrows():
        summary_lines.append(f"- {r['signal']} / {r['direction']} / {bool(r['aligned'])}: {int(r['count'])}")

    summary_txt = "\n".join(summary_lines)
    with open("out/sp500_backtest_summary_with_events.txt", "w") as f:
        f.write(summary_txt)

    print(summary_txt)
    print(f"Details CSV: {full_csv}")

if __name__ == "__main__":
    main()
 
