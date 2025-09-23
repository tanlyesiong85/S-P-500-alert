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
            elif diff >= cfg["rules"]["price_vs_sma200"]["break_pct"]:
                out.append(("bearish", f"{name}: Price {diff:.1f}% above 200-week SMA on {last_dt}", "Extended above 200-wk SMA"))

    # De-dupe headlines
    dedup = []
    seen = set()
    for d, h, c in out:
        if h not in seen:
            dedup.append((d, h, c))
            seen.add(h)
    return dedup

# ------------------ EVENTS ------------------
def within_days(dt, days):
    if not isinstance(dt, datetime):
        return False
    return dt >= datetime.now(timezone.utc) - timedelta(days=days)

def fetch_rss(url):
    try:
        return feedparser.parse(url)
    except Exception:
        return {"entries": []}

def norm_text(x):
    return (x or "").lower()

def classify_event(entry, cfg):
    """Return dict(asset->set({'bullish','bearish'})), plus macro bias for equities."""
    txt = " ".join([
        norm_text(entry.get("title")),
        norm_text(entry.get("summary")),
        norm_text(entry.get("description"))
    ])
    ts = entry.get("published_parsed") or entry.get("updated_parsed")
    if ts:
        dt = datetime(*ts[:6], tzinfo=timezone.utc)
    else:
        dt = datetime.now(timezone.utc)  # if missing, treat as current

    if not within_days(dt, cfg["events"]["lookback_days"]):
        return {}

    k = cfg["events"]["keywords"]
    tags = {}

    # Equities macro bias
    if any(w in txt for w in k["equity_bullish"]):
        tags.setdefault("equity", set()).add("bullish")
    if any(w in txt for w in k["equity_bearish"]):
        tags.setdefault("equity", set()).add("bearish")

    # Bitcoin
    if "bitcoin" in txt or "btc" in txt or any(w in txt for w in (k["btc_bullish"] + k["btc_bearish"])):
        if any(w in txt for w in k["btc_bullish"]):
            tags.setdefault("btc", set()).add("bullish")
        if any(w in txt for w in k["btc_bearish"]):
            tags.setdefault("btc", set()).add("bearish")

    # XRP / Ripple
    if "xrp" in txt or "ripple" in txt or any(w in txt for w in (k["xrp_bullish"] + k["xrp_bearish"])):
        if any(w in txt for w in k["xrp_bullish"]):
            tags.setdefault("xrp", set()).add("bullish")
        if any(w in txt for w in k["xrp_bearish"]):
            tags.setdefault("xrp", set()).add("bearish")

    # Attach short rationale from title
    if tags:
        tags["_why"] = entry.get("title", "")[:160]
        tags["_when"] = dt.strftime("%Y-%m-%d")
    return tags

def pull_events(cfg):
    sentiments = {
        "equity": {"bullish": set(), "bearish": set()},
        "btc": {"bullish": set(), "bearish": set()},
        "xrp": {"bullish": set(), "bearish": set()},
    }
    headlines = {"equity": [], "btc": [], "xrp": []}

    for feed in cfg["events"]["feeds"]:
        parsed = fetch_rss(feed["url"])
        for e in parsed.get("entries", []):
            tag = classify_event(e, cfg)
            if not tag: 
                continue
            why = tag.get("_why", "")
            when = tag.get("_when", "")
            if "equity" in tag:
                for pol in tag["equity"]:
                    sentiments["equity"][pol].add("1")
                if why:
                    headlines["equity"].append(f"[{when}] {why}")
            if "btc" in tag:
                for pol in tag["btc"]:
                    sentiments["btc"][pol].add("1")
                if why:
                    headlines["btc"].append(f"[{when}] {why}")
            if "xrp" in tag:
                for pol in tag["xrp"]:
                    sentiments["xrp"][pol].add("1")
                if why:
                    headlines["xrp"].append(f"[{when}] {why}")
    return sentiments, headlines

def event_alignment(asset_class, name, sentiments):
    """
    Map asset class to relevant sentiment bucket.
    Return 'bullish'/'bearish'/None.
    """
    if asset_class == "equity":
        pos = len(sentiments["equity"]["bullish"])
        neg = len(sentiments["equity"]["bearish"])
        if pos > neg:
            return "bullish"
        if neg > pos:
            return "bearish"
        return None
    elif name.lower().startswith("bitcoin"):
        pos = len(sentiments["btc"]["bullish"])
        neg = len(sentiments["btc"]["bearish"])
        if pos > neg:
            return "bullish"
        if neg > pos:
            return "bearish"
        return None
    elif name.lower().startswith("xrp"):
        pos = len(sentiments["xrp"]["bullish"])
        neg = len(sentiments["xrp"]["bearish"])
        if pos > neg:
            return "bullish"
        if neg > pos:
            return "bearish"
        return None
    return None

# ------------------ MESSAGE BUILDING ------------------
def combine_and_decide(tech_list, ev_bias, ev_headlines, name):
    """
    Only alert when at least ONE strong technical signal aligns with event bias direction.
    If no event bias (None), suppress (to avoid noise).
    """
    aligned = []
    for direction, headline, ctx in tech_list:
        if ev_bias is None:
            continue
        if direction == ev_bias:
            aligned.append((direction, headline, ctx))

    rationale_lines = []
    for d, h, c in aligned:
        rationale_lines.append(f"• {h}  | Technical: {c}  | Events: {ev_bias.upper()}")

    # Add 1–2 brief event headlines as “why”
    extras = []
    if ev_bias and name.lower().startswith("s&p"):
        extras = ev_headlines.get("equity", [])[:2]
    elif name.lower().startswith("bitcoin"):
        extras = ev_headlines.get("btc", [])[:2]
    elif name.lower().startswith("xrp"):
        extras = ev_headlines.get("xrp", [])[:2]

    if aligned:
        return rationale_lines, extras
    return [], []

def format_alert_block(asset_name, rationale_lines, extras):
    msg = [f"🟢 {asset_name}" if any("bullish" in ln.lower() for ln in rationale_lines) else f"🔴 {asset_name}"]
    msg += rationale_lines
    if extras:
        msg.append("Why (events):")
        for e in extras:
            msg.append(f"   - {e}")
    return "\n".join(msg)

# ------------------ MAIN ------------------
def main():
    cfg = load_cfg(CONFIG_PATH)

    # Pull events first (shared for all assets)
    sentiments, ev_headlines = pull_events(cfg)

    messages = []
    for t in cfg["tickers"]:
        df = dl_weekly(t["symbol"], cfg["rules"]["data"]["period"], cfg["rules"]["data"]["interval"])
        tech = technical_signals_weekly(df, t["name"], cfg, t["class"])
        if not tech:
            continue
        bias = event_alignment(t["class"], t["name"], sentiments)
        rationale, extras = combine_and_decide(tech, bias, ev_headlines, t["name"])
        if rationale:
            messages.append(format_alert_block(t["name"], rationale, extras))

    if not messages:
        print("No aligned (Tech + Events) alerts this run.")
        return

    header = f"📈 Long-Term Opportunity Alerts ({now_utc_iso()})"
    body = header + "\n\n" + "\n\n".join(messages)
    print(body)

    if not CALLMEBOT_APIKEY or not WHATSAPP_PHONE:
        print("Missing CALLMEBOT_APIKEY or WHATSAPP_PHONE. Printing instead of sending.")
        return

    status, resp = send_whatsapp(WHATSAPP_PHONE, CALLMEBOT_APIKEY, body)
    print("CallMeBot Status:", status)
    print("CallMeBot Response:", resp)

if __name__ == "__main__":
    main()
