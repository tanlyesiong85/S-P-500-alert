# Long-Term Opportunity Alerts to WhatsApp (Tech + Events)

This repo sends **WhatsApp alerts via CallMeBot** only when a **strong long-term technical signal on weekly data** aligns with a **supportive current event**.  
Assets covered by default: **S&P 500 (^GSPC)**, **Bitcoin (BTC-USD)**, **XRP (XRP-USD)**.

## What you need
1. Register your phone with **CallMeBot** and get your API key.
2. In this repo: **Settings ➜ Secrets and variables ➜ Actions ➜ New repository secret**
   - `CALLMEBOT_APIKEY` – your CallMeBot key
   - `WHATSAPP_PHONE` – your phone in international format, e.g. `+6012XXXXXXX`

That’s it. No other keys required.

## How it works
- **Weekly data** from `yfinance` (5-year history).
- Technicals:
  - RSI(14) **oversold/overbought**
  - **SMA50/200** cross (Golden/Death Cross)
  - **Price vs 200-week SMA** (near/extended)
- Events layer:
  - Public RSS feeds (Fed, BLS, BEA, CoinDesk, CoinTelegraph, SEC).
  - Simple **keyword sentiment** per asset (equity/btc/xrp).
  - Only alerts if **technical direction aligns** with **event bias** (bullish with bullish, bearish with bearish).
- Runs **weekdays 09:10 MYT** by default. You can trigger manually with **Run workflow**.

## Customize (optional)
- Edit `config.yaml`:
  - Add/remove tickers.
  - Tweak RSI thresholds, SMA windows, “near/break” percentages.
  - Adjust RSS sources or keywords.
  - Change `events.lookback_days`.

## Local test
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CALLMEBOT_APIKEY=your_key
export WHATSAPP_PHONE=+6012XXXXXXX
python signals.py
 
