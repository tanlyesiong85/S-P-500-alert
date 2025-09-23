# S&P 500 Backtest (10y) — Technical + Event Alignment

This repo backtests **S&P 500 (^GSPC)** on **weekly data** over the last **10 years**, firing a trigger when technical signals occur.  
It then checks for **nearby macro events** (from a curated CSV) within **±14 days** and reports whether the **event bias** aligns with the **technical direction**.

## Outputs
- `out/sp500_backtest_triggers_with_events.csv` — every trigger with:
  - date, signal, direction, close
  - technical_rationale
  - event_date, event_bias, event_type, event_description
  - aligned (True/False), alignment_reason
- `out/sp500_backtest_summary_with_events.txt` — total counts + breakdowns.

## Run (GitHub Actions)
- Go to **Actions → Backtest S&P500 (10y) with Events → Run workflow**.
- Download artifact **sp500-backtest-with-events** after it completes.

## How it works
- Weekly yfinance data (`^GSPC`) for the last 10 years.
- Technical rules (tunable in `config.yaml`):
  - RSI(14): oversold <30, overbought >70
  - SMA50/200 cross: Golden/Death Cross
  - Price vs 200-week SMA: near ±5%, deep break ±8%
- Event alignment:
  - `data/events_equity.csv` lists major macro/Fed milestones (2015–2024).
  - A technical trigger is marked **aligned** if a nearby event (±14 days) has a **bias** that matches the **technical direction**.

## Notes
- Educational use only. This is not financial advice.
- Event CSVs are curated heuristics; feel free to append more rows for your research.
- BTC/XRP CSVs are included for later use; this workflow currently runs **S&P 500** only.
