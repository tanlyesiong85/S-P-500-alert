# S&P 500 Backtest (10y) — Technical + Event Alignment (SPY proxy)

This backtests **S&P 500** using **SPY ETF** data (more reliable on CI than Yahoo's ^GSPC),
applies weekly technical rules, and checks for nearby macro events (±14 days).

## Run
- Actions → **Backtest S&P500 (10y) with Events** → **Run workflow**
- Artifact: **sp500-backtest-with-events**
  - `sp500_backtest_summary_with_events.txt` (counts & breakdowns)
  - `sp500_backtest_triggers_with_events.csv` (every trigger: date + technical + event rationale)

## Notes
- Educational only, not financial advice.
- Event CSV is curated heuristics and can be extended.
