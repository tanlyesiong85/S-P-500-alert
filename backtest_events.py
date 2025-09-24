name: Backtest S&P500 (10y) with Events

on:
  workflow_dispatch:

jobs:
  backtest:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      # just to see the cached CSV is present
      - name: Check CSV presence
        run: |
          ls -la
          ls -la data || true
          head -n 5 data/spy_history.csv || echo "CSV missing"

      - name: Run backtest
        run: |
          python backtest_events.py

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: sp500-backtest-with-events
          path: out/**
