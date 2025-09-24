name: Refresh SPY Weekly CSV

on:
  workflow_dispatch:
  schedule:
    - cron: "0 0 * * 1" # every Monday at 00:00 UTC

jobs:
  refresh:
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
          pip install yfinance pandas

      - name: Generate data/spy_history.csv (yfinance)
        run: |
          python scripts/refresh_data.py

      - name: Commit updated CSV if changed
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          # Always stage the CSV (create or update)
          git add -A data/spy_history.csv
          # Commit only if something is staged
          if git diff --cached --quiet; then
            echo "No changes in data/spy_history.csv"
          else
            git commit -m "chore(data): refresh SPY weekly CSV [skip ci]"
            git push
          fi
