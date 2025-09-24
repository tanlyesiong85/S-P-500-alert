import os
import time
import pandas as pd
import yfinance as yf

OUT = "data/spy_history.csv"

def fetch_spy_weekly(years=15, interval="1wk", retries=4, sleep=6):
    last_err = None
    for i in range(1, retries + 1):
        try:
            print(f">>> yfinance: SPY {years}y {interval} [try {i}/{retries}]")
            df = yf.download("SPY", period=f"{years}y", interval=interval, auto_adjust=True, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                df = df.rename(columns=str.title)[["Close"]]
                df.index.name = "Date"
                return df
            last_err = "empty dataframe or missing Close"
        except Exception as e:
            last_err = str(e)
        time.sleep(sleep)
    raise RuntimeError(last_err)

def main():
    os.makedirs("data", exist_ok=True)
    df = fetch_spy_weekly()
    df = df.reset_index()  # Date, Close
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")

if __name__ == "__main__":
    main()
