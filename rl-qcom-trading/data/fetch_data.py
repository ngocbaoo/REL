"""
Fetch QCOM daily OHLCV history and split it chronologically into
train (oldest 8 years of the last 10) and test (most recent 2 years) sets.

Usage:
    python data/fetch_data.py
    python data/fetch_data.py --csv path/to/manual_export.csv   # skip yfinance
"""
import argparse
import os
import sys

import pandas as pd

TICKER = "QCOM"
REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]
TOTAL_YEARS = 10
TRAIN_YEARS = 8
TEST_YEARS = 2


def fetch_via_yfinance(ticker: str = TICKER, years: int = TOTAL_YEARS) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker,
        period=f"{years}y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(columns={"Date": "Date"})
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return df


def load_manual_csv(path: str) -> pd.DataFrame:
    """Load a manually-downloaded CSV (e.g. stockscan.io export).

    Expects columns that can be mapped to Date, Open, High, Low, Close, Volume.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc == "date":
            rename_map[col] = "Date"
        elif lc == "open":
            rename_map[col] = "Open"
        elif lc == "high":
            rename_map[col] = "High"
        elif lc == "low":
            rename_map[col] = "Low"
        elif lc in ("close", "close/last", "adj close"):
            rename_map[col] = "Close"
        elif lc == "volume":
            rename_map[col] = "Volume"
    df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Manual CSV at {path} is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}. "
            "Expected Date, Open, High, Low, Close, Volume."
        )
    for col in ["Open", "High", "Low", "Close"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].astype(float)
    if df["Volume"].dtype == object:
        df["Volume"] = df["Volume"].astype(str).str.replace(",", "", regex=False)
    df["Volume"] = df["Volume"].astype(float)
    return df[REQUIRED_COLS]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)

    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    df = df.set_index("Date").reindex(full_range)
    df.index.name = "Date"
    df = df.ffill()
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df.reset_index()

    # keep only business days (5-day week) after forward-filling calendar gaps
    df = df[df["Date"].dt.dayofweek < 5].reset_index(drop=True)
    return df


def split_train_test(df: pd.DataFrame):
    df = df.sort_values("Date").reset_index(drop=True)
    last_date = df["Date"].max()
    window_start = last_date - pd.DateOffset(years=TOTAL_YEARS)
    df = df[df["Date"] >= window_start].reset_index(drop=True)

    train_end = df["Date"].min() + pd.DateOffset(years=TRAIN_YEARS)
    train_df = df[df["Date"] < train_end].reset_index(drop=True)
    test_df = df[df["Date"] >= train_end].reset_index(drop=True)
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to manual CSV export (skips yfinance)")
    parser.add_argument("--out-dir", type=str, default=os.path.dirname(__file__))
    args = parser.parse_args()

    if args.csv:
        print(f"Loading manual CSV from {args.csv}")
        raw = load_manual_csv(args.csv)
    else:
        try:
            print(f"Fetching {TICKER} via yfinance ({TOTAL_YEARS}y)...")
            raw = fetch_via_yfinance()
        except Exception as e:
            print(f"yfinance fetch failed: {e}", file=sys.stderr)
            print("Provide a manual CSV via --csv instead.", file=sys.stderr)
            sys.exit(1)

    df = clean(raw)
    train_df, test_df = split_train_test(df)

    train_path = os.path.join(args.out_dir, "qcom_train.csv")
    test_path = os.path.join(args.out_dir, "qcom_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {len(train_df)} rows [{train_df['Date'].min().date()} .. {train_df['Date'].max().date()}] -> {train_path}")
    print(f"Test:  {len(test_df)} rows [{test_df['Date'].min().date()} .. {test_df['Date'].max().date()}] -> {test_path}")


if __name__ == "__main__":
    main()
