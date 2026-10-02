import pandas as pd

# Load the parquet file
file_path = "market_data/nifty500_5yr_daily.parquet"
df = pd.read_parquet(file_path)

# Print high-level metrics
print("=" * 50)
print(f"Total Rows:     {len(df):,}")
print(f"Unique Tickers: {df['Ticker'].nunique()}")
print(f"Date Range:     {df['Date'].min().date()} to {df['Date'].max().date()}")
print("=" * 50)

# Show the latest session for a few key momentum stocks
watchlist = ["BEL.NS", "POLYCAB.NS", "APARINDS.NS"]
sample = df[df["Ticker"].isin(watchlist)].groupby("Ticker").tail(1)
print("\nLatest Session Snapshot:")
print(sample[["Ticker", "Date", "Close", "Volume"]])