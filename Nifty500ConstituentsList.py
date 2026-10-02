import io
import time
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("market_data")
DATA_DIR.mkdir(exist_ok=True)
PARQUET_FILE = DATA_DIR / "nifty500_5yr_daily.parquet"
YEARS_BACK = 5

def get_nifty500_tickers() -> list[str]:
    """Fetch live Nifty 500 constituent symbols directly from NSE India."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        # Format for Yahoo Finance: add .NS suffix
        tickers = [f"{sym.strip()}.NS" for sym in df["Symbol"].dropna().unique()]
        print(f"Fetched {len(tickers)} symbols from NSE.")
        return sorted(tickers)
    except Exception as e:
        print(f"Failed to fetch live NSE list ({e}). Falling back to sample basket.")
        return ["BEL.NS", "POLYCAB.NS", "APARINDS.NS", "ADANIPORTS.NS", "KOTAKBANK.NS"]

def download_nifty500_history(tickers: list[str], period_years: int = 5) -> pd.DataFrame:
    """
    Downloads historical data in chunks using yfinance vectorized multi-threading.
    Avoids single-ticker for-loop rate limits.
    """
    start_time = time.time()
    chunk_size = 75  # Optimal batch size to prevent HTTP 429 drops
    all_chunks = []
    
    total_chunks = (len(tickers) + chunk_size - 1) // chunk_size
    print(f"Starting batched download across {len(tickers)} tickers in {total_chunks} tranches...")

    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i : i + chunk_size]
        tranche_num = (i // chunk_size) + 1
        print(f"Downloading Tranche {tranche_num}/{total_chunks} ({len(batch)} tickers)...")
        
        # Vectorized download with multi-threading enabled
        data = yf.download(
            tickers=batch,
            period=f"{period_years}y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,  # Automatically adjusts Open, High, Low, Close for splits/dividends
            threads=True,
            progress=False
        )
        
        # Reshape multi-index columns into a standardized long format: [Date, Ticker, OHLCV]
        stacked = (
            data.stack(level=0, future_stack=True)
            .reset_index()
            .rename(columns={"level_1": "Ticker"})
        )
        all_chunks.append(stacked)
        
        # Gentle pacing between tranches
        time.sleep(1.0)
        
    master_df = pd.concat(all_chunks, ignore_index=True)
    
    # Clean up column names and types
    master_df.columns = [c.replace(" ", "_").capitalize() for c in master_df.columns]
    master_df["Date"] = pd.to_datetime(master_df["Date"])
    master_df.sort_values(by=["Ticker", "Date"], inplace=True)
    master_df.reset_index(drop=True, inplace=True)
    
    elapsed = round(time.time() - start_time, 2)
    print(f"Download complete in {elapsed}s. Total rows: {len(master_df):,}")
    return master_df

def save_and_verify(df: pd.DataFrame):
    """Save to Parquet for fast reading and run sanity checks."""
    df.to_parquet(PARQUET_FILE, engine="pyarrow", compression="snappy")
    print(f"Saved dataset to {PARQUET_FILE} ({PARQUET_FILE.stat().st_size / (1024*1024):.2f} MB)")
    
    # Run immediate data integrity check
    print("\n--- SANITY AUDIT ---")
    print("Unique Tickers:", df["Ticker"].nunique())
    print("Date Range:", df["Date"].min().date(), "to", df["Date"].max().date())
    print("Null Counts:\n", df.isnull().sum()[df.isnull().sum() > 0])

if __name__ == "__main__":
    symbols = get_nifty500_tickers()
    dataset = download_nifty500_history(symbols, period_years=YEARS_BACK)
    save_and_verify(dataset)