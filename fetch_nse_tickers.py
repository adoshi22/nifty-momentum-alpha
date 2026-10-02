import io
import requests
import pandas as pd

def fetch_all_nse_equities(output_file: str = "all_nse_tickers.csv") -> list[str]:
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print("Connecting to NSE Archives...")
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    # Read the official CSV into pandas
    df = pd.read_csv(io.StringIO(response.text))

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Filter for standard common equities only ('EQ' series)
    # This excludes warrants, preference shares, and debt bonds
    if "SERIES" in df.columns:
        df = df[df["SERIES"].str.strip() == "EQ"].copy()

    # Extract symbols and append '.NS' for yfinance compatibility
    df["TICKER"] = df["SYMBOL"].str.strip() + ".NS"

    # Export the full master dataframe (Symbol, Name of Company, ISIN, etc.)
    df.to_csv(output_file, index=False)
    
    # Also save a lightweight single-column ticker list
    ticker_list = df["TICKER"].tolist()
    with open("nse_ticker_list.txt", "w") as f:
        f.write("\n".join(ticker_list))

    print(f"Successfully fetched {len(ticker_list)} active NSE equity tickers.")
    print(f"Master details saved to: {output_file}")
    print("Plain ticker list saved to: nse_ticker_list.txt")

    return ticker_list

if __name__ == "__main__":
    tickers = fetch_all_nse_equities()
    print("\nSample tickers (first 10):", tickers[:10])