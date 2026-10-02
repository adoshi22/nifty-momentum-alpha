import io
import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates 14-period Average True Range (ATR) across historical daily bars."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def load_universe(
    ticker_file: str = "nse_ticker_list.txt",
    csv_fallback: str = "all_nse_tickers.csv"
) -> list[str]:
    """
    Dynamically loads the NSE universe:
    1. Reads plain-text ticker list if present.
    2. Reads all_nse_tickers.csv if present.
    3. Falls back to live Nifty 500 constituents directly from NSE if files are absent.
    """
    if os.path.exists(ticker_file):
        with open(ticker_file, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        print(f"Loaded {len(tickers)} tickers from '{ticker_file}'.")
        return tickers

    if os.path.exists(csv_fallback):
        df_meta = pd.read_csv(csv_fallback)
        if "TICKER" in df_meta.columns:
            tickers = df_meta["TICKER"].dropna().unique().tolist()
        elif "SYMBOL" in df_meta.columns:
            tickers = [f"{s.strip()}.NS" for s in df_meta["SYMBOL"].dropna().unique()]
        else:
            tickers = [f"{s.strip()}.NS" for s in df_meta.iloc[:, 0].dropna().unique()]
        print(f"Loaded {len(tickers)} tickers from '{csv_fallback}'.")
        return tickers

    # Fallback to direct download of Nifty 500 constituent list
    print("Local ticker list not found. Fetching live Nifty 500 universe from official source...")
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        df_nifty = pd.read_csv(url, storage_options=headers)
        tickers = [f"{sym.strip()}.NS" for sym in df_nifty["Symbol"].dropna().unique()]
        print(f"Successfully retrieved {len(tickers)} Nifty 500 tickers.")
        return tickers
    except Exception as e:
        print(f"Failed to fetch remote universe ({e}). Falling back to core watchlist.")
        return ["BEL.NS", "POLYCAB.NS", "APARINDS.NS", "NIFTYBEES.NS"]


def run_pipeline(
    output_summary_csv: str = "daily_screener_summary.csv",
    batch_size: int = 50,
    rate_limit_pause: float = 1.0
):
    tickers = load_universe()
    total_tickers = len(tickers)
    print(f"Starting scan for {total_tickers} equities...\n")

    summary_records = []
    failed_tickers = []

    # Process in chunks to prevent connection timeouts while optimizing yfinance throughput
    for i in range(0, total_tickers, batch_size):
        chunk = tickers[i : i + batch_size]
        print(f"Ingesting batch [{i + 1} - {min(i + batch_size, total_tickers)} / {total_tickers}]...")

        for ticker in chunk:
            clean_symbol = ticker.replace(".NS", "").replace(".BO", "")
            try:
                # 1 year of daily bars is sufficient for 50-DMA, 100-DMA, and 200-DMA
                df = yf.download(
                    ticker,
                    period="1y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False
                )

                if df.empty or len(df) < 100:
                    failed_tickers.append(clean_symbol)
                    continue

                # Flatten MultiIndex columns returned in recent yfinance updates
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Core Technical Indicators
                df["50_DMA"] = df["Close"].rolling(window=50).mean()
                df["100_DMA"] = df["Close"].rolling(window=100).mean()
                df["200_DMA"] = df["Close"].rolling(window=200).mean()
                df["ATR_14"] = calculate_atr(df, period=14)

                # Moving Average Pullback Metrics (%)
                df["Dist_50_DMA_%"] = ((df["Close"] - df["50_DMA"]) / df["50_DMA"]) * 100
                df["Dist_100_DMA_%"] = ((df["Close"] - df["100_DMA"]) / df["100_DMA"]) * 100

                # Extract latest bar
                latest = df.iloc[-1]
                close_px = float(latest["Close"])
                dma_50 = float(latest["50_DMA"])
                dma_100 = float(latest["100_DMA"])
                dma_200 = float(latest["200_DMA"]) if not pd.isna(latest["200_DMA"]) else np.nan
                dist_50 = float(latest["Dist_50_DMA_%"])
                dist_100 = float(latest["Dist_100_DMA_%"])

                # Quantitative Regime & Pullback Logic
                uptrend = dma_50 > dma_100
                near_50_dma = abs(dist_50) <= 2.0
                near_100_dma = abs(dist_100) <= 2.0

                if uptrend and near_50_dma:
                    pullback_signal = "50-DMA Pullback Support"
                elif uptrend and near_100_dma:
                    pullback_signal = "100-DMA Pullback Support"
                elif close_px < dma_100:
                    pullback_signal = "Below Trend Support"
                else:
                    pullback_signal = "Extended / No Pullback"
		
		# Calculate Trade Parameters
                # Stop Loss: 1.5x ATR below the 50-DMA
                stop_loss = round(dma_50 - (1.5 * float(latest["ATR_14"])), 2)
                risk_per_share = round(close_px - stop_loss, 2)

                # Position Sizing Example: 1% account risk on ₹5,00,000 capital (₹5,000 risk)
                account_capital = 500000
                risk_budget = account_capital * 0.01  # ₹5,000

                if risk_per_share > 0:
                    suggested_shares = int(risk_budget // risk_per_share)
                else:
                    suggested_shares = 0

                summary_records.append({
                    "Ticker": clean_symbol,
                    "Date": df.index[-1].strftime("%Y-%m-%d"),
                    "Close": round(close_px, 2),
                    "50_DMA": round(dma_50, 2),
                    "100_DMA": round(dma_100, 2),
                    "200_DMA": round(dma_200, 2) if not np.isnan(dma_200) else "N/A",
                    "Dist_to_50_DMA_%": round(dist_50, 2),
                    "Dist_to_100_DMA_%": round(dist_100, 2),
                    "14_Day_ATR": round(float(latest["ATR_14"]), 2),
                    "Uptrend_Aligned": bool(uptrend),
                    "Actionable_Signal": pullback_signal,
                    "Stop_Loss": stop_loss,                  # <-- New Column 1
                    "Suggested_Shares": suggested_shares     # <-- New Column 2
                })

            except Exception:
                failed_tickers.append(clean_symbol)
                continue

        # Polite rate-limiting pause between chunks
        time.sleep(rate_limit_pause)

    summary_df = pd.DataFrame(summary_records)
    summary_df.sort_values(by=["Actionable_Signal", "Ticker"], inplace=True)
    summary_df.to_csv(output_summary_csv, index=False)

    print("\n" + "=" * 60)
    print(f"Scan complete. Successfully processed: {len(summary_df)} stocks.")
    if failed_tickers:
        print(f"Skipped / Insufficient data: {len(failed_tickers)} symbols.")
    print(f"Master summary report generated: '{output_summary_csv}'.")
    print("=" * 60)

    # Show top pullback candidates
    pullbacks = summary_df[summary_df["Actionable_Signal"].str.contains("Pullback Support")]
    print(f"\nDiscovered {len(pullbacks)} actionable pullback setups:")
    print(pullbacks[["Ticker", "Close", "50_DMA", "Dist_to_50_DMA_%", "Actionable_Signal"]].head(15).to_string(index=False))

    return summary_df


if __name__ == "__main__":
    run_pipeline()