import io
import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates 14-period Average True Range (ATR)."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def fetch_benchmark_return(benchmark_ticker: str = "^NSEI", days: int = 126) -> float:
    """Fetches benchmark 6-month (126 trading days) return for relative strength calculation."""
    try:
        bench_df = yf.download(benchmark_ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(bench_df.columns, pd.MultiIndex):
            bench_df.columns = bench_df.columns.get_level_values(0)
        if len(bench_df) >= days:
            close_now = float(bench_df["Close"].iloc[-1])
            close_then = float(bench_df["Close"].iloc[-days])
            return ((close_now - close_then) / close_then) * 100.0
    except Exception as e:
        print(f"Could not fetch benchmark data ({e}). Defaulting benchmark return to 0.0%.")
    return 0.0


def load_universe(
    ticker_file: str = "nse_ticker_list.txt",
    csv_fallback: str = "all_nse_tickers.csv"
) -> list[str]:
    """Loads NSE universe with fallback to official live Nifty 500 list."""
    if os.path.exists(ticker_file):
        with open(ticker_file, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return tickers

    if os.path.exists(csv_fallback):
        df_meta = pd.read_csv(csv_fallback)
        if "TICKER" in df_meta.columns:
            return df_meta["TICKER"].dropna().unique().tolist()
        elif "SYMBOL" in df_meta.columns:
            return [f"{s.strip()}.NS" for s in df_meta["SYMBOL"].dropna().unique()]

    # Remote Fallback
    print("Local ticker list not found. Fetching live Nifty 500 from NSE...")
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        df_nifty = pd.read_csv(url, storage_options=headers)
        return [f"{sym.strip()}.NS" for sym in df_nifty["Symbol"].dropna().unique()]
    except Exception:
        return ["BEL.NS", "POLYCAB.NS", "APARINDS.NS", "IDBI.NS", "NIFTYBEES.NS"]


def run_pipeline(
    output_summary_csv: str = "daily_screener_summary.csv",
    batch_size: int = 50,
    rate_limit_pause: float = 1.0,
    account_capital: float = 500000.0,
    risk_pct: float = 0.01
):
    tickers = load_universe()
    total_tickers = len(tickers)
    
    # 1. Benchmark 6-Month Return (~126 trading days)
    print("Calculating benchmark performance (^NSEI)...")
    bench_6m_ret = fetch_benchmark_return("^NSEI", days=126)
    print(f"Nifty 50 6-Month Baseline Return: {bench_6m_ret:.2f}%\n")

    summary_records = []
    failed_tickers = []
    risk_budget = account_capital * risk_pct  # e.g., ₹5,000

    print(f"Starting scan for {total_tickers} equities...\n")

    for i in range(0, total_tickers, batch_size):
        chunk = tickers[i : i + batch_size]
        print(f"Ingesting batch [{i + 1} - {min(i + batch_size, total_tickers)} / {total_tickers}]...")

        for ticker in chunk:
            clean_symbol = ticker.replace(".NS", "").replace(".BO", "")
            try:
                df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)

                if df.empty or len(df) < 126:
                    failed_tickers.append(clean_symbol)
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Core Moving Averages
                df["50_DMA"] = df["Close"].rolling(window=50).mean()
                df["100_DMA"] = df["Close"].rolling(window=100).mean()
                df["200_DMA"] = df["Close"].rolling(window=200).mean()
                
                # New Parameter 1: 14-Day ATR & Normalized ATR %
                df["ATR_14"] = calculate_atr(df, period=14)
                df["Norm_ATR_%"] = (df["ATR_14"] / df["Close"]) * 100.0

                # New Parameter 2: Relative Volume (RVol 20D)
                df["Vol_20_SMA"] = df["Volume"].rolling(window=20).mean()
                df["RVol_20D"] = df["Volume"] / df["Vol_20_SMA"].replace(0, np.nan)

                # Distance to Moving Averages
                df["Dist_50_DMA_%"] = ((df["Close"] - df["50_DMA"]) / df["50_DMA"]) * 100.0
                df["Dist_100_DMA_%"] = ((df["Close"] - df["100_DMA"]) / df["100_DMA"]) * 100.0

                latest = df.iloc[-1]
                close_px = float(latest["Close"])
                dma_50 = float(latest["50_DMA"])
                dma_100 = float(latest["100_DMA"])
                dma_200 = float(latest["200_DMA"]) if not pd.isna(latest["200_DMA"]) else np.nan
                dist_50 = float(latest["Dist_50_DMA_%"])
                dist_100 = float(latest["Dist_100_DMA_%"])
                atr_14 = float(latest["ATR_14"])
                norm_atr = float(latest["Norm_ATR_%"])
                rvol_20d = float(latest["RVol_20D"]) if not pd.isna(latest["RVol_20D"]) else 1.0

                # New Parameter 3: 6-Month Relative Strength vs Nifty 50
                price_126d_ago = float(df["Close"].iloc[-126])
                stock_6m_ret = ((close_px - price_126d_ago) / price_126d_ago) * 100.0
                rs_score_6m = stock_6m_ret - bench_6m_ret

                # Trend & Pullback Logic
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

                # Trade Management Parameters
                stop_loss = round(dma_50 - (1.5 * atr_14), 2)
                risk_per_share = round(close_px - stop_loss, 2)
                suggested_shares = int(risk_budget // risk_per_share) if risk_per_share > 0 else 0

                summary_records.append({
                    "Ticker": clean_symbol,
                    "Date": df.index[-1].strftime("%Y-%m-%d"),
                    "Close": round(close_px, 2),
                    "50_DMA": round(dma_50, 2),
                    "100_DMA": round(dma_100, 2),
                    "200_DMA": round(dma_200, 2) if not np.isnan(dma_200) else "N/A",
                    "Dist_to_50_DMA_%": round(dist_50, 2),
                    "Dist_to_100_DMA_%": round(dist_100, 2),
                    "14_Day_ATR": round(atr_14, 2),
                    "Norm_ATR_%": round(norm_atr, 2),          # New Parameter 1
                    "RVol_20D": round(rvol_20d, 2),              # New Parameter 2
                    "RS_Score_6M_%": round(rs_score_6m, 2),      # New Parameter 3
                    "Uptrend_Aligned": bool(uptrend),
                    "Actionable_Signal": pullback_signal,
                    "Stop_Loss": stop_loss,
                    "Suggested_Shares": suggested_shares
                })

            except Exception:
                failed_tickers.append(clean_symbol)
                continue

        time.sleep(rate_limit_pause)

    summary_df = pd.DataFrame(summary_records)
    
    # Priority sorting: Actionable Pullbacks with highest Relative Strength first
    summary_df.sort_values(
        by=["Actionable_Signal", "RS_Score_6M_%"], 
        ascending=[True, False], 
        inplace=True
    )
    summary_df.to_csv(output_summary_csv, index=False)

    print("\n" + "=" * 70)
    print(f"Scan complete. Processed: {len(summary_df)} stocks.")
    print(f"Master summary exported to: '{output_summary_csv}'")
    print("=" * 70)

    # Preview Top 10 Pullback Candidates Ranked by Relative Strength
    pullbacks = summary_df[summary_df["Actionable_Signal"].str.contains("Pullback Support")].copy()
    if not pullbacks.empty:
        print("\nTop 10 Pullback Setups Ranked by Relative Strength:")
        display_cols = ["Ticker", "Close", "Dist_to_50_DMA_%", "RS_Score_6M_%", "RVol_20D", "Norm_ATR_%", "Suggested_Shares"]
        print(pullbacks[display_cols].head(10).to_string(index=False))

    return summary_df


if __name__ == "__main__":
    run_pipeline()