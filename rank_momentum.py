import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress

# --- CONFIGURATION ---
PARQUET_FILE = Path("market_data/nifty500_5yr_daily.parquet")
CORE_HOLDINGS_FILE = Path("market_data/core_holdings.txt")
OUTPUT_RANKS_FILE = Path("market_data/nifty500_momentum_ranks.csv")

WINDOW = 90                 # 90 trading days (~4.5 calendar months)
MIN_PRICE = 20.0            # Filter out penny stocks
MIN_TURNOVER_CR = 5.0       # Minimum 20-day Average Daily Turnover: Rs 5 Crore
MAX_100DMA_EXT_PCT = 25.0   # Reject stocks extended >25% above 100-DMA


def load_core_exclusions(file_path: Path) -> set:
    """Reads core portfolio ticker exclusions from an external text file."""
    if not file_path.exists():
        print(f"[WARNING] Exclusion file '{file_path}' not found. Operating with zero exclusions.")
        return set()

    exclusions = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip().upper()
            if clean and not clean.startswith("#"):
                clean = clean.replace(".NS", "").replace(".BO", "")
                exclusions.add(clean)

    print(f"Loaded {len(exclusions)} core holdings to exclude from {file_path}")
    return exclusions


def calculate_clenow_score(series: pd.Series) -> float:
    """
    Fits exponential regression over log(prices) against time steps.
    Returns: Annualized Exponential Slope (%) multiplied by R^2.
    """
    if len(series) < WINDOW:
        return np.nan
    
    y = np.log(series.values)
    x = np.arange(len(y))
    
    slope, intercept, r_value, _, _ = linregress(x, y)
    annualized_slope = (np.exp(slope * 250) - 1) * 100
    r_squared = r_value ** 2
    
    return float(annualized_slope * r_squared)


def run_ranking_pipeline():
    if not PARQUET_FILE.exists():
        print(f"[ERROR] Parquet file not found at: {PARQUET_FILE}")
        return

    core_exclusions = load_core_exclusions(CORE_HOLDINGS_FILE)

    print(f"Loading data from {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    df["Ticker"] = df["Ticker"].astype(str)
    
    # Robust Date Normalization: Eliminates time and timezone mismatches
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df.sort_values(by=["Ticker", "Date"], inplace=True)
    
    latest_date = df["Date"].max()
    print(f"Data evaluated as of session: {latest_date.date()}")
    
    results = []
    
    for ticker, group in df.groupby("Ticker"):
        clean_symbol = ticker.replace(".NS", "").replace(".BO", "")
        
        # Exclude benchmark indices and Account 1 core holdings
        if clean_symbol in core_exclusions or clean_symbol in ["^NSEI", "NIFTY50", "NIFTY 50"]:
            continue

        if len(group) < 200 or group["Date"].iloc[-1] != latest_date:
            continue
            
        group = group.copy()
        group["Close"] = pd.to_numeric(group["Close"], errors="coerce")
        group["High"] = pd.to_numeric(group["High"], errors="coerce")
        group["Low"] = pd.to_numeric(group["Low"], errors="coerce")
        group["Volume"] = pd.to_numeric(group["Volume"], errors="coerce")

        group["SMA_50"] = group["Close"].rolling(50).mean()
        group["SMA_100"] = group["Close"].rolling(100).mean()
        group["SMA_200"] = group["Close"].rolling(200).mean()
        group["Turnover_Cr"] = (group["Volume"] * group["Close"]).rolling(20).mean() / 1e7
        
        # Wilder ATR 14
        h_l = group["High"] - group["Low"]
        h_cp = (group["High"] - group["Close"].shift(1)).abs()
        l_cp = (group["Low"] - group["Close"].shift(1)).abs()
        tr = pd.concat([h_l, h_cp, l_cp], axis=1).max(axis=1)
        group["ATR_14"] = tr.ewm(alpha=1.0 / 14, adjust=False).mean()

        latest = group.iloc[-1]
        close = latest["Close"]
        sma_50 = latest["SMA_50"]
        sma_100 = latest["SMA_100"]
        sma_200 = latest["SMA_200"]
        turnover = latest["Turnover_Cr"]
        atr = latest["ATR_14"]
        
        if any(pd.isna(x) for x in [close, sma_50, sma_100, sma_200, turnover, atr]):
            continue
            
        # Structure & Liquidity filters
        is_bullish = (close > sma_200) and (sma_50 > sma_100 > sma_200)
        has_liquidity = (close >= MIN_PRICE) and (turnover >= MIN_TURNOVER_CR)
        
        if not (is_bullish and has_liquidity):
            continue

        # Extension Check
        dist_100dma = ((close - sma_100) / sma_100) * 100
        if dist_100dma > MAX_100DMA_EXT_PCT:
            continue
            
        score = calculate_clenow_score(group["Close"].tail(WINDOW))
        if pd.isna(score):
            continue
            
        dist_50dma = ((close - sma_50) / sma_50) * 100
        
        results.append({
            "Ticker": clean_symbol,
            "Close": round(float(close), 2),
            "50_DMA": round(float(sma_50), 2),
            "100_DMA": round(float(sma_100), 2),
            "Dist_50DMA_%": round(float(dist_50dma), 2),
            "Dist_100DMA_%": round(float(dist_100dma), 2),
            "Turnover_20D_Cr": round(float(turnover), 2),
            "ATR_14": round(float(atr), 2),
            "Clenow_Score": round(float(score), 2)
        })
        
    rank_df = pd.DataFrame(results)
    if rank_df.empty:
        print("[WARNING] Zero stocks passed filtering.")
        return
        
    rank_df.sort_values(by="Clenow_Score", ascending=False, inplace=True)
    rank_df.reset_index(drop=True, inplace=True)
    rank_df.index += 1
    
    OUTPUT_RANKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    rank_df.to_csv(OUTPUT_RANKS_FILE, index_label="Rank")
    print(f"\n[SUCCESS] Saved {len(rank_df)} qualified momentum stocks to {OUTPUT_RANKS_FILE}")
    
    print("\n" + "=" * 90)
    print("TOP 15 MOMENTUM LEADERS")
    print("=" * 90)
    display_cols = ["Ticker", "Close", "50_DMA", "Dist_50DMA_%", "Dist_100DMA_%", "Turnover_20D_Cr", "ATR_14", "Clenow_Score"]
    print(rank_df.head(15)[display_cols].to_string())
    
    print("\n" + "=" * 90)
    print("PRIME PULLBACK SETUPS (Near 50-DMA Shelf: -3% to +3%)")
    print("=" * 90)
    pullbacks = rank_df[rank_df["Dist_50DMA_%"].between(-3.0, 3.0)].head(10)
    if not pullbacks.empty:
        print(pullbacks[display_cols].to_string())
    else:
        print("No momentum leaders currently sitting within +-3% of 50-DMA shelf.")


if __name__ == "__main__":
    run_ranking_pipeline()