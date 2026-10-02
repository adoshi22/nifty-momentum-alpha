import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress

# --- CONFIGURATION ---
PARQUET_FILE = Path("market_data/nifty500_5yr_daily.parquet")
WINDOW = 90             # 90 trading days (~4.5 calendar months)
MIN_PRICE = 20.0        # Filter out micro-penny stocks
MIN_TURNOVER = 5_00_00_000  # Minimum 20-day Average Daily Turnover: ₹5 Crore

def calculate_clenow_score(series: pd.Series) -> float:
    """
    Fits exponential regression over log(prices) against time steps.
    Returns: Annualized Exponential Slope (%) multiplied by R^2.
    """
    if len(series) < WINDOW:
        return np.nan
    
    y = np.log(series.values)
    x = np.arange(len(y))
    
    # Linear regression on log-prices
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Annualize exponential slope (assuming 250 trading days/year)
    annualized_slope = (np.exp(slope * 250) - 1) * 100
    r_squared = r_value ** 2
    
    # Clenow momentum score: trend velocity weighted by trend smoothness
    score = annualized_slope * r_squared
    return score

def run_ranking_pipeline():
    print(f"Loading data from {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    df.sort_values(by=["Ticker", "Date"], inplace=True)
    
    latest_date = df["Date"].max()
    print(f"Data evaluated as of latest session: {latest_date.date()}")
    
    results = []
    
    # Process ticker by ticker
    for ticker, group in df.groupby("Ticker"):
        if len(group) < 200:
            continue  # Require at least 200 bars for 200-DMA calculation
            
        group = group.copy()
        
        # Calculate moving averages
        group["SMA_50"] = group["Close"].rolling(50).mean()
        group["SMA_100"] = group["Close"].rolling(100).mean()
        group["SMA_200"] = group["Close"].rolling(200).mean()
        
        # 20-day Average Daily Turnover (Volume * Close) in Crores
        group["Turnover_Cr"] = (group["Volume"] * group["Close"]).rolling(20).mean() / 1e7
        
        # Latest bar snapshot
        latest = group.iloc[-1]
        close = latest["Close"]
        sma_50 = latest["SMA_50"]
        sma_100 = latest["SMA_100"]
        sma_200 = latest["SMA_200"]
        turnover = latest["Turnover_Cr"]
        
        # REGIME FILTERS:
        # 1. Price above 200-DMA (Macro bull regime)
        # 2. 50-DMA > 200-DMA (Medium-term structural uptrend)
        # 3. Minimum price and turnover hurdle
        if pd.isna(sma_200) or pd.isna(sma_50) or pd.isna(turnover):
            continue
            
        is_bullish_structure = (close > sma_200) and (sma_50 > sma_200)
        has_adequate_liquidity = (close >= MIN_PRICE) and (turnover >= 5.0)
        
        if not (is_bullish_structure and has_adequate_liquidity):
            continue
            
        # Calculate Clenow score on last 90 closing prices
        last_90_closes = group["Close"].tail(WINDOW)
        score = calculate_clenow_score(last_90_closes)
        
        # Pullback shelf distance: % distance from 50-DMA
        dist_50dma_pct = ((close - sma_50) / sma_50) * 100
        
        results.append({
            "Ticker": ticker.replace(".NS", ""),
            "Close": round(close, 2),
            "50_DMA": round(sma_50, 2),
            "Dist_50DMA_%": round(dist_50dma_pct, 2),
            "Turnover_20D_Cr": round(turnover, 2),
            "Clenow_Score": round(score, 2)
        })
        
    rank_df = pd.DataFrame(results)
    
    if rank_df.empty:
        print("No stocks passed the bullish regime and liquidity filters.")
        return
        
    # Rank descending by momentum score
    rank_df.sort_values(by="Clenow_Score", ascending=False, inplace=True)
    rank_df.reset_index(drop=True, inplace=True)
    rank_df.index += 1  # 1-based rank
    
    # Save the output CSV for review
    output_path = Path("market_data/nifty500_momentum_ranks.csv")
    rank_df.to_csv(output_path, index_label="Rank")
    print(f"\nSaved full rankings ({len(rank_df)} qualified stocks) to {output_path}")
    
    print("\n" + "=" * 75)
    print("TOP 15 MOMENTUM LEADERS (NIFTY 500)")
    print("=" * 75)
    print(rank_df.head(15)[["Ticker", "Close", "50_DMA", "Dist_50DMA_%", "Clenow_Score"]].to_string())
    
    print("\n" + "=" * 75)
    print("TOP CANDIDATES NEAR 50-DMA PULLBACK SHELF (Dist between -3% and +4%)")
    print("=" * 75)
    pullback_candidates = rank_df[rank_df["Dist_50DMA_%"].between(-3.0, 4.0)].head(10)
    print(pullback_candidates[["Ticker", "Close", "50_DMA", "Dist_50DMA_%", "Clenow_Score"]].to_string())

if __name__ == "__main__":
    run_ranking_pipeline()