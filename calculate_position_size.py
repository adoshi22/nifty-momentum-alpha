import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
PARQUET_FILE = Path("market_data/nifty500_5yr_daily.parquet")
RANKS_CSV = Path("market_data/nifty500_momentum_ranks.csv")
OUTPUT_ORDERS_CSV = Path("market_data/daily_trade_orders.csv")

# --- RISK PARAMETERS ---
TOTAL_PORTFOLIO_CAPITAL = 5_00_000   # ₹10,00,000 default (adjust to your capital)
RISK_PER_TRADE_PCT = 0.0075           # 0.75% portfolio risk per trade (₹7,500)
ATR_STOP_MULTIPLIER = 2.5             # Stop-loss set at 2.5 x ATR below entry
MAX_POSITION_WEIGHT_PCT = 0.15        # Hard cap: Max 15% capital per single stock (₹1,50,000)

def compute_atr(df_ticker: pd.DataFrame, period: int = 14) -> pd.Series:
    """Computes Wilders 14-day Average True Range."""
    high = df_ticker["High"]
    low = df_ticker["Low"]
    close_prev = df_ticker["Close"].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Exponential moving average with alpha = 1 / period (Wilder's ATR)
    atr = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr

def generate_orders():
    if not RANKS_CSV.exists():
        print(f"Error: {RANKS_CSV} not found. Run rank_momentum.py first.")
        return

    print("Loading momentum rankings and raw parquet historical bars...")
    ranks_df = pd.read_csv(RANKS_CSV)
    raw_df = pd.read_parquet(PARQUET_FILE)
    
    # Filter candidates near the 50-DMA base/pullback shelf (-3% to +4%)
    candidates = ranks_df[ranks_df["Dist_50DMA_%"].between(-3.0, 4.0)].copy()
    
    if candidates.empty:
        print("No candidates resting near the 50-DMA shelf today.")
        return

    # Select the top 10 candidates by Clenow score
    top_candidates = candidates.head(10).copy()
    
    orders = []
    rupee_risk_budget = TOTAL_PORTFOLIO_CAPITAL * RISK_PER_TRADE_PCT
    max_position_capital = TOTAL_PORTFOLIO_CAPITAL * MAX_POSITION_WEIGHT_PCT

    for _, row in top_candidates.iterrows():
        ticker = row["Ticker"]
        ticker_symbol = f"{ticker}.NS"
        
        # Extract ticker bars from parquet
        stock_bars = raw_df[raw_df["Ticker"] == ticker_symbol].sort_values("Date").copy()
        if len(stock_bars) < 30:
            continue
            
        stock_bars["ATR_14"] = compute_atr(stock_bars, period=14)
        latest = stock_bars.iloc[-1]
        
        entry_price = round(latest["Close"], 2)
        atr_value = round(latest["ATR_14"], 2)
        
        # Stop-loss price and risk gap
        risk_per_share = round(atr_value * ATR_STOP_MULTIPLIER, 2)
        stop_loss_price = round(entry_price - risk_per_share, 2)
        stop_loss_pct = round((risk_per_share / entry_price) * 100, 2)
        
        # Sizing math
        raw_quantity = int(rupee_risk_budget // risk_per_share) if risk_per_share > 0 else 0
        allocated_capital = raw_quantity * entry_price
        
        # Apply max exposure guardrail (15% cap)
        if allocated_capital > max_position_capital:
            raw_quantity = int(max_position_capital // entry_price)
            allocated_capital = raw_quantity * entry_price
            capped_flag = "YES"
        else:
            capped_flag = "NO"
            
        if raw_quantity <= 0:
            continue
            
        orders.append({
            "Rank": int(row["Rank"]),
            "Ticker": ticker,
            "Entry_LTP": entry_price,
            "ATR_14": atr_value,
            "Stop_Price": stop_loss_price,
            "SL_Dist_%": stop_loss_pct,
            "Shares_Qty": raw_quantity,
            "Trade_Value_Rs": round(allocated_capital, 2),
            "Max_Rs_Risk": round(raw_quantity * risk_per_share, 2),
            "Cap_Applied": capped_flag
        })
        
    orders_df = pd.DataFrame(orders)
    
    # Save output
    orders_df.to_csv(OUTPUT_ORDERS_CSV, index=False)
    
    print("\n" + "=" * 90)
    print(f"SYSTEMATIC ORDERS (Capital: ₹{TOTAL_PORTFOLIO_CAPITAL:,.0f} | Risk/Trade: ₹{rupee_risk_budget:,.0f} [0.75%])")
    print("=" * 90)
    print(orders_df[["Rank", "Ticker", "Entry_LTP", "Stop_Price", "SL_Dist_%", "Shares_Qty", "Trade_Value_Rs", "Max_Rs_Risk"]].to_string(index=False))
    print("=" * 90)
    print(f"\nOrder sheet successfully saved to: {OUTPUT_ORDERS_CSV}")

if __name__ == "__main__":
    generate_orders()