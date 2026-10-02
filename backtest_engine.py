import numpy as np
import pandas as pd
from pathlib import Path

PARQUET_FILE = Path("market_data/nifty500_5yr_daily.parquet")

# --- SIMULATION PARAMETERS ---
INITIAL_CAPITAL = 5_00_000      # ₹10,00,000 starting equity
RISK_PER_TRADE_PCT = 0.0075      # 0.75% portfolio risk per trade (₹7,500)
ROUND_TRIP_FRICTION = 0.0035     # 0.35% (STT, stamp duty, SEBI, GST, slippage)
STCG_TAX_RATE = 0.20             # 20% Short-Term Capital Gains tax
MAX_OPEN_POSITIONS = 8           # Up to 8 concurrent non-overlapping trades

def run_backtest():
    print("Loading historical dataset...")
    df = pd.read_parquet(PARQUET_FILE)
    df.sort_values(by=["Date", "Ticker"], inplace=True)
    
    # Pre-calculate indicators
    print("Pre-computing moving averages and ATR indicators...")
    all_tickers = []
    for ticker, group in df.groupby("Ticker"):
        if len(group) < 220:
            continue
        g = group.copy().sort_values("Date")
        g["SMA_50"] = g["Close"].rolling(50).mean()
        g["SMA_200"] = g["Close"].rolling(200).mean()
        
        # 14-day ATR
        high_low = g["High"] - g["Low"]
        high_cp = (g["High"] - g["Close"].shift(1)).abs()
        low_cp = (g["Low"] - g["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        g["ATR_14"] = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
        
        # 90-day momentum proxy (90-day rolling return * trend stability)
        ret_90 = g["Close"].pct_change(90)
        vol_90 = g["Close"].pct_change().rolling(90).std()
        g["Mom_Score"] = (ret_90 / (vol_90 + 1e-6)).fillna(0)
        
        all_tickers.append(g)
        
    master = pd.concat(all_tickers, ignore_index=True)
    master.sort_values(by="Date", inplace=True)
    
    trading_dates = master["Date"].drop_duplicates().sort_values().values
    
    # Portfolio Ledger State
    cash = float(INITIAL_CAPITAL)
    open_positions = {}   # {ticker: {"entry_price": float, "qty": int, "stop": float}}
    trade_ledger = []
    
    print(f"Simulating daily bars across {len(trading_dates)} sessions...")
    
    for current_date in trading_dates:
        day_data = master[master["Date"] == current_date].set_index("Ticker")
        
        # 1. PROCESS EXITS FIRST
        to_close = []
        for ticker, pos in open_positions.items():
            if ticker not in day_data.index:
                continue
            bar = day_data.loc[ticker]
            close_price = bar["Close"]
            stop_price = pos["stop"]
            sma_50 = bar["SMA_50"]
            
            # Exit Conditions:
            # - Hit ATR Stop-Loss
            # - OR Closed more than 3% below 50-DMA (trend broken)
            hit_stop = close_price <= stop_price
            trend_broken = close_price < (sma_50 * 0.97)
            
            if hit_stop or trend_broken:
                exit_price = close_price * (1.0 - (ROUND_TRIP_FRICTION / 2))  # Deduct exit friction
                proceeds = pos["qty"] * exit_price
                gross_pnl = (exit_price - pos["entry_price"]) * pos["qty"]
                
                trade_ledger.append({
                    "Ticker": ticker,
                    "Exit_Date": current_date,
                    "Entry_Price": pos["entry_price"],
                    "Exit_Price": round(exit_price, 2),
                    "Qty": pos["qty"],
                    "Gross_PnL": round(gross_pnl, 2),
                    "Exit_Reason": "Stop Loss" if hit_stop else "Trend Broken"
                })
                cash += proceeds
                to_close.append(ticker)
                
        for t in to_close:
            del open_positions[t]
            
        # 2. PROCESS ENTRIES (if capacity permits)
        slots_available = MAX_OPEN_POSITIONS - len(open_positions)
        if slots_available > 0:
            # Filter qualified candidates today
            pullback_candidates = day_data[
                (day_data["Close"] > day_data["SMA_200"]) &
                (day_data["SMA_50"] > day_data["SMA_200"]) &
                (day_data["Close"].between(day_data["SMA_50"] * 0.97, day_data["SMA_50"] * 1.04)) &
                (~day_data.index.isin(open_positions.keys()))
            ].copy()
            
            if not pullback_candidates.empty:
                ranked = pullback_candidates.sort_values(by="Mom_Score", ascending=False).head(slots_available)
                
                for ticker, row in ranked.iterrows():
                    entry_price = row["Close"] * (1.0 + (ROUND_TRIP_FRICTION / 2))  # Add entry friction
                    atr = row["ATR_14"]
                    stop_loss = round(entry_price - (2.5 * atr), 2)
                    risk_per_share = entry_price - stop_loss
                    
                    if risk_per_share <= 0:
                        continue
                        
                    risk_budget = INITIAL_CAPITAL * RISK_PER_TRADE_PCT
                    qty = int(risk_budget // risk_per_share)
                    cost = qty * entry_price
                    
                    # Cap check (max 15% per trade and must have cash)
                    if cost > (INITIAL_CAPITAL * 0.15):
                        qty = int((INITIAL_CAPITAL * 0.15) // entry_price)
                        cost = qty * entry_price
                        
                    if qty > 0 and cash >= cost:
                        cash -= cost
                        open_positions[ticker] = {
                            "entry_price": round(entry_price, 2),
                            "qty": qty,
                            "stop": stop_loss
                        }

    # Summary Calculations
    ledger_df = pd.DataFrame(trade_ledger)
    if ledger_df.empty:
        print("No completed trades generated.")
        return

    gross_profit = ledger_df[ledger_df["Gross_PnL"] > 0]["Gross_PnL"].sum()
    gross_loss = abs(ledger_df[ledger_df["Gross_PnL"] < 0]["Gross_PnL"].sum())
    net_pnl_pre_tax = ledger_df["Gross_PnL"].sum()
    stcg_tax_liability = max(0.0, net_pnl_pre_tax * STCG_TAX_RATE)
    net_post_tax_pnl = net_pnl_pre_tax - stcg_tax_liability
    
    total_trades = len(ledger_df)
    win_trades = len(ledger_df[ledger_df["Gross_PnL"] > 0])
    win_rate = (win_trades / total_trades) * 100
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else np.nan

    print("\n" + "=" * 65)
    print("WALK-FORWARD BACKTEST RESULTS (WITH 0.35% FRICTION + 20% STCG)")
    print("=" * 65)
    print(f"Total Completed Trades:    {total_trades}")
    print(f"Win Rate:                  {win_rate:.2f}%")
    print(f"Profit Factor:             {profit_factor}")
    print(f"Gross P&L (Pre-Tax):       ₹{net_pnl_pre_tax:,.2f}")
    print(f"Estimated 20% STCG Tax:    -₹{stcg_tax_liability:,.2f}")
    print(f"Net Post-Tax Profit:       ₹{net_post_tax_pnl:,.2f}")
    print(f"Final Account Balance:     ₹{(INITIAL_CAPITAL + net_post_tax_pnl):,.2f}")
    print("=" * 65)

if __name__ == "__main__":
    run_backtest()