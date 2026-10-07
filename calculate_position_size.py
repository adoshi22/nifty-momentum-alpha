import numpy as np
import pandas as pd
from pathlib import Path

# --- FILE PATH CONFIGURATION ---
PARQUET_FILE = Path("market_data/nifty500_5yr_daily.parquet")
RANKS_CSV = Path("market_data/nifty500_momentum_ranks.csv")
OUTPUT_ORDERS_CSV = Path("market_data/daily_trade_orders.csv")
LEDGER_FILE = Path("market_data/paper_trade_ledger.csv")

# --- RISK PARAMETERS (ACCOUNT 2) ---
TOTAL_PORTFOLIO_CAPITAL = 500000.0   # Rs 5,00,000 deployment pool
RISK_PER_TRADE_PCT = 0.0075          # 0.75% portfolio risk per trade (Rs 3,750)
ATR_STOP_MULTIPLIER = 2.5             # Stop-loss set at 2.5 x ATR below entry
TARGET_R_MULTIPLE = 2.5               # Target set at 2.5R (matches backtest payoff)
MAX_POSITION_WEIGHT_PCT = 0.15        # Hard cap: Max 15% capital per stock (Rs 75,000)


def round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """Enforces NSE Rs 0.05 minimum tick size constraint for valid exchange orders."""
    return round(round(float(price) / tick_size) * tick_size, 2)


def check_macro_regime(raw_df: pd.DataFrame, required_consecutive_days: int = 2) -> dict:
    """
    Evaluates Nifty 50 regime with ticker normalization.
    Requires 2 consecutive closes above the 50-DMA to avoid single-day bull traps.
    """
    nifty = (
        raw_df[raw_df["Ticker"].isin(["^NSEI", "NIFTY 50", "NIFTY50.NS"])]
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .copy()
    )
    nifty["Close"] = pd.to_numeric(nifty["Close"], errors="coerce")
    nifty = nifty.dropna(subset=["Close"]).reset_index(drop=True)

    if len(nifty) < 52:
        return {"status": "INSUFFICIENT_DATA", "latest_close": 0.0, "latest_50dma": 0.0, "streak": 0}

    nifty["SMA_50"] = nifty["Close"].rolling(window=50, min_periods=40).mean()
    nifty["Above"] = nifty["Close"] > nifty["SMA_50"]

    last_close = float(nifty["Close"].iloc[-1])
    last_50dma = float(nifty["SMA_50"].iloc[-1])

    series = nifty["Above"].astype(int)
    streaks = series.groupby((~series.astype(bool)).cumsum()).cumsum()
    current_streak = int(streaks.iloc[-1])

    if current_streak >= required_consecutive_days:
        status = "BULLISH_CONFIRMED"
    elif current_streak == 1:
        status = "REGIME_WARMING_DAY_1"
    else:
        status = "BEARISH_DEFENSE"

    return {
        "status": status,
        "latest_close": last_close,
        "latest_50dma": last_50dma,
        "streak": current_streak
    }


def calculate_gtt_fields(stop_price: float, target_price: float, atr_14: float, turnover_cr: float) -> dict:
    """
    Computes exact Zerodha Kite GTT trigger and limit fields with tick-size rounding.
    Guarantees SL_Limit is strictly at least one full Rs 0.05 tick below SL_Trigger.
    """
    if turnover_cr >= 250.0:
        offset_pct = 0.0030  # 0.30%
    elif turnover_cr >= 50.0:
        offset_pct = 0.0050  # 0.50%
    else:
        offset_pct = 0.0075  # 0.75%

    sl_offset = max(0.10 * atr_14, stop_price * offset_pct)
    sl_trig = round_to_tick(stop_price)
    
    # Floor check: Ensure SL_Limit is never rounded back up to the Trigger price
    sl_lim = min(round_to_tick(stop_price - sl_offset), round_to_tick(sl_trig - 0.05))
    tgt_trig = round_to_tick(target_price)

    return {
        "SL_Trigger": sl_trig,
        "SL_Limit": sl_lim,
        "Target_Trigger": tgt_trig,
        "Target_Limit": tgt_trig
    }


def auto_log_to_ledger(orders_df: pd.DataFrame, max_trades: int = 3):
    """Directly pushes top generated orders into the paper trade ledger."""
    if orders_df.empty:
        return

    confirm = input(f"\nAuto-log top {min(max_trades, len(orders_df))} orders to paper_trade_ledger.csv? (y/n): ").strip().lower()
    if confirm != "y":
        print("Auto-log skipped. Order sheet saved in CSV.")
        return

    cols = [
        "Trade_ID", "Status", "Ticker", "Entry_Date", "Entry_Price", "Shares_Qty", 
        "Stop_Price", "Target_Price", "Allocated_Capital", "Exit_Date", "Exit_Price", 
        "Exit_Reason", "Gross_PnL", "Net_PnL_PostTax", "R_Multiple_Realized"
    ]
    ledger_df = pd.read_csv(LEDGER_FILE) if LEDGER_FILE.exists() else pd.DataFrame(columns=cols)

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    open_tickers = set(ledger_df[ledger_df["Status"] == "OPEN"]["Ticker"].unique()) if not ledger_df.empty else set()
    start_id = len(ledger_df) + 1
    new_rows = []

    for _, o in orders_df.iterrows():
        ticker = o["Ticker"]
        if ticker in open_tickers:
            print(f"Skipping {ticker}: Already an OPEN trade in ledger.")
            continue
        if len(new_rows) >= max_trades:
            break

        new_rows.append({
            "Trade_ID": start_id + len(new_rows),
            "Status": "OPEN",
            "Ticker": ticker,
            "Entry_Date": today,
            "Entry_Price": float(o["Entry_LTP"]),
            "Shares_Qty": int(o["Shares_Qty"]),
            "Stop_Price": float(o["SL_Trigger"]),
            "Target_Price": float(o["Target_Trigger"]),
            "Allocated_Capital": float(o["Trade_Value_Rs"]),
            "Exit_Date": None,
            "Exit_Price": None,
            "Exit_Reason": None,
            "Gross_PnL": None,
            "Net_PnL_PostTax": None,
            "R_Multiple_Realized": None
        })

    if new_rows:
        updated = pd.concat([ledger_df, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_csv(LEDGER_FILE, index=False)
        print(f"[SUCCESS] Logged {len(new_rows)} trade(s) to {LEDGER_FILE} under status OPEN.")


def generate_orders():
    if not (RANKS_CSV.exists() and PARQUET_FILE.exists()):
        print("[ERROR] Required CSV or Parquet missing. Run rank_momentum.py first.")
        return

    raw_df = pd.read_parquet(PARQUET_FILE)
    regime = check_macro_regime(raw_df, required_consecutive_days=2)
    
    print("\n" + "=" * 98)
    print(f"REGIME CHECK: Nifty @ {regime['latest_close']:,.2f} | 50-DMA @ {regime['latest_50dma']:,.2f} | Status: {regime['status']}")
    print("=" * 98)

    if regime["status"] == "REGIME_WARMING_DAY_1":
        print("STATUS: REGIME WARMING (Day 1 above 50-DMA). Waiting for Day 2 confirmation. Zero orders placed.")
        print("=" * 98)
        return
    elif regime["status"] == "BEARISH_DEFENSE":
        print("STATUS: BEARISH DEFENSE (Nifty < 50-DMA). Cash defense active. All buy orders suspended.")
        print("=" * 98)
        return

    ranks_df = pd.read_csv(RANKS_CSV)
    candidates = ranks_df[ranks_df["Dist_50DMA_%"].between(-3.0, 3.0)].head(10).copy()
    if candidates.empty:
        print("No candidates resting on the 50-DMA shelf today.")
        return

    risk_budget = TOTAL_PORTFOLIO_CAPITAL * RISK_PER_TRADE_PCT        # Rs 3,750
    max_cap = TOTAL_PORTFOLIO_CAPITAL * MAX_POSITION_WEIGHT_PCT       # Rs 75,000
    orders = []

    for _, row in candidates.iterrows():
        entry = round_to_tick(row["Close"])
        atr = float(row["ATR_14"])
        risk_per_share = round_to_tick(atr * ATR_STOP_MULTIPLIER)
        
        if risk_per_share <= 0:
            continue

        stop_price = round_to_tick(entry - risk_per_share)
        target_price = round_to_tick(entry + (TARGET_R_MULTIPLE * risk_per_share))

        # Mathematical Sizing: Bounds risk and enforces position cap simultaneously
        qty_risk = int(risk_budget / risk_per_share)
        qty_cap = int(max_cap / entry)
        
        qty = min(qty_risk, qty_cap)
        capped = "YES" if qty_cap < qty_risk else "NO"

        if qty <= 0:
            continue

        allocated = round(qty * entry, 2)
        turnover_cr = float(row.get("Turnover_20D_Cr", 100.0))
        gtt = calculate_gtt_fields(stop_price, target_price, atr, turnover_cr)

        orders.append({
            "Rank": int(row["Rank"]) if "Rank" in row else 0,
            "Ticker": str(row["Ticker"]),
            "Entry_LTP": entry,
            "Shares_Qty": qty,
            "SL_Trigger": gtt["SL_Trigger"],
            "SL_Limit": gtt["SL_Limit"],
            "Target_Trigger": gtt["Target_Trigger"],
            "Target_Limit": gtt["Target_Limit"],
            "Trade_Value_Rs": allocated,
            "Max_Rs_Risk": round(qty * (entry - stop_price), 2),
            "Cap_Applied": capped
        })

    orders_df = pd.DataFrame(orders)
    if orders_df.empty:
        print("Zero orders sized under risk parameters.")
        return

    orders_df.to_csv(OUTPUT_ORDERS_CSV, index=False)

    print("\n" + "=" * 98)
    print("ZERODHA KITE OCO GTT ORDER SHEET")
    print("=" * 98)
    display_cols = [
        "Rank", "Ticker", "Entry_LTP", "Shares_Qty", 
        "SL_Trigger", "SL_Limit", "Target_Trigger", "Target_Limit", 
        "Trade_Value_Rs", "Max_Rs_Risk", "Cap_Applied"
    ]
    print(orders_df[display_cols].to_string(index=False))
    print("=" * 98)
    print(f"Order sheet exported to: {OUTPUT_ORDERS_CSV}")

    print("\n" + "-" * 40 + " KITE OCO INPUTS (TOP 3) " + "-" * 40)
    for _, o in orders_df.head(3).iterrows():
        print(f"[{o['Ticker']}] Qty: {o['Shares_Qty']} | Buy CMP: Rs {o['Entry_LTP']}")
        print(f"  +- Stop-Loss -> Trigger: Rs {o['SL_Trigger']} | Limit Price: Rs {o['SL_Limit']}")
        print(f"  +- Target    -> Trigger: Rs {o['Target_Trigger']} | Limit Price: Rs {o['Target_Limit']}")
    print("-" * 105)

    auto_log_to_ledger(orders_df, max_trades=3)


if __name__ == "__main__":
    generate_orders()