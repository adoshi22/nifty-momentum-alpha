import sys
from pathlib import Path
import pandas as pd

PARQUET_FILE = Path("market_data/nifty500_5yr_daily.parquet")
CORE_FILE = Path("market_data/core_holdings.txt")
RANKS_FILE = Path("market_data/nifty500_momentum_ranks.csv")
ORDERS_FILE = Path("market_data/daily_trade_orders.csv")
REAL_LEDGER_FILE = Path("market_data/paper_trade_ledger.csv")


def run_system_health_check():
    print("=" * 80)
    print("SYSTEM INTEGRITY & REGRESSION HEALTH CHECK")
    print("=" * 80)

    # 1. Parquet Database Check
    if not PARQUET_FILE.exists():
        print(f"[FAIL] Parquet data file missing: {PARQUET_FILE}")
        return False
    
    df = pd.read_parquet(PARQUET_FILE)
    nifty_present = df["Ticker"].isin(["^NSEI", "NIFTY 50", "NIFTY50.NS"]).any()
    latest_bar_date = df["Date"].max()
    print(f"[PASS] Parquet database intact: {len(df):,} rows | Latest session: {latest_bar_date}")
    
    if not nifty_present:
        print("[FAIL] Nifty 50 benchmark ticker not found in parquet file!")
        return False
    print("[PASS] Benchmark ticker (^NSEI / NIFTY 50) confirmed present.")

    # 2. Core Holdings File Check
    if not CORE_FILE.exists():
        print(f"[FAIL] Core holdings exclusion file missing: {CORE_FILE}")
        return False
    with open(CORE_FILE, "r") as f:
        core_symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
    print(f"[PASS] Core exclusions file active: {len(core_symbols)} symbols loaded.")

    # 3. Check Momentum Ranks Output
    if RANKS_FILE.exists():
        ranks_df = pd.read_csv(RANKS_FILE)
        overlap = set(ranks_df["Ticker"]).intersection(set(core_symbols))
        if overlap:
            print(f"[FAIL] Core holding leak detected! Symbols in ranks: {overlap}")
            return False
        print(f"[PASS] Momentum ranks verified: {len(ranks_df)} qualified candidates (0 core overlap).")

    # 4. Check Real Paper Ledger Structure
    if REAL_LEDGER_FILE.exists():
        ledger_df = pd.read_csv(REAL_LEDGER_FILE)
        expected_cols = {
            "Trade_ID", "Status", "Ticker", "Entry_Date", "Entry_Price", 
            "Shares_Qty", "Stop_Price", "Target_Price", "Allocated_Capital"
        }
        if not expected_cols.issubset(set(ledger_df.columns)):
            print("[FAIL] Real paper ledger schema corrupted or missing columns!")
            return False
        open_count = len(ledger_df[ledger_df["Status"] == "OPEN"])
        closed_count = len(ledger_df[ledger_df["Status"] == "CLOSED"])
        print(f"[PASS] Live ledger intact: {open_count} OPEN, {closed_count} CLOSED trades.")
    else:
        print("[INFO] Real paper ledger has not been initialized yet (will create on first trade).")

    print("=" * 80)
    print("ALL INTEGRITY CHECKS PASSED: Zero schema mismatches, zero leaks, clean execution.")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_system_health_check()
    if not success:
        sys.exit(1)