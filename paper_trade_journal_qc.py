import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

LEDGER_FILE = Path("market_data/paper_trade_ledger.csv")

# --- QC THRESHOLDS (BENCHMARK CRITERIA) ---
MIN_SAMPLE_SIZE = 25              # Minimum completed trade cycles required
TARGET_PAYOFF_MIN = 2.00          # Win/Loss Ratio must be >= 2.0 : 1
MIN_EXPECTANCY_R = 0.25           # Mathematical expectancy >= +0.25R per trade
MAX_DRAWDOWN_TOLERANCE_PCT = 20.0 # Paper MDD must not exceed 20%
MAX_ALLOWED_CONSEC_LOSSES = 8     # Reject if losing streak exceeds backtest max + 1
ROUND_TRIP_FRICTION = 0.0035      # 0.35% round-trip trading friction
STCG_TAX_RATE = 0.20              # 20% Short-Term Capital Gains tax

LEDGER_COLUMNS = [
    "Trade_ID", "Status", "Ticker", 
    "Entry_Date", "Entry_Price", "Shares_Qty", 
    "Stop_Price", "Target_Price", "Allocated_Capital",
    "Exit_Date", "Exit_Price", "Exit_Reason", 
    "Gross_PnL", "Net_PnL_PostTax", "R_Multiple_Realized"
]


def init_ledger() -> pd.DataFrame:
    """Initializes the CSV ledger if it does not exist."""
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER_FILE.exists():
        df = pd.DataFrame(columns=LEDGER_COLUMNS)
        df.to_csv(LEDGER_FILE, index=False)
        return df
    return pd.read_csv(LEDGER_FILE)


def log_new_entry():
    """Manually records a trade entry."""
    df = init_ledger()
    print("\n--- RECORD NEW PAPER TRADE ENTRY ---")
    
    ticker = input("Ticker Symbol (e.g. SONACOMS): ").strip().upper()
    entry_date = input("Entry Date (YYYY-MM-DD) [Default: Today]: ").strip()
    if not entry_date:
        entry_date = datetime.today().strftime("%Y-%m-%d")
        
    entry_price = float(input("Filled Entry Price (Rs): ").strip())
    shares_qty = int(input("Quantity of Shares: ").strip())
    stop_price = float(input("Stop-Loss Price (Rs): ").strip())
    target_price = float(input("Target Price (Rs): ").strip())
    
    trade_id = len(df) + 1
    allocated_cap = round(entry_price * shares_qty, 2)

    new_row = {
        "Trade_ID": trade_id,
        "Status": "OPEN",
        "Ticker": ticker,
        "Entry_Date": entry_date,
        "Entry_Price": entry_price,
        "Shares_Qty": shares_qty,
        "Stop_Price": stop_price,
        "Target_Price": target_price,
        "Allocated_Capital": allocated_cap,
        "Exit_Date": None,
        "Exit_Price": None,
        "Exit_Reason": None,
        "Gross_PnL": None,
        "Net_PnL_PostTax": None,
        "R_Multiple_Realized": None
    }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LEDGER_FILE, index=False)
    print(f"[SUCCESS] Trade #{trade_id} [{ticker}] logged as OPEN.")


def log_exit():
    """Closes an open trade and calculates PnL, R-multiples, and STCG tax."""
    df = init_ledger()
    open_trades = df[df["Status"] == "OPEN"]
    
    if open_trades.empty:
        print("No OPEN trades found in ledger to close.")
        return

    print("\nCURRENT OPEN TRADES:")
    display_cols = ["Trade_ID", "Ticker", "Entry_Date", "Entry_Price", "Shares_Qty", "Stop_Price", "Target_Price"]
    print(open_trades[display_cols].to_string(index=False))
    
    trade_id_input = input("\nEnter Trade_ID to close: ").strip()
    try:
        t_id = int(trade_id_input)
    except ValueError:
        print("Invalid ID.")
        return

    if t_id not in df["Trade_ID"].values:
        print("Trade_ID not found.")
        return

    idx = df[df["Trade_ID"] == t_id].index[0]
    if df.loc[idx, "Status"] == "CLOSED":
        print("Trade is already closed.")
        return

    exit_date = input("Exit Date (YYYY-MM-DD) [Default: Today]: ").strip()
    if not exit_date:
        exit_date = datetime.today().strftime("%Y-%m-%d")

    exit_price = float(input("Filled Exit Price (Rs): ").strip())
    exit_reason = input("Exit Reason [1: Target Hit, 2: Stop Hit, 3: Trend Broken, 4: Manual]: ").strip()
    reason_map = {"1": "Target Hit", "2": "Stop Loss Hit", "3": "Trend Broken (<50 DMA)", "4": "Manual Exit"}
    reason_str = reason_map.get(exit_reason, "Other")

    entry_price = float(df.loc[idx, "Entry_Price"])
    qty = int(df.loc[idx, "Shares_Qty"])
    initial_risk_per_share = entry_price - float(df.loc[idx, "Stop_Price"])
    
    # Financial Accounting: Gross, friction, and taxes
    raw_proceeds = (exit_price - entry_price) * qty
    friction_cost = (entry_price * qty * (ROUND_TRIP_FRICTION / 2)) + (exit_price * qty * (ROUND_TRIP_FRICTION / 2))
    gross_pnl = round(raw_proceeds - friction_cost, 2)
    
    stcg_tax = max(0.0, gross_pnl * STCG_TAX_RATE)
    net_post_tax = round(gross_pnl - stcg_tax, 2)

    # Realized R-Multiple
    if initial_risk_per_share > 0:
        realized_r = round(gross_pnl / (initial_risk_per_share * qty), 2)
    else:
        realized_r = 0.0

    df.loc[idx, "Status"] = "CLOSED"
    df.loc[idx, "Exit_Date"] = exit_date
    df.loc[idx, "Exit_Price"] = exit_price
    df.loc[idx, "Exit_Reason"] = reason_str
    df.loc[idx, "Gross_PnL"] = gross_pnl
    df.loc[idx, "Net_PnL_PostTax"] = net_post_tax
    df.loc[idx, "R_Multiple_Realized"] = realized_r

    df.to_csv(LEDGER_FILE, index=False)
    print(f"\n[SUCCESS] Trade #{t_id} CLOSED. Gross PnL: Rs {gross_pnl:,.2f} | Realized: {realized_r:+.2f}R | Net Post-Tax: Rs {net_post_tax:,.2f}")


def run_deployment_qc():
    """Audits closed paper trades against institutional readiness criteria."""
    df = init_ledger()
    closed = df[df["Status"] == "CLOSED"].copy()

    print("\n" + "=" * 80)
    print("QUANTITATIVE QC DEPLOYMENT AUDIT ENGINE")
    print("=" * 80)

    total_closed = len(closed)
    if total_closed == 0:
        print(f"QC Audit Aborted: 0 closed trades in ledger. Log at least {MIN_SAMPLE_SIZE} trades.")
        print("=" * 80)
        return

    wins = closed[closed["Gross_PnL"] > 0]
    losses = closed[closed["Gross_PnL"] <= 0]
    
    win_count = len(wins)
    win_rate = (win_count / total_closed) * 100.0

    avg_win = float(wins["Gross_PnL"].mean()) if not wins.empty else 0.0
    avg_loss = abs(float(losses["Gross_PnL"].mean())) if not losses.empty else 1.0
    payoff_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

    win_frac = win_rate / 100.0
    loss_frac = 1.0 - win_frac
    expectancy_r = round((win_frac * payoff_ratio) - loss_frac, 2)

    # Consecutive losses streak
    is_loss = (closed["Gross_PnL"] <= 0).astype(int)
    streaks = is_loss.groupby((~is_loss.astype(bool)).cumsum()).cumsum()
    max_streak = int(streaks.max()) if not streaks.empty else 0

    # Drawdown Calculation (Baseline 0.0 prepended to capture initial losses)
    pnl_series = pd.concat([pd.Series([0.0]), closed["Gross_PnL"].astype(float)]).reset_index(drop=True)
    cum_pnl = pnl_series.cumsum()
    running_max = cum_pnl.cummax()
    drawdown_rs = running_max - cum_pnl
    max_dd_pct = round((float(drawdown_rs.max()) / 500000.0) * 100.0, 2)

    total_gross = closed["Gross_PnL"].sum()
    total_net = closed["Net_PnL_PostTax"].sum()

    checks = [
        ("1. Sample Size", f"{total_closed} / {MIN_SAMPLE_SIZE} trades", total_closed >= MIN_SAMPLE_SIZE),
        ("2. Payoff Ratio", f"{payoff_ratio:.2f} : 1 (Min: {TARGET_PAYOFF_MIN:.2f})", payoff_ratio >= TARGET_PAYOFF_MIN),
        ("3. Expectancy", f"{expectancy_r:+.2f}R (Min: +{MIN_EXPECTANCY_R:.2f}R)", expectancy_r >= MIN_EXPECTANCY_R),
        ("4. Max Losing Streak", f"{max_streak} trades (Max Allowed: {MAX_ALLOWED_CONSEC_LOSSES})", max_streak <= MAX_ALLOWED_CONSEC_LOSSES),
        ("5. Portfolio Drawdown", f"{max_dd_pct:.2f}% (Limit: {MAX_DRAWDOWN_TOLERANCE_PCT:.2f}%)", max_dd_pct <= MAX_DRAWDOWN_TOLERANCE_PCT)
    ]

    all_passed = True
    print(f"{'CRITERIA':<25} | {'OBSERVED VALUE':<35} | {'STATUS'}")
    print("-" * 80)
    for name, observed, passed in checks:
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"{name:<25} | {observed:<35} | {status_str}")

    print("-" * 80)
    print(f"Summary: Win Rate: {win_rate:.1f}% | Gross PnL: Rs {total_gross:,.2f} | Net Post-Tax: Rs {total_net:,.2f}")
    print("=" * 80)

    if all_passed:
        print("\n[PASS] QC VERDICT: SYSTEM QUALIFIED FOR REAL CAPITAL DEPLOYMENT")
        print("  - All 5 empirical criteria match backtest parameters.")
        print("  - Action: Begin deployment at 25% scale (Rs 1,25,000 tranche).")
    else:
        print("\n[NO-GO] QC VERDICT: NOT READY FOR REAL MONEY")
        print("  - One or more risk gates failed. Continue logging paper trades.")
        print("  - Do NOT allocate live capital until all 5 criteria print PASS.")
    print("=" * 80 + "\n")


def main():
    while True:
        print("\n==================================")
        print(" PAPER TRADING CONTROL SUITE ")
        print("==================================")
        print("1. Log New Trade Entry (OPEN)")
        print("2. Log Trade Exit (CLOSE)")
        print("3. View Current Trade Ledger")
        print("4. Run Deployment QC Audit")
        print("5. Exit")
        choice = input("Select Option (1-5): ").strip()

        if choice == "1":
            log_new_entry()
        elif choice == "2":
            log_exit()
        elif choice == "3":
            df = init_ledger()
            if df.empty:
                print("Ledger is empty.")
            else:
                display_cols = ["Trade_ID", "Status", "Ticker", "Entry_Date", "Entry_Price", "Shares_Qty", "Exit_Date", "Exit_Price", "Gross_PnL", "R_Multiple_Realized"]
                print(df[display_cols].to_string(index=False))
        elif choice == "4":
            run_deployment_qc()
        elif choice == "5":
            sys.exit(0)
        else:
            print("Invalid selection.")


if __name__ == "__main__":
    main()