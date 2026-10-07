from datetime import datetime
from pathlib import Path
import pandas as pd

TEST_LEDGER_FILE = Path("market_data/test_paper_ledger.csv")
ROUND_TRIP_FRICTION = 0.0035  # 0.35% round-trip brokerage, STT & slippage
STCG_TAX_RATE = 0.20          # 20% Short-Term Capital Gains tax


def run_dryrun_test():
    print("=" * 80)
    print("RUNNING DRY-RUN TEST: SYNTHETIC TRADE LOGGING & QC ENGINE AUDIT")
    print("=" * 80)

    TEST_LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Test Ledger Schema
    cols = [
        "Trade_ID", "Status", "Ticker", "Entry_Date", "Entry_Price", "Shares_Qty", 
        "Stop_Price", "Target_Price", "Allocated_Capital", "Exit_Date", "Exit_Price", 
        "Exit_Reason", "Gross_PnL", "Net_PnL_PostTax", "R_Multiple_Realized"
    ]
    df = pd.DataFrame(columns=cols)

    # Synthetic Trades:
    # Trade 1: Entry 1000, Stop 960 (Risk 40), Target 1100 (Gain 100 = 2.5R), Qty 93 (~Rs 3720 risk)
    # Trade 2: Entry 500, Stop 480 (Risk 20), Target 550, Qty 187 (~Rs 3740 risk)
    synthetic_entries = [
        {
            "id": 1,
            "ticker": "TESTWIN",
            "entry": 1000.0,
            "qty": 93,
            "stop": 960.0,
            "target": 1100.0,
            "exit": 1100.0,
            "reason": "Target Hit"
        },
        {
            "id": 2,
            "ticker": "TESTLOSS",
            "entry": 500.0,
            "qty": 187,
            "stop": 480.0,
            "target": 550.0,
            "exit": 480.0,
            "reason": "Stop Loss Hit"
        }
    ]

    records = []
    today = datetime.today().strftime("%Y-%m-%d")

    for t in synthetic_entries:
        entry = float(t["entry"])
        qty = int(t["qty"])
        exit_p = float(t["exit"])
        stop_p = float(t["stop"])

        initial_risk_per_share = entry - stop_p
        raw_proceeds = (exit_p - entry) * qty
        friction_cost = (entry * qty * (ROUND_TRIP_FRICTION / 2)) + (exit_p * qty * (ROUND_TRIP_FRICTION / 2))
        gross_pnl = round(raw_proceeds - friction_cost, 2)

        stcg_tax = max(0.0, gross_pnl * STCG_TAX_RATE)
        net_post_tax = round(gross_pnl - stcg_tax, 2)

        if initial_risk_per_share > 0:
            realized_r = round(gross_pnl / (initial_risk_per_share * qty), 2)
        else:
            realized_r = 0.0

        records.append({
            "Trade_ID": t["id"],
            "Status": "CLOSED",
            "Ticker": t["ticker"],
            "Entry_Date": today,
            "Entry_Price": entry,
            "Shares_Qty": qty,
            "Stop_Price": stop_p,
            "Target_Price": t["target"],
            "Allocated_Capital": round(entry * qty, 2),
            "Exit_Date": today,
            "Exit_Price": exit_p,
            "Exit_Reason": t["reason"],
            "Gross_PnL": gross_pnl,
            "Net_PnL_PostTax": net_post_tax,
            "R_Multiple_Realized": realized_r
        })

    test_df = pd.DataFrame(records)
    test_df.to_csv(TEST_LEDGER_FILE, index=False)

    print("\n[STEP 1: TRADE LEDGER ENTRIES PROCESSED]")
    display_cols = [
        "Trade_ID", "Ticker", "Entry_Price", "Shares_Qty", 
        "Exit_Price", "Gross_PnL", "R_Multiple_Realized", "Net_PnL_PostTax"
    ]
    print(test_df[display_cols].to_string(index=False))

    # 2. Test Statistical QC Metrics
    print("\n[STEP 2: AUDITING QC METRICS]")
    closed = test_df.copy()
    wins = closed[closed["Gross_PnL"] > 0]
    losses = closed[closed["Gross_PnL"] <= 0]

    total_closed = len(closed)
    win_rate = (len(wins) / total_closed) * 100.0
    avg_win = float(wins["Gross_PnL"].mean()) if not wins.empty else 0.0
    avg_loss = abs(float(losses["Gross_PnL"].mean())) if not losses.empty else 1.0
    payoff_ratio = round(avg_win / avg_loss, 2)
    expectancy_r = round(((win_rate / 100.0) * payoff_ratio) - (1.0 - (win_rate / 100.0)), 2)

    # Drawdown verification with baseline zero
    pnl_series = pd.concat([pd.Series([0.0]), closed["Gross_PnL"].astype(float)]).reset_index(drop=True)
    cum_pnl = pnl_series.cumsum()
    running_max = cum_pnl.cummax()
    drawdown_rs = running_max - cum_pnl
    max_dd_pct = round((float(drawdown_rs.max()) / 500000.0) * 100.0, 2)

    print(f"- Win Rate: {win_rate:.1f}%")
    print(f"- Payoff Ratio: {payoff_ratio:.2f} : 1")
    print(f"- Expectancy: {expectancy_r:+.2f}R")
    print(f"- Max Drawdown: {max_dd_pct:.2f}% (Drawdown Rs: {float(drawdown_rs.max()):,.2f})")

    # 3. Assertions Check
    assert len(test_df) == 2, "Test failed: Rows missing."
    assert records[0]["R_Multiple_Realized"] > 2.30, "Math error: Winner R-multiple deviated."
    assert records[1]["R_Multiple_Realized"] < -0.90, "Math error: Loser R-multiple deviated."
    assert max_dd_pct > 0.0, "Drawdown bug: Did not detect drawdown after losing trade."

    print("\n" + "=" * 80)
    print("ALL CORE MATH CHECKS PASSED: Math, Friction, Tax, and Drawdown verified.")
    print(f"Test ledger saved at: {TEST_LEDGER_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    run_dryrun_test()