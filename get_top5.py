import pandas as pd

# Load your generated summary
df = pd.read_csv("daily_screener_summary.csv")

# 1. Filter for Pullback Support candidates
candidates = df[df["Actionable_Signal"].str.contains("Pullback Support", na=False)].copy()

# 2. Apply Quantitative Filters:
#    - Positive Relative Strength vs Benchmark
#    - Normalized ATR under 4.0% (smooth, orderly price action)
filtered = candidates[
    (candidates["RS_Score_6M_%"] > 0) & 
    (candidates["Norm_ATR_%"] < 4.0)
].copy()

# 3. Sort by highest Relative Strength
top5 = filtered.sort_values(by="RS_Score_6M_%", ascending=False).head(5)

# Display the final selection
display_cols = [
    "Ticker", "Close", "50_DMA", "Dist_to_50_DMA_%", 
    "RS_Score_6M_%", "Norm_ATR_%", "RVol_20D", "Stop_Loss", "Suggested_Shares"
]

print("\n" + "=" * 80)
print("TOP 5 SYSTEMATIC CANDIDATES (RANKED BY RELATIVE STRENGTH)")
print("=" * 80)
print(top5[display_cols].to_string(index=False))