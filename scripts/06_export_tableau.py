# 06_export_tableau.py
# Geopolitical Conflict and Oil Stock Performance
# Group 26 - Sana Shah, Maya Knutsvig, Illina Islam
# 
# Generates 5 clean CSVs for Tableau dashboard building.
# Run this after 02_etl_cleaning.py has produced the cleaned data files.
# Output: data/tableau/ folder

import pandas as pd
import numpy as np
import os

os.makedirs("data/tableau", exist_ok=True)

print("Loading cleaned data...")
df  = pd.read_csv("data/cleaned/all_stocks_cleaned.csv")
oil = pd.read_csv("data/cleaned/oil/wti_oil_prices_cleaned.csv")

df["date"]  = pd.to_datetime(df["date"])
oil["date"] = pd.to_datetime(oil["date"])

# Daily return per ticker per conflict period
df["daily_return"] = df.groupby(["ticker", "conflict_period"])["close"].pct_change() * 100

PHASE_ORDER = {"Before": 1, "During": 2, "After": 3}

# ─────────────────────────────────────────────────────────
# EXPORT 1: GW1 vs GW2 Ticker Comparison
# Used for: Dashboard 1 — bar chart, scatter plot
# Rows: 7 tickers × 2 wars × 3 phases = 42 rows
# ─────────────────────────────────────────────────────────
print("Generating export 1: GW1 vs GW2 comparison...")

COMMON_TICKERS = ["BP", "COP", "CVX", "SHEL", "SU", "TTE", "XOM"]
gw_df = df[
    df["conflict_period"].isin(["First Gulf War", "Second Gulf War"]) &
    df["ticker"].isin(COMMON_TICKERS)
].copy()

phase_stats = gw_df.groupby(["ticker", "region", "conflict_period", "period_phase"]).agg(
    avg_close=("close", "mean"),
    avg_daily_return=("daily_return", "mean"),
    volatility=("daily_return", "std")
).reset_index().round(4)

before = phase_stats[phase_stats["period_phase"] == "Before"][["ticker", "conflict_period", "avg_close"]].rename(
    columns={"avg_close": "before_avg_close"})
export1 = phase_stats.merge(before, on=["ticker", "conflict_period"], how="left")
export1["pct_change_vs_before"] = ((export1["avg_close"] - export1["before_avg_close"]) / export1["before_avg_close"] * 100).round(2)
export1["phase_order"] = export1["period_phase"].map(PHASE_ORDER)
export1 = export1.sort_values(["ticker", "conflict_period", "phase_order"])

export1.to_csv("data/tableau/01_gw1_vs_gw2_comparison.csv", index=False)
print(f"  Saved: 01_gw1_vs_gw2_comparison.csv ({len(export1)} rows)")

# ─────────────────────────────────────────────────────────
# EXPORT 2: Phase Summary (volatility + oil price)
# Used for: Dashboard 2 — grouped bar, line charts
# Rows: 3 conflicts × 3 phases = 9 rows
# Note: US-Iran oil prices are approximate (EIA data ends 2008)
# ─────────────────────────────────────────────────────────
print("Generating export 2: Phase summary...")

vol = df.groupby(["conflict_period", "period_phase"])["daily_return"].agg(["std", "mean"]).reset_index()
vol.columns = ["conflict_period", "period_phase", "volatility", "avg_return"]
vol = vol.round(4)

oil_phase = oil[oil["period_phase"].isin(["Before", "During", "After"])].groupby(
    ["conflict_period", "period_phase"])["wti_price"].mean().reset_index()
oil_phase.columns = ["conflict_period", "period_phase", "avg_oil_price"]
oil_phase["avg_oil_price"] = oil_phase["avg_oil_price"].round(2)

# US-Iran oil approximations (from public WTI records, EIA data ends 2008)
us_iran_oil = pd.DataFrame([
    {"conflict_period": "US-Iran", "period_phase": "Before", "avg_oil_price": 57.04},
    {"conflict_period": "US-Iran", "period_phase": "During", "avg_oil_price": 68.12},
    {"conflict_period": "US-Iran", "period_phase": "After",  "avg_oil_price": 72.34},
])
oil_phase = pd.concat([oil_phase, us_iran_oil], ignore_index=True)

export2 = vol.merge(oil_phase, on=["conflict_period", "period_phase"], how="left")
export2["phase_order"] = export2["period_phase"].map(PHASE_ORDER)
export2 = export2.sort_values(["conflict_period", "phase_order"])

export2.to_csv("data/tableau/02_phase_summary.csv", index=False)
print(f"  Saved: 02_phase_summary.csv ({len(export2)} rows)")

# ─────────────────────────────────────────────────────────
# EXPORT 3: Full Ticker-Phase Breakdown (all 10 tickers, all 3 conflicts)
# Used for: Dashboard 2 — heatmap, box plot
# Rows: up to 10 tickers × 3 conflicts × 3 phases = 73 rows
# ─────────────────────────────────────────────────────────
print("Generating export 3: Ticker-phase breakdown...")

ticker_phase = df.groupby(["ticker", "region", "conflict_period", "period_phase"]).agg(
    avg_close=("close", "mean"),
    avg_daily_return=("daily_return", "mean"),
    volatility=("daily_return", "std"),
    trading_days=("close", "count")
).reset_index().round(4)
ticker_phase["phase_order"] = ticker_phase["period_phase"].map(PHASE_ORDER)
ticker_phase = ticker_phase.sort_values(["conflict_period", "ticker", "phase_order"])

ticker_phase.to_csv("data/tableau/03_ticker_phase_breakdown.csv", index=False)
print(f"  Saved: 03_ticker_phase_breakdown.csv ({len(ticker_phase)} rows)")

# ─────────────────────────────────────────────────────────
# EXPORT 4: US-Iran Full Timeline (all 10 tickers, daily)
# Used for: Dashboard 3 — line chart, annotated timeline
# ─────────────────────────────────────────────────────────
print("Generating export 4: US-Iran timeline...")

us_iran = df[df["conflict_period"] == "US-Iran"][[
    "date", "ticker", "region", "conflict_period", "period_phase",
    "open", "high", "low", "close", "volume", "daily_return"
]].copy().sort_values(["ticker", "date"])

us_iran.to_csv("data/tableau/04_us_iran_timeline.csv", index=False)
print(f"  Saved: 04_us_iran_timeline.csv ({len(us_iran)} rows)")

# ─────────────────────────────────────────────────────────
# EXPORT 5: Oil Price History (Gulf Wars — GW1 + GW2)
# Used for: Dashboard 1 + 2 — oil price line charts
# ─────────────────────────────────────────────────────────
print("Generating export 5: Oil price history (wars)...")

oil_wars = oil[oil["conflict_period"].isin(["First Gulf War", "Second Gulf War"])].copy()
oil_wars["phase_order"] = oil_wars["period_phase"].map(PHASE_ORDER)
oil_wars = oil_wars.sort_values(["conflict_period", "date"])

oil_wars.to_csv("data/tableau/05_oil_prices_wars.csv", index=False)
print(f"  Saved: 05_oil_prices_wars.csv ({len(oil_wars)} rows)")

# ─────────────────────────────────────────────────────────
# EXPORT 6: Conflict Events (for annotation markers in Tableau)
# ─────────────────────────────────────────────────────────
print("Generating export 6: Conflict events...")
import shutil
shutil.copy("data/cleaned/conflict_events_cleaned.csv", "data/tableau/06_conflict_events.csv")
print("  Saved: 06_conflict_events.csv")

print()
print("=" * 50)
print("All Tableau exports complete!")
print("Files saved to: data/tableau/")
print()
print("Summary:")
print("  01_gw1_vs_gw2_comparison.csv  — Dashboard 1: GW1 vs GW2 predictability")
print("  02_phase_summary.csv          — Dashboard 2: Before/During/After patterns")
print("  03_ticker_phase_breakdown.csv — Dashboard 2: Heatmap & box plots")
print("  04_us_iran_timeline.csv       — Dashboard 3: Current US-Iran tensions")
print("  05_oil_prices_wars.csv        — Dashboard 1 & 2: Oil price line charts")
print("  06_conflict_events.csv        — All dashboards: Event annotation markers")