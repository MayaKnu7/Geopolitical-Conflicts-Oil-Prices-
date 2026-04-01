# 03_eda.py
# Geopolitical Conflict and Oil Stock Performance
# Group 26 - Sana Shah, Maya Knutsvig, Illina Islam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────
os.makedirs("reports/eda", exist_ok=True)

# ── Database Connection ─────────────────────────────────────
def get_db_connection():
    return psycopg2.connect(
        host        = os.getenv("DB_HOST"),
        dbname      = os.getenv("DB_NAME"),
        user        = os.getenv("DB_USER"),
        password    = os.getenv("DB_PASSWORD"),
        sslmode     = "verify-full",
        sslrootcert = "./global-bundle.pem"
    )

# ── Helper: Load from DB ────────────────────────────────────
def load_data():
    print("Loading data from database...")
    conn = get_db_connection()

    stock_df  = pd.read_sql("SELECT * FROM stock_data", conn)
    oil_df    = pd.read_sql("SELECT * FROM oil_prices", conn)
    events_df = pd.read_sql("SELECT * FROM conflict_events", conn)

    conn.close()

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    oil_df["date"]   = pd.to_datetime(oil_df["date"])

    print(f"  Stock rows: {len(stock_df)}")
    print(f"  Oil rows:   {len(oil_df)}")
    print(f"  Events:     {len(events_df)}")

    return stock_df, oil_df, events_df

# ── EDA 1: Summary Statistics ───────────────────────────────
def summary_statistics(stock_df, oil_df):
    print("\n── Summary Statistics ──")

    print("\nStock data summary:")
    print(stock_df[["open", "high", "low", "close", "volume"]].describe().round(2))

    print("\nOil price summary:")
    print(oil_df[["wti_price", "daily_change", "pct_change"]].describe().round(2))

    print("\nMissing values in stock data:")
    print(stock_df.isnull().sum())

    print("\nMissing values in oil data:")
    print(oil_df.isnull().sum())

    print("\nStock data by conflict period:")
    print(stock_df.groupby("conflict_period")["close"].describe().round(2))

    print("\nStock data by region:")
    print(stock_df.groupby("region")["close"].describe().round(2))

# ── EDA 2: Distribution Plots ───────────────────────────────
def distribution_plots(stock_df, oil_df):
    print("\n── Distribution Plots ──")

    # Distribution of closing prices by region
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    regions = stock_df["region"].unique()
    for i, region in enumerate(regions):
        data = stock_df[stock_df["region"] == region]["close"].dropna()
        axes[i].hist(data, bins=50, color="steelblue", edgecolor="black")
        axes[i].set_title(f"Close Price Distribution - {region}")
        axes[i].set_xlabel("Close Price")
        axes[i].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("reports/eda/01_close_price_distribution.png")
    plt.close()
    print("  Saved: 01_close_price_distribution.png")

    # Oil price distribution
    plt.figure(figsize=(10, 5))
    plt.hist(oil_df["wti_price"].dropna(), bins=50, color="coral", edgecolor="black")
    plt.title("WTI Oil Price Distribution")
    plt.xlabel("Price (USD/barrel)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("reports/eda/02_oil_price_distribution.png")
    plt.close()
    print("  Saved: 02_oil_price_distribution.png")

    # Closing price distribution by conflict period
    plt.figure(figsize=(12, 5))
    for period in stock_df["conflict_period"].unique():
        data = stock_df[stock_df["conflict_period"] == period]["close"].dropna()
        plt.hist(data, bins=50, alpha=0.5, label=period)
    plt.title("Close Price Distribution by Conflict Period")
    plt.xlabel("Close Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/eda/03_close_by_conflict_period.png")
    plt.close()
    print("  Saved: 03_close_by_conflict_period.png")

# ── EDA 3: Time Series Plots ────────────────────────────────
def time_series_plots(stock_df, oil_df, events_df):
    print("\n── Time Series Plots ──")

    # Oil price over time with conflict events marked
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(oil_df["date"], oil_df["wti_price"], color="black", linewidth=1)

    colors = {"High": "red", "Medium": "orange", "Low": "yellow"}
    for _, event in events_df.iterrows():
        color = colors.get(event["severity"], "gray")
        ax.axvline(x=event["date"], color=color, alpha=0.6, linewidth=1.5,
                   linestyle="--")

    ax.set_title("WTI Oil Price Over Time with Conflict Events")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/barrel)")
    plt.tight_layout()
    plt.savefig("reports/eda/04_oil_price_timeline.png")
    plt.close()
    print("  Saved: 04_oil_price_timeline.png")

    # Stock price over time per ticker for US-Iran period
    us_iran = stock_df[stock_df["conflict_period"] == "US-Iran"]
    tickers = us_iran["ticker"].unique()

    fig, ax = plt.subplots(figsize=(16, 8))
    for ticker in tickers:
        data = us_iran[us_iran["ticker"] == ticker].sort_values("date")
        ax.plot(data["date"], data["close"], label=ticker, linewidth=1)

    for _, event in events_df[events_df["conflict_period"] == "US-Iran"].iterrows():
        ax.axvline(x=event["date"], color="red", alpha=0.4, linewidth=1, linestyle="--")

    ax.set_title("Oil & Gas Stock Prices During US-Iran Tensions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/eda/05_stock_prices_us_iran.png")
    plt.close()
    print("  Saved: 05_stock_prices_us_iran.png")

    # Stock price during First Gulf War
    gulf1 = stock_df[stock_df["conflict_period"] == "First Gulf War"]
    tickers = gulf1["ticker"].unique()

    fig, ax = plt.subplots(figsize=(16, 8))
    for ticker in tickers:
        data = gulf1[gulf1["ticker"] == ticker].sort_values("date")
        ax.plot(data["date"], data["close"], label=ticker, linewidth=1)

    for _, event in events_df[events_df["conflict_period"] == "First Gulf War"].iterrows():
        ax.axvline(x=event["date"], color="red", alpha=0.4, linewidth=1, linestyle="--")

    ax.set_title("Oil & Gas Stock Prices During First Gulf War")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/eda/06_stock_prices_gulf_war_1.png")
    plt.close()
    print("  Saved: 06_stock_prices_gulf_war_1.png")

# ── EDA 4: Relationship Analysis ────────────────────────────
def relationship_analysis(stock_df, oil_df):
    print("\n── Relationship Analysis ──")

    # Average close price by period phase
    phase_avg = stock_df.groupby(
        ["conflict_period", "period_phase"])["close"].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    periods = stock_df["conflict_period"].unique()

    for i, period in enumerate(periods):
        data = phase_avg[phase_avg["conflict_period"] == period]
        order = ["Before", "During", "After"]
        data = data[data["period_phase"].isin(order)]
        data["period_phase"] = pd.Categorical(
            data["period_phase"], categories=order, ordered=True)
        data = data.sort_values("period_phase")
        axes[i].bar(data["period_phase"], data["close"],
                    color=["steelblue", "coral", "green"])
        axes[i].set_title(f"Avg Close Price\n{period}")
        axes[i].set_xlabel("Phase")
        axes[i].set_ylabel("Avg Close Price")

    plt.tight_layout()
    plt.savefig("reports/eda/07_avg_price_by_phase.png")
    plt.close()
    print("  Saved: 07_avg_price_by_phase.png")

    # Average close price by region and conflict period
    region_avg = stock_df.groupby(
        ["region", "conflict_period"])["close"].mean().unstack()

    region_avg.plot(kind="bar", figsize=(12, 6))
    plt.title("Average Close Price by Region and Conflict Period")
    plt.xlabel("Region")
    plt.ylabel("Average Close Price")
    plt.xticks(rotation=0)
    plt.legend(title="Conflict Period")
    plt.tight_layout()
    plt.savefig("reports/eda/08_avg_price_by_region.png")
    plt.close()
    print("  Saved: 08_avg_price_by_region.png")

    # Correlation heatmap for US-Iran period
    us_iran = stock_df[stock_df["conflict_period"] == "US-Iran"]
    pivot   = us_iran.pivot_table(
        index="date", columns="ticker", values="close")
    corr    = pivot.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5)
    plt.title("Stock Price Correlation During US-Iran Tensions")
    plt.tight_layout()
    plt.savefig("reports/eda/09_correlation_heatmap.png")
    plt.close()
    print("  Saved: 09_correlation_heatmap.png")

# ── EDA 5: Volatility Analysis ──────────────────────────────
def volatility_analysis(stock_df):
    print("\n── Volatility Analysis ──")

    # Calculate daily returns per ticker per period
    stock_df = stock_df.sort_values(["ticker", "conflict_period", "date"])
    stock_df["daily_return"] = stock_df.groupby(
        ["ticker", "conflict_period"])["close"].pct_change() * 100

    # Volatility by phase
    volatility = stock_df.groupby(
        ["conflict_period", "period_phase"])["daily_return"].std().reset_index()
    volatility.columns = ["conflict_period", "period_phase", "volatility"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    periods   = stock_df["conflict_period"].unique()

    for i, period in enumerate(periods):
        data  = volatility[volatility["conflict_period"] == period]
        order = ["Before", "During", "After"]
        data  = data[data["period_phase"].isin(order)]
        data["period_phase"] = pd.Categorical(
            data["period_phase"], categories=order, ordered=True)
        data = data.sort_values("period_phase")
        axes[i].bar(data["period_phase"], data["volatility"],
                    color=["steelblue", "coral", "green"])
        axes[i].set_title(f"Volatility by Phase\n{period}")
        axes[i].set_xlabel("Phase")
        axes[i].set_ylabel("Std Dev of Daily Returns (%)")

    plt.tight_layout()
    plt.savefig("reports/eda/10_volatility_by_phase.png")
    plt.close()
    print("  Saved: 10_volatility_by_phase.png")

    return stock_df

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Starting EDA...")
    print("=" * 50)

    stock_df, oil_df, events_df = load_data()

    summary_statistics(stock_df, oil_df)
    distribution_plots(stock_df, oil_df)
    time_series_plots(stock_df, oil_df, events_df)
    relationship_analysis(stock_df, oil_df)
    stock_df = volatility_analysis(stock_df)

    print("\n" + "=" * 50)
    print("EDA complete! Charts saved to reports/eda/")
    print("=" * 50)