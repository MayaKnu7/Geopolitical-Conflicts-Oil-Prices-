# 05_live_predictions.py
# Geopolitical Conflict and Oil Stock Performance
# Group 26 - Sana Shah, Maya Knutsvig, Illina Islam

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pickle
import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# ── Configuration ──────────────────────────────────────────
TICKERS = {
    "Canada":        ["SU", "CNQ", "CVE"],
    "US":            ["XOM", "CVX", "COP"],
    "International": ["SHEL", "TTE", "BP", "2222.SR"]
}

ALL_TICKERS = [t for region in TICKERS.values() for t in region]

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

# ── Load Trained Models ─────────────────────────────────────
def load_models():
    print("Loading trained models...")
    try:
        with open("models/logistic_regression_iran.pkl", "rb") as f:
            lr_model = pickle.load(f)
        with open("models/random_forest_iran.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("models/scaler_iran.pkl", "rb") as f:
            scaler = pickle.load(f)
        print("  Models loaded successfully")
        return lr_model, rf_model, scaler
    except Exception as e:
        print(f"  Error loading models: {e}")
        return None, None, None

# ── Fetch Live Stock Data ───────────────────────────────────
def fetch_live_stock_data():
    print("\nFetching live stock data...")

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=365)

    all_data = []
    for region, tickers in TICKERS.items():
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start = start_date.strftime("%Y-%m-%d"),
                    end   = end_date.strftime("%Y-%m-%d"),
                    auto_adjust = True,
                    progress    = False
                )
                if df.empty:
                    print(f"  No data for {ticker}")
                    continue

                df = df.reset_index()
                df.columns = [c[0] if isinstance(c, tuple) else c
                              for c in df.columns]
                df = df.rename(columns={"Date": "date", "Close": "close",
                                        "Open": "open", "High": "high",
                                        "Low": "low", "Volume": "volume"})
                df["ticker"] = ticker
                df["region"] = region
                df["date"]   = pd.to_datetime(df["date"])
                all_data.append(df)
                print(f"  Fetched {ticker}: {len(df)} rows")

            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"])
    return combined

# ── Fetch Live Oil Price ────────────────────────────────────
def fetch_live_oil_price():
    print("\nFetching live oil price...")
    try:
        oil = yf.download("CL=F", period="5d",
                          auto_adjust=True, progress=False)
        if oil.empty:
            return None, None
        latest_price   = float(oil["Close"].iloc[-1].iloc[0])
        prev_price     = float(oil["Close"].iloc[-2].iloc[0])
        oil_pct_change = ((latest_price - prev_price) / prev_price) * 100
        print(f"  WTI Oil Price: ${latest_price:.2f} "
              f"({oil_pct_change:+.2f}%)")
        return latest_price, oil_pct_change
    except Exception as e:
        print(f"  Error fetching oil price: {e}")
        return None, None

# ── Engineer Live Features ──────────────────────────────────
def engineer_live_features(stock_df, oil_pct_change):
    print("\nEngineering live features...")

    stock_df = stock_df.sort_values(["ticker", "date"])

    # Daily return
    stock_df["daily_return"] = stock_df.groupby(
        "ticker")["close"].pct_change() * 100

    # Volatility
    stock_df["volatility_10d"] = stock_df.groupby(
        "ticker")["daily_return"].transform(
        lambda x: x.rolling(10, min_periods=1).std())

    stock_df["volatility_30d"] = stock_df.groupby(
        "ticker")["daily_return"].transform(
        lambda x: x.rolling(30, min_periods=1).std())

    # Moving averages of returns
    stock_df["moving_avg_return_20"] = stock_df.groupby(
        "ticker")["daily_return"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())

    stock_df["moving_avg_return_50"] = stock_df.groupby(
        "ticker")["daily_return"].transform(
        lambda x: x.rolling(50, min_periods=1).mean())

    # RSI
    def compute_rsi(series, window=14):
        delta    = series.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    stock_df["rsi"] = stock_df.groupby(
        "ticker")["close"].transform(compute_rsi)

    # Oil price change
    stock_df["oil_pct_change"] = oil_pct_change if oil_pct_change else 0

    # Days since Soleimani assassination (key US-Iran event)
    soleimani_date = pd.Timestamp("2020-01-03")
    stock_df["days_since_event"] = (
        stock_df["date"] - soleimani_date).dt.days.clip(lower=0)

    return stock_df

# ── Generate Predictions ────────────────────────────────────
def generate_predictions(stock_df, lr_model, rf_model, scaler):
    print("\nGenerating predictions...")

    feature_cols = [
        "volatility_10d",
        "volatility_30d",
        "moving_avg_return_20",
        "moving_avg_return_50",
        "rsi",
        "oil_pct_change",
        "days_since_event"
    ]

    # Get latest row per ticker
    latest = stock_df.sort_values("date").groupby("ticker").last().reset_index()
    latest = latest.dropna(subset=feature_cols)

    if latest.empty:
        print("  No data available for predictions")
        return pd.DataFrame()

    X = scaler.transform(latest[feature_cols])

    latest["lr_prediction"]    = lr_model.predict(X)
    latest["rf_prediction"]    = rf_model.predict(X)
    latest["lr_probability"]   = lr_model.predict_proba(X)[:, 1]
    latest["rf_probability"]   = rf_model.predict_proba(X)[:, 1]
    latest["lr_signal"]        = latest["lr_prediction"].map(
        {1: "UP", 0: "DOWN"})
    latest["rf_signal"]        = latest["rf_prediction"].map(
        {1: "UP", 0: "DOWN"})
    latest["agreement"]        = (
        latest["lr_prediction"] == latest["rf_prediction"])
    latest["consensus_signal"] = latest.apply(
        lambda row: row["lr_signal"] if row["agreement"] else "MIXED",
        axis=1
    )

    return latest

# ── Save Predictions to DB ──────────────────────────────────
def save_predictions_to_db(predictions_df, oil_price, oil_pct_change):
    print("\nSaving predictions to database...")
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_predictions (
                id              SERIAL PRIMARY KEY,
                prediction_date DATE,
                ticker          VARCHAR(10),
                region          VARCHAR(20),
                latest_close    FLOAT,
                oil_price       FLOAT,
                oil_pct_change  FLOAT,
                lr_signal       VARCHAR(20),
                rf_signal       VARCHAR(20),
                lr_probability  FLOAT,
                rf_probability  FLOAT,
                consensus       VARCHAR(20)
            )
        """)

        for _, row in predictions_df.iterrows():
            cursor.execute("""
                INSERT INTO live_predictions
                (prediction_date, ticker, region, latest_close,
                 oil_price, oil_pct_change, lr_signal, rf_signal,
                 lr_probability, rf_probability, consensus)
                VALUES (CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row["ticker"], row["region"], row.get("close"),
                oil_price, oil_pct_change,
                row["lr_signal"], row["rf_signal"],
                row["lr_probability"], row["rf_probability"],
                row["consensus_signal"]
            ))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"  Saved {len(predictions_df)} predictions to database")

    except Exception as e:
        print(f"  Error saving predictions: {e}")

# ── Print Prediction Report ─────────────────────────────────
def print_prediction_report(predictions_df, oil_price, oil_pct_change):
    print("\n" + "="*60)
    print("LIVE PREDICTIONS - CURRENT US-IRAN TENSIONS")
    print(f"Generated: {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    print(f"\nCurrent WTI Oil Price: ${oil_price:.2f} "
          f"({oil_pct_change:+.2f}%)")

    print("\nDISCLAIMER: For academic purposes only.")
    print("Not financial advice.\n")

    for region in ["Canada", "US", "International"]:
        region_data = predictions_df[
            predictions_df["region"] == region]
        if region_data.empty:
            continue

        print(f"\n-- {region} --")
        print(f"{'Ticker':<10} {'Close':>8} {'LR Signal':<12} "
              f"{'RF Signal':<12} {'Consensus':<12} {'Confidence':>10}")
        print("-" * 65)

        for _, row in region_data.iterrows():
            confidence = max(row["lr_probability"],
                           row["rf_probability"]) * 100
            print(f"{row['ticker']:<10} "
                  f"${row.get('close', 0):>7.2f} "
                  f"{row['lr_signal']:<12} "
                  f"{row['rf_signal']:<12} "
                  f"{row['consensus_signal']:<12} "
                  f"{confidence:>9.1f}%")

    up_count   = (predictions_df["consensus_signal"] == "UP").sum()
    down_count = (predictions_df["consensus_signal"] == "DOWN").sum()
    mixed      = (predictions_df["consensus_signal"] == "MIXED").sum()

    print(f"\n-- Summary --")
    print(f"  Bullish signals: {up_count}")
    print(f"  Bearish signals: {down_count}")
    print(f"  Mixed signals:   {mixed}")
    print("="*60)

    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/live_predictions_{datetime.today().strftime('%Y%m%d')}.txt"
    with open(report_path, "w") as f:
        f.write(f"Live Predictions - {datetime.today().strftime('%Y-%m-%d')}\n")
        f.write(predictions_df[[
            "ticker", "region", "close",
            "lr_signal", "rf_signal",
            "consensus_signal", "lr_probability", "rf_probability"
        ]].to_string())
    print(f"\n  Report saved: {report_path}")

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Starting live predictions...")
    print("=" * 50)

    lr_model, rf_model, scaler = load_models()
    if lr_model is None:
        print("Could not load models. Run 04_modeling.py first.")
        exit(1)

    stock_df                  = fetch_live_stock_data()
    oil_price, oil_pct_change = fetch_live_oil_price()

    if stock_df.empty:
        print("Could not fetch live stock data.")
        exit(1)

    if oil_pct_change is None:
        oil_pct_change = 0.0
        oil_price      = 0.0

    stock_df = engineer_live_features(stock_df, oil_pct_change)

    predictions_df = generate_predictions(
        stock_df, lr_model, rf_model, scaler)

    if not predictions_df.empty:
        save_predictions_to_db(
            predictions_df, oil_price, oil_pct_change)
        print_prediction_report(
            predictions_df, oil_price, oil_pct_change)

    print("\n" + "=" * 50)
    print("Live predictions complete!")
    print("=" * 50)