# 01_data_acquisition.py
# Geopolitical Conflict and Oil Stock Performance
# Group 26 - Sana Shah, Maya Knutsvig, Illina Islam

import yfinance as yf
import pandas as pd
import requests
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────
TICKERS = {
    "Canada":        ["SU", "CNQ", "CVE"],
    "US":            ["XOM", "CVX", "COP"],
    "International": ["SHEL", "TTE", "BP", "2222.SR"]
}

CONFLICT_PERIODS = {
    "First Gulf War":  ("1989-01-01", "1992-12-31"),
    "Second Gulf War": ("2002-01-01", "2004-12-31"),
    "US-Iran":         ("2019-01-01", "2026-01-01")
}

# Tickers excluded from Gulf War periods (didn't exist yet)
GULF_WAR_EXCLUDE = ["CVE", "2222.SR"]

S3_BUCKET = "geo-conflicts-oil-prices-data-319279229929-ca-west-1-an"

# ── S3 Client ───────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    aws_access_key_id     = os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key = os.getenv("AWS_SECRET_KEY"),
    region_name           = "ca-west-1"
)

# ── Helper: Upload to S3 ────────────────────────────────────
def upload_to_s3(local_path, s3_key):
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"Uploaded to S3: {s3_key}")

# ── Step 1: Download Stock Data ─────────────────────────────
def download_stock_data():
    os.makedirs("data/raw/stocks", exist_ok=True)

    for region, tickers in TICKERS.items():
        for ticker in tickers:
            print(f"\nDownloading {ticker} ({region})...")

            for period_name, (start, end) in CONFLICT_PERIODS.items():

                # Skip CVE and 2222.SR for Gulf War periods
                if ticker in GULF_WAR_EXCLUDE and period_name != "US-Iran":
                    print(f"  Skipping {ticker} for {period_name} — insufficient historical data")
                    continue

                try:
                    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

                    if df.empty:
                        print(f"  No data found for {ticker} in {period_name}")
                        continue

                    # Add metadata columns
                    df["ticker"]          = ticker
                    df["region"]          = region
                    df["conflict_period"] = period_name

                    # Save locally
                    filename   = f"{ticker}_{period_name.replace(' ', '_')}.csv"
                    local_path = f"data/raw/stocks/{filename}"
                    df.to_csv(local_path)
                    print(f"  Saved: {local_path}")

                    # Upload to S3
                    upload_to_s3(local_path, f"raw/stocks/{filename}")

                except Exception as e:
                    print(f"  Error downloading {ticker} for {period_name}: {e}")

# ── Step 2: Download Oil Price Data (EIA) ───────────────────
def download_oil_data():
    os.makedirs("data/raw/oil", exist_ok=True)

    print("\nDownloading WTI oil prices from EIA...")

    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key=DEMO_KEY&frequency=daily&data[0]=value&facets[series][]=RWTC&start=1989-01-01&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"

    try:
        response = requests.get(url)
        data     = response.json()
        records  = data["response"]["data"]
        df       = pd.DataFrame(records)
        df       = df.rename(columns={"period": "date", "value": "wti_price"})
        df       = df[["date", "wti_price"]]
        df["date"] = pd.to_datetime(df["date"])
        df["wti_price"] = pd.to_numeric(df["wti_price"], errors="coerce")
        df = df.sort_values("date")

        local_path = "data/raw/oil/wti_oil_prices.csv"
        df.to_csv(local_path, index=False)
        print(f"Saved: {local_path}")

        upload_to_s3(local_path, "raw/oil/wti_oil_prices.csv")

    except Exception as e:
        print(f"Error downloading oil data: {e}")

# ── Step 3: Create Conflict Events Table ────────────────────
def create_conflict_events():
    os.makedirs("data/raw", exist_ok=True)

    events = [
        # First Gulf War
        {"date": "1990-08-02", "conflict_period": "First Gulf War",  "event_description": "Iraq invades Kuwait",                          "severity": "High"},
        {"date": "1990-08-06", "conflict_period": "First Gulf War",  "event_description": "UN imposes sanctions on Iraq",                 "severity": "Medium"},
        {"date": "1990-11-29", "conflict_period": "First Gulf War",  "event_description": "UN authorizes force against Iraq",             "severity": "High"},
        {"date": "1991-01-17", "conflict_period": "First Gulf War",  "event_description": "Operation Desert Storm begins",               "severity": "High"},
        {"date": "1991-02-28", "conflict_period": "First Gulf War",  "event_description": "Ceasefire declared",                          "severity": "High"},
        # Second Gulf War
        {"date": "2002-11-08", "conflict_period": "Second Gulf War", "event_description": "UN resolution demands Iraq disarmament",      "severity": "Medium"},
        {"date": "2003-03-20", "conflict_period": "Second Gulf War", "event_description": "US invasion of Iraq begins",                  "severity": "High"},
        {"date": "2003-04-09", "conflict_period": "Second Gulf War", "event_description": "Baghdad falls",                              "severity": "High"},
        {"date": "2003-05-01", "conflict_period": "Second Gulf War", "event_description": "Bush declares end of major combat operations", "severity": "Medium"},
        # US-Iran
        {"date": "2019-05-08", "conflict_period": "US-Iran",         "event_description": "US withdraws from Iran nuclear deal",         "severity": "High"},
        {"date": "2019-06-20", "conflict_period": "US-Iran",         "event_description": "Iran shoots down US drone",                   "severity": "High"},
        {"date": "2020-01-03", "conflict_period": "US-Iran",         "event_description": "Soleimani assassination",                     "severity": "High"},
        {"date": "2020-01-08", "conflict_period": "US-Iran",         "event_description": "Iran retaliatory missile strikes on US bases", "severity": "High"},
        {"date": "2020-01-10", "conflict_period": "US-Iran",         "event_description": "US imposes new sanctions on Iran",            "severity": "Medium"},
        {"date": "2024-04-01", "conflict_period": "US-Iran",         "event_description": "Israel strikes Iranian consulate in Syria",   "severity": "High"},
        {"date": "2025-06-22", "conflict_period": "US-Iran",         "event_description": "US strikes Iranian nuclear facilities",       "severity": "High"},
    ]

    df         = pd.DataFrame(events)
    df["date"] = pd.to_datetime(df["date"])
    local_path = "data/raw/conflict_events.csv"
    df.to_csv(local_path, index=False)
    print(f"\nSaved conflict events: {local_path}")

    upload_to_s3(local_path, "raw/conflict_events.csv")

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Starting data acquisition...")
    print("=" * 50)

    download_stock_data()
    download_oil_data()
    create_conflict_events()

    print("\n" + "=" * 50)
    print("Data acquisition complete!")
    print("=" * 50)