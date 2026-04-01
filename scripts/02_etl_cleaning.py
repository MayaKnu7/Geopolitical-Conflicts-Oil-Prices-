# 02_etl_cleaning.py
# Geopolitical Conflict and Oil Stock Performance
# Group 26 - Sana Shah, Maya Knutsvig, Illina Islam

import pandas as pd
import numpy as np
import boto3
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────
S3_BUCKET = "geo-conflicts-oil-prices-data-319279229929-ca-west-1-an"

CONFLICT_PHASES = {
    "First Gulf War": {
        "before": ("1989-01-01", "1990-08-01"),
        "during": ("1990-08-02", "1991-02-28"),
        "after":  ("1991-03-01", "1992-12-31")
    },
    "Second Gulf War": {
        "before": ("2002-01-01", "2003-03-19"),
        "during": ("2003-03-20", "2003-05-01"),
        "after":  ("2003-05-02", "2004-12-31")
    },
    "US-Iran": {
        "before": ("2019-01-01", "2020-01-02"),
        "during": ("2020-01-03", "2020-01-19"),
        "after":  ("2020-01-20", "2026-01-01")
    }
}

REGION_MAP = {
    "SU":      "Canada",
    "CNQ":     "Canada",
    "CVE":     "Canada",
    "XOM":     "US",
    "CVX":     "US",
    "COP":     "US",
    "SHEL":    "International",
    "TTE":     "International",
    "BP":      "International",
    "2222.SR": "International"
}

# ── S3 Client ───────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    aws_access_key_id     = os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key = os.getenv("AWS_SECRET_KEY"),
    region_name           = "ca-west-1"
)

# ── Database Connection ─────────────────────────────────────
def get_db_connection():
    return psycopg2.connect(
        host     = os.getenv("DB_HOST"),
        dbname   = os.getenv("DB_NAME"),
        user     = os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD"),
        sslmode  = "verify-full",
        sslrootcert = "./global-bundle.pem"
    )

# ── Helper: Upload to S3 ────────────────────────────────────
def upload_to_s3(local_path, s3_key):
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"  Uploaded to S3: {s3_key}")

# ── Helper: Assign Period Phase ─────────────────────────────
def assign_phase(date, conflict_period):
    if conflict_period not in CONFLICT_PHASES:
        return "Unknown"
    phases = CONFLICT_PHASES[conflict_period]
    for phase, (start, end) in phases.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return phase.capitalize()
    return "Unknown"

# ── Step 1: Clean Stock Data ────────────────────────────────
def clean_stock_data():
    print("\n── Cleaning stock data ──")
    os.makedirs("data/cleaned/stocks", exist_ok=True)

    all_cleaned = []

    stock_dir = "data/raw/stocks"
    for filename in os.listdir(stock_dir):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(stock_dir, filename)
        print(f"\nProcessing: {filename}")

        try:
            df = pd.read_csv(filepath, header=[0,1])

            # Flatten multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

            # Reset and rename date column
            df = df.reset_index() if 'Date' not in df.columns else df
            df.columns = [c.strip() for c in df.columns]

            # Find date column
            date_col = None
            for col in df.columns:
                if 'date' in col.lower() or col == 'Price':
                    date_col = col
                    break

            if date_col is None:
                print(f"  Could not find date column in {filename}, skipping")
                continue

            df = df.rename(columns={date_col: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            # Drop rows with invalid dates
            df = df.dropna(subset=["date"])

            # Extract ticker and period from filename
            parts = filename.replace(".csv", "").split("_", 1)
            ticker = parts[0]
            conflict_period = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

            # Standardize column names
            col_map = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower == "open":
                    col_map[col] = "open"
                elif col_lower == "high":
                    col_map[col] = "high"
                elif col_lower == "low":
                    col_map[col] = "low"
                elif col_lower == "close":
                    col_map[col] = "close"
                elif col_lower in ["adj close", "adj_close"]:
                    col_map[col] = "adj_close"
                elif col_lower == "volume":
                    col_map[col] = "volume"
            df = df.rename(columns=col_map)

            # Keep only needed columns
            keep_cols = ["date"]
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col in df.columns:
                    keep_cols.append(col)
            df = df[keep_cols]

            # Add metadata
            df["ticker"]          = ticker
            df["company_name"]    = ticker
            df["region"]          = REGION_MAP.get(ticker, "Unknown")
            df["conflict_period"] = conflict_period

            # Assign period phase
            df["period_phase"] = df["date"].apply(
                lambda d: assign_phase(d, conflict_period)
            )

            # Convert numeric columns
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop duplicate dates
            df = df.drop_duplicates(subset=["date"])

            # Drop rows where close price is missing
            df = df.dropna(subset=["close"])

            # Sort by date
            df = df.sort_values("date").reset_index(drop=True)

            print(f"  Rows: {len(df)} | Date range: {df['date'].min().date()} to {df['date'].max().date()}")

            # Save cleaned file
            clean_filename = f"cleaned_{filename}"
            local_path     = f"data/cleaned/stocks/{clean_filename}"
            df.to_csv(local_path, index=False)
            upload_to_s3(local_path, f"cleaned/stocks/{clean_filename}")

            all_cleaned.append(df)

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    if all_cleaned:
        combined = pd.concat(all_cleaned, ignore_index=True)
        combined_path = "data/cleaned/all_stocks_cleaned.csv"
        combined.to_csv(combined_path, index=False)
        upload_to_s3(combined_path, "cleaned/all_stocks_cleaned.csv")
        print(f"\nCombined cleaned stock data: {len(combined)} total rows")
        return combined

    return pd.DataFrame()

# ── Step 2: Clean Oil Price Data ────────────────────────────
def clean_oil_data():
    print("\n── Cleaning oil price data ──")
    os.makedirs("data/cleaned/oil", exist_ok=True)

    try:
        df = pd.read_csv("data/raw/oil/wti_oil_prices.csv")
        df["date"]      = pd.to_datetime(df["date"], errors="coerce")
        df["wti_price"] = pd.to_numeric(df["wti_price"], errors="coerce")

        # Drop missing
        df = df.dropna()
        df = df.drop_duplicates(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Calculate daily change and pct change
        df["daily_change"] = df["wti_price"].diff()
        df["pct_change"]   = df["wti_price"].pct_change() * 100

        # Assign conflict period and phase
        def get_conflict_period(date):
            if pd.Timestamp("1989-01-01") <= date <= pd.Timestamp("1992-12-31"):
                return "First Gulf War"
            elif pd.Timestamp("2002-01-01") <= date <= pd.Timestamp("2004-12-31"):
                return "Second Gulf War"
            elif pd.Timestamp("2019-01-01") <= date <= pd.Timestamp("2026-01-01"):
                return "US-Iran"
            return "Other"

        df["conflict_period"] = df["date"].apply(get_conflict_period)
        df["period_phase"]    = df.apply(
            lambda row: assign_phase(row["date"], row["conflict_period"]), axis=1
        )

        local_path = "data/cleaned/oil/wti_oil_prices_cleaned.csv"
        df.to_csv(local_path, index=False)
        upload_to_s3(local_path, "cleaned/oil/wti_oil_prices_cleaned.csv")

        print(f"  Rows: {len(df)} | Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        return df

    except Exception as e:
        print(f"  Error cleaning oil data: {e}")
        return pd.DataFrame()

# ── Step 3: Clean Conflict Events ───────────────────────────
def clean_conflict_events():
    print("\n── Cleaning conflict events ──")

    try:
        df = pd.read_csv("data/raw/conflict_events.csv")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        local_path = "data/cleaned/conflict_events_cleaned.csv"
        df.to_csv(local_path, index=False)
        upload_to_s3(local_path, "cleaned/conflict_events_cleaned.csv")

        print(f"  Conflict events cleaned: {len(df)} events")
        return df

    except Exception as e:
        print(f"  Error cleaning conflict events: {e}")
        return pd.DataFrame()

# ── Step 4: Load into PostgreSQL ────────────────────────────
def load_to_database(stock_df, oil_df, events_df):
    print("\n── Loading data into PostgreSQL ──")

    try:
        conn   = get_db_connection()
        cursor = conn.cursor()

        # Load stock data
        print("  Loading stock data...")
        for _, row in stock_df.iterrows():
            cursor.execute("""
                INSERT INTO stock_data 
                (date, ticker, company_name, region, open, high, low, close, adj_close, volume, conflict_period, period_phase)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                row["date"], row["ticker"], row["company_name"],
                row["region"],
                row.get("open"), row.get("high"), row.get("low"),
                row.get("close"), row.get("adj_close"), row.get("volume"),
                row["conflict_period"], row["period_phase"]
            ))
        print(f"  Loaded {len(stock_df)} stock rows")

        # Load oil data
        print("  Loading oil price data...")
        for _, row in oil_df.iterrows():
            cursor.execute("""
                INSERT INTO oil_prices 
                (date, wti_price, daily_change, pct_change, conflict_period, period_phase)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                row["date"], row["wti_price"],
                row.get("daily_change"), row.get("pct_change"),
                row["conflict_period"], row["period_phase"]
            ))
        print(f"  Loaded {len(oil_df)} oil price rows")

        # Load conflict events
        print("  Loading conflict events...")
        for _, row in events_df.iterrows():
            cursor.execute("""
                INSERT INTO conflict_events 
                (date, conflict_period, event_description, severity)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                row["date"], row["conflict_period"],
                row["event_description"], row["severity"]
            ))
        print(f"  Loaded {len(events_df)} conflict events")

        conn.commit()
        cursor.close()
        conn.close()
        print("  Database load complete!")

    except Exception as e:
        print(f"  Database error: {e}")

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Starting ETL cleaning...")
    print("=" * 50)

    stock_df  = clean_stock_data()
    oil_df    = clean_oil_data()
    events_df = clean_conflict_events()

    if not stock_df.empty and not oil_df.empty and not events_df.empty:
        load_to_database(stock_df, oil_df, events_df)
    else:
        print("\nWarning: some dataframes are empty, skipping database load")

    print("\n" + "=" * 50)
    print("ETL cleaning complete!")
    print("=" * 50)