# fix_oil_data.py
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host        = os.getenv("DB_HOST"),
        dbname      = os.getenv("DB_NAME"),
        user        = os.getenv("DB_USER"),
        password    = os.getenv("DB_PASSWORD"),
        sslmode     = "verify-full",
        sslrootcert = "./global-bundle.pem"
    )

def reload_oil_data():
    df = pd.read_csv("data/raw/oil/wti_oil_prices.csv")
    df["date"]      = pd.to_datetime(df["date"])
    df["wti_price"] = pd.to_numeric(df["wti_price"], errors="coerce")
    df = df.dropna()
    df = df.drop_duplicates(subset=["date"])
    df["daily_change"] = df["wti_price"].diff()
    df["pct_change"]   = df["wti_price"].pct_change() * 100

    def get_conflict_period(date):
        if pd.Timestamp("1989-01-01") <= date <= pd.Timestamp("1992-12-31"):
            return "First Gulf War"
        elif pd.Timestamp("2002-01-01") <= date <= pd.Timestamp("2004-12-31"):
            return "Second Gulf War"
        elif pd.Timestamp("2019-01-01") <= date <= pd.Timestamp("2026-01-01"):
            return "US-Iran"
        return "Other"

    def assign_phase(date, conflict_period):
        phases = {
            "First Gulf War":  {"before": ("1989-01-01","1990-08-01"), "during": ("1990-08-02","1991-02-28"), "after": ("1991-03-01","1992-12-31")},
            "Second Gulf War": {"before": ("2002-01-01","2003-03-19"), "during": ("2003-03-20","2003-05-01"), "after": ("2003-05-02","2004-12-31")},
            "US-Iran":         {"before": ("2019-01-01","2020-01-02"), "during": ("2020-01-03","2020-01-19"), "after": ("2020-01-20","2026-01-01")}
        }
        if conflict_period not in phases:
            return "Unknown"
        for phase, (start, end) in phases[conflict_period].items():
            if pd.Timestamp(start) <= date <= pd.Timestamp(end):
                return phase.capitalize()
        return "Unknown"

    df["conflict_period"] = df["date"].apply(get_conflict_period)
    df["period_phase"]    = df.apply(lambda row: assign_phase(row["date"], row["conflict_period"]), axis=1)

    conn   = get_db_connection()
    cursor = conn.cursor()

    # Clear existing oil data
    cursor.execute("DELETE FROM oil_prices")
    print("Cleared existing oil prices")

    # Insert new data
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO oil_prices
            (date, wti_price, daily_change, pct_change, conflict_period, period_phase)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (row["date"], row["wti_price"], row.get("daily_change"),
              row.get("pct_change"), row["conflict_period"], row["period_phase"]))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Loaded {len(df)} oil price rows into database")

if __name__ == "__main__":
    reload_oil_data()