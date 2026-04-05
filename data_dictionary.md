# Data Dictionary
## Geopolitical Conflict and Oil Stock Performance
### COSC 301 — Group 26
Sana Shah, Maya Knutsvig, Illina Islam

---

## Data Sources

| Source | Description | License |
|--------|-------------|---------|
| yfinance (Yahoo Finance) | Historical and live stock price data for 10 oil & gas companies | Educational/personal use |
| EIA (U.S. Energy Information Administration) | Daily WTI crude oil spot prices | Public domain |
| GDELT Project | Global event database used to identify key conflict dates | Open/free for research |

---

## Tickers

| Ticker | Company | Region | Gulf War 1 | Gulf War 2 | US-Iran |
|--------|---------|--------|-----------|-----------|---------|
| SU | Suncor Energy | Canada | Yes | Yes | Yes |
| CNQ | Canadian Natural Resources | Canada | No — insufficient data | Yes | Yes |
| CVE | Cenovus Energy | Canada | No — founded 2009 | No — founded 2009 | Yes |
| XOM | ExxonMobil | US | Yes | Yes | Yes |
| CVX | Chevron | US | Yes | Yes | Yes |
| COP | ConocoPhillips | US | Yes | Yes | Yes |
| SHEL | Shell | International | Yes | Yes | Yes |
| TTE | TotalEnergies | International | Partial — from Oct 1991 | Yes | Yes |
| BP | BP | International | Yes | Yes | Yes |
| 2222.SR | Saudi Aramco | International | No — IPO 2019 | No — IPO 2019 | Yes |

---

## Conflict Periods

| Period | Date Range | Phase: Before | Phase: During | Phase: After |
|--------|-----------|---------------|---------------|--------------|
| First Gulf War | 1989-01-01 to 1992-12-31 | 1989-01-01 to 1990-08-01 | 1990-08-02 to 1991-02-28 | 1991-03-01 to 1992-12-31 |
| Second Gulf War | 2002-01-01 to 2004-12-31 | 2002-01-01 to 2003-03-19 | 2003-03-20 to 2003-05-01 | 2003-05-02 to 2004-12-31 |
| US-Iran | 2019-01-01 to 2026-01-01 | 2019-01-01 to 2020-01-02 | 2020-01-03 to 2020-01-19 | 2020-01-20 to 2026-01-01 |

---

## Database Tables

### stock_data
Raw and cleaned daily stock price data for all 10 companies.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| id | SERIAL | Unique row identifier | 1 |
| date | DATE | Trading date | 2020-01-03 |
| ticker | VARCHAR(10) | Stock ticker symbol | XOM |
| company_name | VARCHAR(50) | Full company name | ExxonMobil |
| region | VARCHAR(20) | Geographic region | US |
| open | FLOAT | Opening price (USD) | 70.21 |
| high | FLOAT | Daily high price (USD) | 71.45 |
| low | FLOAT | Daily low price (USD) | 69.87 |
| close | FLOAT | Closing price (USD) | 70.98 |
| adj_close | FLOAT | Adjusted closing price — accounts for splits and dividends | 70.98 |
| volume | BIGINT | Number of shares traded | 12500000 |
| conflict_period | VARCHAR(30) | Which conflict period this row belongs to | US-Iran |
| period_phase | VARCHAR(10) | Phase within conflict period | During |

---

### oil_prices
Daily WTI crude oil spot prices from the EIA.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| id | SERIAL | Unique row identifier | 1 |
| date | DATE | Trading date | 2020-01-03 |
| wti_price | FLOAT | WTI crude oil price in USD per barrel | 63.05 |
| daily_change | FLOAT | Raw price change from previous day (USD) | 1.25 |
| pct_change | FLOAT | Percentage price change from previous day | 2.02 |
| conflict_period | VARCHAR(30) | Which conflict period this row belongs to | US-Iran |
| period_phase | VARCHAR(10) | Phase within conflict period | During |

---

### conflict_events
Manually curated table of key geopolitical events sourced from Reuters and BBC news archives.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| id | SERIAL | Unique row identifier | 1 |
| date | DATE | Date of the event | 2020-01-03 |
| conflict_period | VARCHAR(30) | Which conflict period this event belongs to | US-Iran |
| event_description | VARCHAR(255) | Brief description of the event | Soleimani assassination |
| severity | VARCHAR(10) | Assessed severity of the event | High |

**Severity definitions:**
- **High** — Direct military action (invasions, airstrikes, assassinations)
- **Medium** — Major political escalations, sanctions, diplomatic breakdowns
- **Low** — Minor incidents, warnings, travel advisories

---

### features
Engineered features calculated from stock and oil data for use in modeling.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| id | SERIAL | Unique row identifier | 1 |
| date | DATE | Trading date | 2020-01-03 |
| ticker | VARCHAR(10) | Stock ticker symbol | XOM |
| daily_return | FLOAT | Percentage price change from previous day | 1.23 |
| volatility_10d | FLOAT | 10-day rolling standard deviation of daily returns | 2.45 |
| volatility_30d | FLOAT | 30-day rolling standard deviation of daily returns | 1.98 |
| moving_avg_20 | FLOAT | 20-day moving average of closing price | 69.50 |
| moving_avg_50 | FLOAT | 50-day moving average of closing price | 68.20 |
| moving_avg_200 | FLOAT | 200-day moving average of closing price | 65.10 |
| rsi | FLOAT | Relative Strength Index — measures momentum (0-100) | 58.3 |
| oil_price | FLOAT | WTI oil price on that date | 63.05 |
| oil_pct_change | FLOAT | WTI oil price percentage change on that date | 2.02 |
| days_since_event | INT | Days elapsed since the start of the conflict | 15 |
| target | INT | Model label — 1 if price went up, 0 if down | 1 |

---

### model_results
Stores performance metrics for each model training run.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| id | SERIAL | Unique row identifier | 1 |
| run_date | DATE | Date the model was run | 2026-03-31 |
| model_name | VARCHAR(50) | Name of the model | Random Forest |
| ticker | VARCHAR(10) | Ticker or ALL for aggregate results | ALL |
| conflict_period | VARCHAR(30) | Train/test split description | First→Second Gulf War |
| accuracy | FLOAT | Overall prediction accuracy | 0.5783 |
| precision_score | FLOAT | Of predicted UP days, how many were correct | 0.6435 |
| recall | FLOAT | Of actual UP days, how many were predicted correctly | 0.4421 |
| f1_score | FLOAT | Harmonic mean of precision and recall | 0.5241 |
| notes | TEXT | Additional notes about the run | Train:GW1 Test:GW2 |

---

### live_predictions
Stores daily model predictions for current US-Iran tensions.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| id | SERIAL | Unique row identifier | 1 |
| prediction_date | DATE | Date prediction was generated | 2026-03-31 |
| ticker | VARCHAR(10) | Stock ticker symbol | XOM |
| region | VARCHAR(20) | Geographic region | US |
| latest_close | FLOAT | Most recent closing price | 171.47 |
| oil_price | FLOAT | WTI oil price on prediction date | 102.62 |
| oil_pct_change | FLOAT | WTI oil price percentage change | -0.25 |
| lr_signal | VARCHAR(20) | Logistic Regression prediction | UP |
| rf_signal | VARCHAR(20) | Random Forest prediction | UP |
| lr_probability | FLOAT | Logistic Regression confidence (0-1) | 0.730 |
| rf_probability | FLOAT | Random Forest confidence (0-1) | 0.731 |
| consensus | VARCHAR(20) | Agreement between both models | UP |

---

## Feature Engineering Transformations

| Feature | Formula | Purpose |
|---------|---------|---------|
| daily_return | (close - prev_close) / prev_close * 100 | Measures day-over-day price change as percentage |
| volatility_10d | Rolling 10-day std dev of daily_return | Captures short-term market nervousness |
| volatility_30d | Rolling 30-day std dev of daily_return | Captures medium-term volatility trend |
| moving_avg_return_20 | Rolling 20-day mean of daily_return | Short-term return momentum |
| moving_avg_return_50 | Rolling 50-day mean of daily_return | Medium-term return momentum |
| rsi | 100 - (100 / (1 + avg_gain/avg_loss)) over 14 days | Momentum indicator — above 70 overbought, below 30 oversold |
| oil_pct_change | (wti_price - prev_wti) / prev_wti * 100 | Daily oil price movement |
| days_since_event | (date - conflict_start_date).days | Proximity to key conflict event |
| target | 1 if daily_return > 0 else 0 | Binary label for model prediction |

---

## Raw vs Cleaned Data

| Stage | Location | Description |
|-------|---------|-------------|
| Raw | data/raw/stocks/ | Original CSV files downloaded from yfinance, one file per ticker per conflict period |
| Raw | data/raw/oil/ | Original WTI oil price CSV from EIA |
| Raw | data/raw/conflict_events.csv | Manually created events table |
| Cleaned | data/cleaned/stocks/ | Standardized column names, removed duplicates, added metadata columns |
| Cleaned | data/cleaned/oil/ | Added daily_change, pct_change, conflict_period, period_phase columns |
| Cleaned | data/cleaned/all_stocks_cleaned.csv | All tickers combined into single file |
| Database | AWS RDS PostgreSQL | All cleaned data loaded into structured tables for querying |
| Cloud | AWS S3 | Raw and cleaned files backed up to cloud storage |