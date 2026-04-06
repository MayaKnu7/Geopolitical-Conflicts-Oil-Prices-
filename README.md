# Geopolitical Conflict and Oil Stock Performance

**Project — Group 26**  
Sana Shah (94945664), Maya Knutsvig (17950502), Illina Islam (58189903)

---

## Overview

This project analyzes whether patterns in oil prices and stock market behavior observed during the **First Gulf War (1990–1991)** can be used to predict market responses during the **Second Gulf War (2003)** for 10 oil and gas companies. Based on these findings, the project explores whether similar patterns can provide insight into potential market reactions during current **U.S.–Iran geopolitical tensions**.

[Take a look!](http://56.112.29.64:8501)

---

## Companies Tracked

| Region | Ticker | Company |
| --- | --- | --- |
| Canada | SU | Suncor Energy |
| Canada | CNQ | Canadian Natural Resources |
| Canada | CVE* | Cenovus Energy |
| US | XOM | ExxonMobil |
| US | CVX | Chevron |
| US | COP | ConocoPhillips |
| International | SHEL | Shell |
| International | TTE | TotalEnergies |
| International | BP | BP |
| International | 2222.SR* | Saudi Aramco |

*CVE (founded 2009) and 2222.SR (IPO 2019) lack data for Gulf War periods and are included only in the current U.S.–Iran analysis. The remaining 8 companies are used for the full historical comparison.

---

## Research Questions

1. To what extent can oil price and oil & gas stock patterns observed during the First Gulf War predict market behavior during the Second Gulf War?
2. How do oil prices and stock markets respond before, during, and after major geopolitical conflict events?
3. Can patterns learned from historical conflicts provide insight into potential market behavior during the current U.S.–Iran tensions?

---

## Datasets

| Source | Description |
| --- | --- |
| [yfinance](https://pypi.org/project/yfinance/) | Historical stock prices for all 10 companies |
| [EIA](https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm) | Daily spot oil prices |
| [GDELT](https://www.gdeltproject.org) | Geopolitical event data |

---

## Pipeline

```text
01_data_acquisition.py   →   Fetch stock + oil price data via yfinance & EIA API
02_etl_cleaning.py       →   Clean, standardize, engineer features (returns, volatility, MAs)
03_eda.py                →   Exploratory analysis across conflict periods
04_modeling.py           →   Train on Gulf War I, test on Gulf War II
05_live_predictions.py   →   Apply model to live U.S.–Iran data
06_export_tableau.py     →   Export data for dashboards
app/app.py               →   Streamlit dashboard
```

### Stage Details

**Data Acquisition** — Collect historical stock and oil price data using Python (yfinance). Define conflict periods: First Gulf War (1990–1991), Second Gulf War (2003). Store raw data locally and in AWS S3.

**ETL / Cleaning** — Handle missing values and inconsistent dates. Standardize time-series data. Calculate derived features: daily returns, moving averages, volatility. Document all variables in `data_dictionary.md`. Load cleaned datasets into PostgreSQL (AWS RDS) with tables: `Stock_data`, `Oil_prices`, `Features`.

**EDA** — Analyze oil price trends during conflict periods. Compare stock performance before, during, and after conflicts. Identify volatility spikes and recovery patterns.

**Modeling** — Train a Logistic Regression baseline and a Random Forest Classifier on First Gulf War data. Test on Second Gulf War data, predicting oil price movement direction and stock direction. Evaluate generalizability across conflict periods.

**Live Predictions (Extension)** — Pull live oil price and stock data via EIA API and yfinance. Apply the trained model to current U.S.–Iran geopolitical conditions to generate real-time directional predictions for all 10 companies.

---

## Tools & Technologies

| Tool | Purpose |
| --- | --- |
| Python (pandas, numpy, scikit-learn, yfinance) | Data acquisition, ETL, EDA, modeling |
| PostgreSQL (AWS RDS) | Structured storage of cleaned datasets |
| AWS S3 | Raw and processed dataset storage |
| Streamlit | Interactive web dashboard |
| Tableau | Reporting dashboard — conflict period comparisons and model predictions vs. actuals |

---

## Project Structure

```text
.
├── scripts/
│   ├── 01_data_acquisition.py
│   ├── 02_etl_cleaning.py
│   ├── 03_eda.py
│   ├── 04_modeling.py
│   └── 05_live_predictions.py
├── app/
│   └── app.py
├── data/
├── models/
├── reports/
├── data_dictionary.md
├── requirements.txt
└── .streamlit/
    └── config.toml
```

---

## Risks & Limitations

- **Survivorship/availability bias** — CVE and 2222.SR did not exist during the First Gulf War and are excluded from that portion of the analysis.
- **Causation vs. correlation** — Market movements during conflict periods may be driven by many simultaneous factors beyond geopolitical events (e.g., recessions, interest rates).
- **Generalizability** — Patterns from 1990–2003 may not reliably predict modern markets due to algorithmic trading, ETFs, and structural changes in energy markets.
- **Data licensing** — yfinance is for educational/personal use. EIA data is public domain. Datasets used under their respective licenses.
- **No financial advice** — This project is purely academic and should not be interpreted as investment guidance.
