import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import pickle
import os
import yfinance as yf
from dotenv import load_dotenv
from datetime import datetime, timedelta


load_dotenv()


# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
   page_title = "Geopolitical Conflict & Oil Stock Performance",
   page_icon  = "🛢",
   layout     = "wide",
   initial_sidebar_state = "expanded"
)


# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
   <style>
   .main { background-color: #0E1117; }
   .metric-card {
       background-color: #1E2130;
       padding: 20px;
       border-radius: 10px;
       border-left: 4px solid #FF4B4B;
       margin: 10px 0;
   }
   .section-header {
       color: #FF4B4B;
       font-size: 1.3rem;
       font-weight: bold;
       margin-top: 20px;
   }
   .stSelectbox label { color: #FFFFFF; }
   </style>
""", unsafe_allow_html=True)


# ── Database Connection ──────────────────────────────────────
@st.cache_resource
def get_db_connection():
   return psycopg2.connect(
       host        = os.getenv("DB_HOST"),
       dbname      = os.getenv("DB_NAME"),
       user        = os.getenv("DB_USER"),
       password    = os.getenv("DB_PASSWORD"),
       sslmode     = "verify-full",
       sslrootcert = "./global-bundle.pem"
   )


# ── Load Data ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_stock_data():
   conn = get_db_connection()
   df   = pd.read_sql("SELECT * FROM stock_data", conn)
   df["date"] = pd.to_datetime(df["date"])
   return df


@st.cache_data(ttl=3600)
def load_oil_data():
   conn = get_db_connection()
   df   = pd.read_sql("SELECT * FROM oil_prices", conn)
   df["date"] = pd.to_datetime(df["date"])
   return df


@st.cache_data(ttl=3600)
def load_events_data():
   conn = get_db_connection()
   df   = pd.read_sql("SELECT * FROM conflict_events", conn)
   df["date"] = pd.to_datetime(df["date"])
   return df


@st.cache_data(ttl=3600)
def load_model_results():
   conn = get_db_connection()
   df   = pd.read_sql("SELECT * FROM model_results", conn)
   return df


@st.cache_data(ttl=300)
def load_live_predictions():
   conn = get_db_connection()
   df   = pd.read_sql("""
       SELECT * FROM live_predictions
       WHERE prediction_date = (
           SELECT MAX(prediction_date) FROM live_predictions
       )
   """, conn)
   return df


# ── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
   try:
       with open("models/logistic_regression_iran.pkl", "rb") as f:
           lr_model = pickle.load(f)
       with open("models/random_forest_iran.pkl", "rb") as f:
           rf_model = pickle.load(f)
       with open("models/scaler_iran.pkl", "rb") as f:
           scaler = pickle.load(f)
       return lr_model, rf_model, scaler
   except:
       return None, None, None


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
   st.title("Geopolitical Conflict & Oil Stocks")
   st.markdown("---")


   page = st.radio("Navigation", [
       "Overview",
       "Historical Analysis",
       "Model Results",
       "Live Predictions"
   ])


   st.markdown("---")
   st.markdown("**Group 26**")
   st.markdown("Sana Shah")
   st.markdown("Maya Knutsvig")
   st.markdown("Illina Islam")
   st.markdown("---")
   st.markdown("*COSC 301 — Data Analytics*")
   st.markdown("*For academic purposes only.*")
   st.markdown("*Not financial advice.*")


# ── Load all data ────────────────────────────────────────────
stock_df    = load_stock_data()
oil_df      = load_oil_data()
events_df   = load_events_data()
model_df    = load_model_results()


# ════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "Overview":
   st.title("Geopolitical Conflict & Oil Stock Performance")
   st.markdown("""
   This project analyzes whether oil price and stock market patterns
   observed during the **First Gulf War (1990-1991)** can predict market
   behavior during the **Second Gulf War (2003)**, and whether these
   patterns provide insight into current **US-Iran tensions**.
   """)


   st.markdown("---")


   # Key metrics
   col1, col2, col3, col4 = st.columns(4)
   with col1:
       st.metric("Companies Tracked", "10", "3 regions")
   with col2:
       st.metric("Conflict Periods", "3", "1990-2026")
   with col3:
       st.metric("Total Data Points", f"{len(stock_df):,}", "stock rows")
   with col4:
       st.metric("Conflict Events", str(len(events_df)), "key dates")


   st.markdown("---")


   # Tickers table
   st.subheader("Companies Analyzed")
   ticker_data = {
       "Region":  ["Canada", "Canada", "Canada",
                   "US", "US", "US",
                   "International", "International",
                   "International", "International"],
       "Ticker":  ["SU", "CNQ", "CVE",
                   "XOM", "CVX", "COP",
                   "SHEL", "TTE", "BP", "2222.SR"],
       "Company": ["Suncor Energy", "Canadian Natural Resources",
                   "Cenovus Energy", "ExxonMobil", "Chevron",
                   "ConocoPhillips", "Shell", "TotalEnergies",
                   "BP", "Saudi Aramco"],
       "Note":    ["Full history", "No Gulf War 1 data", "US-Iran only",
                   "Full history", "Full history", "Full history",
                   "Full history", "Partial Gulf War 1", "Full history",
                   "US-Iran only"]
   }
   st.dataframe(pd.DataFrame(ticker_data), use_container_width=True)


   st.markdown("---")


   # Conflict events timeline
   st.subheader("Key Conflict Events")
   fig = px.scatter(
       events_df,
       x     = "date",
       y     = "conflict_period",
       color = "severity",
       hover_data = ["event_description"],
       color_discrete_map = {
           "High":   "#FF4B4B",
           "Medium": "#FFA500",
           "Low":    "#00CC00"
       },
       title = "Conflict Events Timeline"
   )
   fig.update_traces(marker=dict(size=12))
   fig.update_layout(
       plot_bgcolor  = "#1E2130",
       paper_bgcolor = "#0E1117",
       font_color    = "white",
       height        = 350
   )
   st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2: HISTORICAL ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "Historical Analysis":
   st.title("Historical Analysis")


   # Filters
   col1, col2 = st.columns(2)
   with col1:
       selected_period = st.selectbox(
           "Conflict Period",
           ["First Gulf War", "Second Gulf War", "US-Iran"]
       )
   with col2:
       selected_region = st.multiselect(
           "Region",
           ["Canada", "US", "International"],
           default=["US"]
       )


   filtered_stock = stock_df[
       (stock_df["conflict_period"] == selected_period) &
       (stock_df["region"].isin(selected_region))
   ]
   filtered_oil = oil_df[
       oil_df["conflict_period"] == selected_period
   ]
   filtered_events = events_df[
       events_df["conflict_period"] == selected_period
   ]


   st.markdown("---")


   # Stock price chart
   st.subheader(f"Stock Prices — {selected_period}")
   if not filtered_stock.empty:
       fig = px.line(
           filtered_stock,
           x     = "date",
           y     = "close",
           color = "ticker",
           title = f"Oil & Gas Stock Prices During {selected_period}"
       )
       for _, event in filtered_events.iterrows():
           fig.add_vline(
               x           = event["date"].timestamp() * 1000,
               line_dash   = "dash",
               line_color  = "red" if event["severity"] == "High" else "orange",
               annotation_text = event["event_description"][:20],
               annotation_font_size = 9
           )
       fig.update_layout(
           plot_bgcolor  = "#1E2130",
           paper_bgcolor = "#0E1117",
           font_color    = "white",
           height        = 450
       )
       st.plotly_chart(fig, use_container_width=True)


   # Oil price chart
   st.subheader(f"WTI Oil Price — {selected_period}")
   if not filtered_oil.empty:
       fig2 = px.line(
           filtered_oil,
           x     = "date",
           y     = "wti_price",
           title = f"WTI Oil Price During {selected_period}"
       )
       for _, event in filtered_events.iterrows():
           fig2.add_vline(
               x          = event["date"].timestamp() * 1000,
               line_dash  = "dash",
               line_color = "red" if event["severity"] == "High" else "orange"
           )
       fig2.update_layout(
           plot_bgcolor  = "#1E2130",
           paper_bgcolor = "#0E1117",
           font_color    = "white",
           height        = 350
       )
       st.plotly_chart(fig2, use_container_width=True)


   st.markdown("---")


   # Before / During / After comparison
   st.subheader("Average Stock Price by Phase")
   phase_avg = filtered_stock.groupby(
       ["period_phase", "ticker"])["close"].mean().reset_index()
   phase_avg = phase_avg[phase_avg["period_phase"].isin(
       ["Before", "During", "After"])]


   if not phase_avg.empty:
       fig3 = px.bar(
           phase_avg,
           x        = "period_phase",
           y        = "close",
           color    = "ticker",
           barmode  = "group",
           category_orders = {"period_phase": ["Before", "During", "After"]},
           title    = "Average Close Price: Before vs During vs After Conflict"
       )
       fig3.update_layout(
           plot_bgcolor  = "#1E2130",
           paper_bgcolor = "#0E1117",
           font_color    = "white",
           height        = 400
       )
       st.plotly_chart(fig3, use_container_width=True)


   # Correlation heatmap
   st.subheader("Stock Price Correlation")
   if not filtered_stock.empty:
       pivot = filtered_stock.pivot_table(
           index="date", columns="ticker", values="close")
       corr = pivot.corr()


       fig4 = px.imshow(
           corr,
           color_continuous_scale = "RdBu_r",
           title = f"Stock Correlation — {selected_period}",
           zmin  = -1,
           zmax  = 1
       )
       fig4.update_layout(
           plot_bgcolor  = "#1E2130",
           paper_bgcolor = "#0E1117",
           font_color    = "white",
           height        = 400
       )
       st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3: MODEL RESULTS
# ════════════════════════════════════════════════════════════
elif page == "Model Results":
   st.title("Model Results")


   st.markdown("""
   Two models were trained and evaluated:
   - **Logistic Regression** — baseline model
   - **Random Forest** — primary model with constraints to prevent overfitting


   Models were trained on Gulf War data and tested on subsequent conflict periods.
   """)


   st.markdown("---")


   if not model_df.empty:
       # Model comparison table
       st.subheader("Performance Metrics")
       display_cols = [
           "model_name", "conflict_period",
           "accuracy", "precision_score", "recall", "f1_score"
       ]
       available_cols = [c for c in display_cols if c in model_df.columns]
       st.dataframe(
           model_df[available_cols].round(4),
           use_container_width=True
       )


       st.markdown("---")


       # Accuracy comparison chart
       st.subheader("Accuracy Comparison")
       fig5 = px.bar(
           model_df,
           x        = "conflict_period",
           y        = "accuracy",
           color    = "model_name",
           barmode  = "group",
           title    = "Model Accuracy by Experiment"
       )
       fig5.update_layout(
           plot_bgcolor  = "#1E2130",
           paper_bgcolor = "#0E1117",
           font_color    = "white",
           height        = 400,
           yaxis_range   = [0, 1]
       )
       st.plotly_chart(fig5, use_container_width=True)


       # F1 comparison chart
       st.subheader("F1 Score Comparison")
       fig6 = px.bar(
           model_df,
           x        = "conflict_period",
           y        = "f1_score",
           color    = "model_name",
           barmode  = "group",
           title    = "Model F1 Score by Experiment"
       )
       fig6.update_layout(
           plot_bgcolor  = "#1E2130",
           paper_bgcolor = "#0E1117",
           font_color    = "white",
           height        = 400,
           yaxis_range   = [0, 1]
       )
       st.plotly_chart(fig6, use_container_width=True)


   st.markdown("---")


   # Feature importance image
   st.subheader("Feature Importance")
   if os.path.exists("reports/modeling/feature_importance.png"):
       st.image("reports/modeling/feature_importance.png",
                use_column_width=True)


   # Model interpretation
   st.markdown("---")
   st.subheader("Interpretation")
   st.markdown("""
   - Models achieve **58-64% accuracy** on out-of-sample conflict periods
   - This is meaningful — random guessing would give 50%
   - **Logistic Regression** slightly outperforms Random Forest on generalization
   - Simpler models transfer better across time periods
   - Accuracy improves when trained on more conflict periods (Experiment 2 vs 1)
   - Limitations: small dataset, structural market changes over decades,
     many confounding factors beyond geopolitical events
   """)


# ════════════════════════════════════════════════════════════
# PAGE 4: LIVE PREDICTIONS
# ════════════════════════════════════════════════════════════
elif page == "Live Predictions":
   st.title("Live Predictions — Current US-Iran Tensions")


   st.warning("DISCLAIMER: For academic purposes only. Not financial advice.")


   st.markdown("---")


   # Load latest predictions from DB
   try:
       predictions_df = load_live_predictions()


       if predictions_df.empty:
           st.info("No predictions found. Run 05_live_predictions.py first.")
       else:
           pred_date = predictions_df["prediction_date"].iloc[0]
           oil_price = predictions_df["oil_price"].iloc[0]
           oil_chg   = predictions_df["oil_pct_change"].iloc[0]


           # Oil price metric
           col1, col2, col3 = st.columns(3)
           with col1:
               st.metric(
                   "WTI Oil Price",
                   f"${oil_price:.2f}",
                   f"{oil_chg:+.2f}%"
               )
           with col2:
               up_count = (predictions_df["consensus"] == "UP").sum()
               st.metric("Bullish Signals", str(up_count), "of 10 stocks")
           with col3:
               st.metric("Prediction Date", str(pred_date))


           st.markdown("---")


           # Predictions table
           st.subheader("Predictions by Company")
           display_df = predictions_df[[
               "ticker", "region", "latest_close",
               "lr_signal", "rf_signal", "consensus",
               "lr_probability", "rf_probability"
           ]].copy()
           display_df["lr_probability"] = (
               display_df["lr_probability"] * 100).round(1).astype(str) + "%"
           display_df["rf_probability"] = (
               display_df["rf_probability"] * 100).round(1).astype(str) + "%"
           display_df.columns = [
               "Ticker", "Region", "Latest Close",
               "LR Signal", "RF Signal", "Consensus",
               "LR Confidence", "RF Confidence"
           ]
           st.dataframe(display_df, use_container_width=True)


           st.markdown("---")


           # Confidence chart
           st.subheader("Prediction Confidence by Ticker")
           conf_df = predictions_df.copy()
           conf_df["avg_confidence"] = (
               conf_df["lr_probability"] + conf_df["rf_probability"]
           ) / 2 * 100


           fig7 = px.bar(
               conf_df,
               x     = "ticker",
               y     = "avg_confidence",
               color = "region",
               title = "Average Model Confidence by Ticker",
               color_discrete_map = {
                   "Canada":        "#4B9FFF",
                   "US":            "#FF4B4B",
                   "International": "#00CC88"
               }
           )
           fig7.add_hline(
               y             = 50,
               line_dash     = "dash",
               line_color    = "white",
               annotation_text = "Random chance (50%)"
           )
           fig7.update_layout(
               plot_bgcolor  = "#1E2130",
               paper_bgcolor = "#0E1117",
               font_color    = "white",
               height        = 400,
               yaxis_range   = [0, 100]
           )
           st.plotly_chart(fig7, use_container_width=True)


           # Live stock chart
           st.markdown("---")
           st.subheader("Recent Stock Performance (Last 90 Days)")
           selected_ticker = st.selectbox(
               "Select ticker",
               predictions_df["ticker"].tolist()
           )


           live_data = yf.download(
               selected_ticker,
               start    = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d"),
               end      = datetime.today().strftime("%Y-%m-%d"),
               progress = False
           )


           if not live_data.empty:
               live_data = live_data.reset_index()
               # Flatten multi-level columns
               live_data.columns = [col[0] if isinstance(col, tuple) else col
                                   for col in live_data.columns]
               fig8 = px.line(
                   live_data,
                   x     = "Date",
                   y     = "Close",
                   title = f"{selected_ticker} — Last 90 Days"
               )
               fig8.update_layout(
                   plot_bgcolor  = "#1E2130",
                   paper_bgcolor = "#0E1117",
                   font_color    = "white",
                   height        = 350
               )
               st.plotly_chart(fig8, use_container_width=True)


   except Exception as e:
       st.error(f"Error loading predictions: {e}")
       st.info("Run 05_live_predictions.py to generate predictions first.")



