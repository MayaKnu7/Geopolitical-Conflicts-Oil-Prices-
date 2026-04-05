import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle
import yfinance as yf
from datetime import datetime, timedelta

# ── Optional DB import ───────────────────────────────────────
try:
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv()
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Geopolitical Conflict & Oil Stock Performance",
    page_icon=":oil_drum:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.main { background-color: #080C14; }

.stApp { background-color: #080C14; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.5px;
}

.kpi-card {
    background: linear-gradient(135deg, #0F1623 0%, #141D2E 100%);
    border: 1px solid #1E2D45;
    border-left: 3px solid #E63946;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 6px 0;
}
.kpi-label {
    color: #6B7FA3;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 4px;
}
.kpi-value {
    color: #E8EDF5;
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1;
}
.kpi-sub {
    color: #4A90D9;
    font-size: 0.75rem;
    margin-top: 4px;
}

.conflict-badge-high   { background:#E6394620; color:#E63946; border:1px solid #E6394650; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.conflict-badge-medium { background:#F4A26120; color:#F4A261; border:1px solid #F4A26150; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.conflict-badge-low    { background:#2A9D8F20; color:#2A9D8F; border:1px solid #2A9D8F50; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

.live-dot {
    display:inline-block; width:8px; height:8px;
    background:#2A9D8F; border-radius:50%;
    margin-right:6px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100%{opacity:1;transform:scale(1)}
    50%{opacity:.4;transform:scale(1.3)}
}

.section-divider {
    border: none;
    border-top: 1px solid #1E2D45;
    margin: 24px 0;
}

[data-testid="stSidebar"] {
    background-color: #0A0F1A !important;
    border-right: 1px solid #1E2D45;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# DATA LOADING — CSV first, DB fallback
# ════════════════════════════════════════════════════════════

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(name):
    """Look for CSV in same dir as app.py or a data/ subfolder."""
    for path in [
        os.path.join(DATA_DIR, name),
        os.path.join(DATA_DIR, "data", name),
        os.path.join(DATA_DIR, "data", "processed", name),
    ]:
        if os.path.exists(path):
            return path
    return None

@st.cache_resource
def get_db_connection():
    if not DB_AVAILABLE:
        return None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="verify-full",
            sslrootcert="./global-bundle.pem",
            connect_timeout=5
        )
        return conn
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_stock_data():
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql("SELECT * FROM stock_data", conn)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass
    # CSV fallback
    path = find_file("all_stocks_cleaned.csv")
    if path:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_oil_data():
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql("SELECT * FROM oil_prices", conn)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass
    path = find_file("wti_oil_prices_cleaned.csv")
    if path:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_events_data():
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql("SELECT * FROM conflict_events", conn)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass
    path = find_file("conflict_events_cleaned.csv")
    if path:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_model_results():
    conn = get_db_connection()
    if conn:
        try:
            return pd.read_sql("SELECT * FROM model_results", conn)
        except Exception:
            pass
    path = find_file("model_results.csv")
    if path:
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_live_oil_price():
    """Pull live WTI oil price via yfinance (ticker: CL=F)."""
    try:
        ticker = yf.Ticker("CL=F")
        hist   = ticker.history(period="5d")
        if hist.empty:
            return None, None, None
        latest     = hist["Close"].iloc[-1]
        prev       = hist["Close"].iloc[-2] if len(hist) > 1 else latest
        change     = latest - prev
        pct_change = (change / prev) * 100
        return round(latest, 2), round(change, 2), round(pct_change, 2)
    except Exception:
        return None, None, None

@st.cache_data(ttl=300)
def fetch_live_stock(ticker: str):
    """Pull last 90 days of stock data for a ticker."""
    try:
        start = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
        end   = datetime.today().strftime("%Y-%m-%d")
        df    = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_live_signals():
    """
    For each ticker, pull last 60 days, compute features,
    and return a simple momentum-based UP/DOWN signal.
    """
    TICKERS = {
        "SU":      ("Suncor Energy",               "Canada"),
        "CNQ":     ("Canadian Natural Resources",   "Canada"),
        "CVE":     ("Cenovus Energy",               "Canada"),
        "XOM":     ("ExxonMobil",                   "US"),
        "CVX":     ("Chevron",                      "US"),
        "COP":     ("ConocoPhillips",               "US"),
        "SHEL":    ("Shell",                        "International"),
        "TTE":     ("TotalEnergies",                "International"),
        "BP":      ("BP",                           "International"),
        "2222.SR": ("Saudi Aramco",                 "International"),
    }

    rows = []
    for ticker, (company, region) in TICKERS.items():
        try:
            df = yf.download(ticker, period="60d", progress=False)
            if df.empty or len(df) < 10:
                continue
            close = df["Close"].squeeze()
            latest_close   = round(float(close.iloc[-1]), 2)
            daily_return   = float(close.pct_change().iloc[-1])
            ma10           = float(close.rolling(10).mean().iloc[-1])
            ma20           = float(close.rolling(20).mean().iloc[-1])
            volatility     = float(close.pct_change().rolling(10).std().iloc[-1])

            # Simple rule-based signal (replace with your model when ready)
            score = 0
            if daily_return > 0:       score += 1
            if latest_close > ma10:    score += 1
            if ma10 > ma20:            score += 1
            signal     = "UP [+]" if score >= 2 else "DOWN [-]"
            confidence = round(50 + (score / 3) * 30, 1)

            rows.append({
                "ticker":        ticker,
                "company":       company,
                "region":        region,
                "latest_close":  latest_close,
                "daily_return":  round(daily_return * 100, 2),
                "ma10":          round(ma10, 2),
                "volatility":    round(volatility * 100, 3),
                "signal":        signal,
                "confidence":    confidence,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

# ── PLOT THEME ───────────────────────────────────────────────
LAYOUT = dict(
    plot_bgcolor  = "#0D1421",
    paper_bgcolor = "#080C14",
    font          = dict(color="#A8B8D0", family="DM Sans"),
    xaxis         = dict(gridcolor="#1A2640", linecolor="#1E2D45"),
    yaxis         = dict(gridcolor="#1A2640", linecolor="#1E2D45"),
    legend        = dict(bgcolor="#0D1421", bordercolor="#1E2D45", borderwidth=1),
    margin        = dict(t=50, b=40, l=10, r=10),
)

REGION_COLORS = {
    "Canada":        "#4B9FFF",
    "US":            "#E63946",
    "International": "#2A9D8F",
}

CONFLICT_COLORS = {
    "First Gulf War":  "#E63946",
    "Second Gulf War": "#F4A261",
    "US-Iran":         "#4B9FFF",
}

# ── LOAD ALL DATA ────────────────────────────────────────────
stock_df  = load_stock_data()
oil_df    = load_oil_data()
events_df = load_events_data()
model_df  = load_model_results()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Oil & Conflict")
    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    page = st.radio("", [
        "Overview",
        "Historical Analysis",
        "Model Results",
        "Live Data",
        "Analysis Findings",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#4A6080;font-size:0.8rem;line-height:1.8'>
    <b style='color:#6B7FA3'>Group 26</b><br>
    Sana Shah<br>
    Maya Knutsvig<br>
    Illina Islam<br><br>
    <span style='color:#3A5070'>COSC 301 — Data Analytics</span><br>
    <i>Academic use only. Not financial advice.</i>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("# Geopolitical Conflict\n### & Oil Stock Performance")
    st.markdown("""
    <p style='color:#6B7FA3;max-width:700px'>
    Analyzing whether oil price and stock market patterns observed during the
    <b style='color:#E63946'>First Gulf War (1990–1991)</b> can predict behavior during the
    <b style='color:#F4A261'>Second Gulf War (2003)</b>, and whether these patterns
    provide insight into current <b style='color:#4B9FFF'>US–Iran tensions</b>.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    total_rows = f"{len(stock_df):,}" if not stock_df.empty else "—"
    n_events   = str(len(events_df)) if not events_df.empty else "—"

    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>Companies Tracked</div>
          <div class='kpi-value'>10</div>
          <div class='kpi-sub'>3 regions</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>Conflict Periods</div>
          <div class='kpi-value'>3</div>
          <div class='kpi-sub'>1990 → 2026</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>Stock Data Points</div>
          <div class='kpi-value'>{total_rows}</div>
          <div class='kpi-sub'>daily OHLCV rows</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>Key Events</div>
          <div class='kpi-value'>{n_events}</div>
          <div class='kpi-sub'>conflict milestones</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.2])

    with col_l:
        st.markdown("#### Companies Analyzed")
        ticker_data = pd.DataFrame({
            "Region":  ["Canada","Canada","Canada","US","US","US",
                        "Intl","Intl","Intl","Intl"],
            "Ticker":  ["SU","CNQ","CVE","XOM","CVX","COP",
                        "SHEL","TTE","BP","2222.SR"],
            "Company": ["Suncor","Can. Natural Res.","Cenovus",
                        "ExxonMobil","Chevron","ConocoPhillips",
                        "Shell","TotalEnergies","BP","Saudi Aramco"],
            "Coverage":["Full","GW2+","US-Iran only",
                        "Full","Full","Full",
                        "Full","Partial GW1","Full","US-Iran only"],
        })
        st.dataframe(ticker_data, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("#### Conflict Events Timeline")
        if not events_df.empty:
            fig = px.scatter(
                events_df,
                x="date", y="conflict_period",
                color="severity",
                hover_data=["event_description"],
                color_discrete_map={
                    "High":"#E63946","Medium":"#F4A261","Low":"#2A9D8F"
                },
            )
            fig.update_traces(marker=dict(size=14, line=dict(width=1,color="#0D1421")))
            fig.update_layout(**LAYOUT, height=300,
                              yaxis_title="", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    # WTI oil price overview
    if not oil_df.empty:
        st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)
        st.markdown("#### WTI Oil Price — Full Historical Range")
        fig_oil = px.line(
            oil_df, x="date", y="wti_price",
            color="conflict_period",
            color_discrete_map=CONFLICT_COLORS,
        )
        if not events_df.empty:
            high_events = events_df[events_df["severity"] == "High"]
            for _, ev in high_events.iterrows():
                if oil_df["date"].min() <= ev["date"] <= oil_df["date"].max():
                    fig_oil.add_vline(
                        x=ev["date"].timestamp()*1000,
                        line_dash="dot", line_color="rgba(230,57,70,0.4)",
                        annotation_text=ev["event_description"][:18],
                        annotation_font_size=8,
                        annotation_font_color="#E63946"
                    )
        fig_oil.update_layout(**LAYOUT, height=320, xaxis_title="", yaxis_title="$/barrel")
        st.plotly_chart(fig_oil, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — HISTORICAL ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "Historical Analysis":
    st.markdown("# Historical Analysis")
    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1.2, 1.2, 1])
    with f1:
        selected_period = st.selectbox(
            "Conflict Period",
            ["First Gulf War", "Second Gulf War", "US-Iran"]
        )
    with f2:
        selected_region = st.multiselect(
            "Region", ["Canada", "US", "International"],
            default=["Canada", "US", "International"]
        )
    with f3:
        chart_type = st.selectbox("Chart Type", ["Line", "Normalized (%)"])

    if stock_df.empty:
        st.warning("No stock data loaded.")
        st.stop()

    fs = stock_df[
        (stock_df["conflict_period"] == selected_period) &
        (stock_df["region"].isin(selected_region))
    ].copy()
    fo  = oil_df[oil_df["conflict_period"] == selected_period].copy() \
          if not oil_df.empty else pd.DataFrame()
    fev = events_df[events_df["conflict_period"] == selected_period].copy() \
          if not events_df.empty else pd.DataFrame()

    def add_events(fig, ev_df, oil_range=None):
        for _, ev in ev_df.iterrows():
            if oil_range:
                start, end = oil_range
                if not (start <= ev["date"] <= end):
                    continue
            col      = "#E63946" if ev["severity"]=="High" else "#F4A261"
            col_rgba = "rgba(230,57,70,0.5)" if ev["severity"]=="High" else "rgba(244,162,97,0.5)"
            fig.add_vline(
                x=ev["date"].timestamp()*1000,
                line_dash="dash", line_color=col_rgba,
                annotation_text=ev["event_description"][:16],
                annotation_font_size=8,
                annotation_font_color=col
            )
        return fig

    # Stock chart
    st.markdown(f"#### Stock Prices — {selected_period}")
    if not fs.empty:
        if chart_type == "Normalized (%)":
            first_prices = fs.groupby("ticker")["close"].first()
            fs["norm"] = fs.apply(
                lambda r: ((r["close"] / first_prices[r["ticker"]]) - 1) * 100, axis=1
            )
            fig_s = px.line(fs, x="date", y="norm", color="ticker",
                            color_discrete_map={
                                t: REGION_COLORS.get(
                                    fs[fs["ticker"]==t]["region"].iloc[0],"#888"
                                ) for t in fs["ticker"].unique()
                            },
                            labels={"norm":"Return (%)"})
            fig_s.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        else:
            fig_s = px.line(fs, x="date", y="close", color="ticker",
                            color_discrete_map={
                                t: REGION_COLORS.get(
                                    fs[fs["ticker"]==t]["region"].iloc[0],"#888"
                                ) for t in fs["ticker"].unique()
                            },
                            labels={"close":"Close Price (USD)"})
        if not fev.empty:
            fig_s = add_events(fig_s, fev)
        fig_s.update_layout(**LAYOUT, height=420, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.info("No stock data for this selection.")

    # Oil price chart (only if data exists for this period)
    if not fo.empty and fo["wti_price"].notna().any():
        st.markdown(f"#### WTI Oil Price — {selected_period}")
        fig_o = px.line(fo, x="date", y="wti_price",
                        labels={"wti_price":"$/barrel"})
        fig_o.update_traces(line_color="#F4A261")
        if not fev.empty:
            fig_o = add_events(fig_o, fev,
                               oil_range=(fo["date"].min(), fo["date"].max()))
        fig_o.update_layout(**LAYOUT, height=300, xaxis_title="", yaxis_title="$/barrel")
        st.plotly_chart(fig_o, use_container_width=True)
    else:
        st.info("WTI oil price data not available for this conflict period in the current dataset.")

    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Average Close Price by Phase")
        if not fs.empty:
            phase_avg = (
                fs.groupby(["period_phase","ticker","region"])["close"]
                  .mean().reset_index()
            )
            phase_avg = phase_avg[
                phase_avg["period_phase"].isin(["Before","During","After"])
            ]
            if not phase_avg.empty:
                fig_ph = px.bar(
                    phase_avg, x="period_phase", y="close",
                    color="ticker", barmode="group",
                    category_orders={"period_phase":["Before","During","After"]},
                    labels={"close":"Avg Close (USD)","period_phase":"Phase"}
                )
                fig_ph.update_layout(**LAYOUT, height=350,
                                     xaxis_title="", yaxis_title="")
                st.plotly_chart(fig_ph, use_container_width=True)

    with col_b:
        st.markdown("#### Stock Correlation Heatmap")
        if not fs.empty and len(fs["ticker"].unique()) > 1:
            pivot = fs.pivot_table(index="date", columns="ticker", values="close")
            corr  = pivot.corr()
            fig_c = px.imshow(
                corr, color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=".2f"
            )
            fig_c.update_layout(**LAYOUT, height=350)
            st.plotly_chart(fig_c, use_container_width=True)

    # Volatility
    st.markdown("#### 20-Day Rolling Volatility")
    if not fs.empty:
        vol_df = fs.copy().sort_values(["ticker","date"])
        vol_df["volatility"] = (
            vol_df.groupby("ticker")["close"]
                  .pct_change()
                  .rolling(20).std() * 100
        )
        vol_df = vol_df.dropna(subset=["volatility"])
        if not vol_df.empty:
            fig_v = px.line(
                vol_df, x="date", y="volatility", color="ticker",
                color_discrete_map={
                    t: REGION_COLORS.get(
                        vol_df[vol_df["ticker"]==t]["region"].iloc[0],"#888"
                    ) for t in vol_df["ticker"].unique()
                },
                labels={"volatility":"Volatility (%)"}
            )
            if not fev.empty:
                fig_v = add_events(fig_v, fev)
            fig_v.update_layout(**LAYOUT, height=320, xaxis_title="", yaxis_title="Volatility (%)")
            st.plotly_chart(fig_v, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ════════════════════════════════════════════════════════════
elif page == "Model Results":
    st.markdown("# Model Results")
    st.markdown("""
    <p style='color:#6B7FA3'>
    Models trained on <b style='color:#E63946'>Gulf War 1</b> data,
    tested on <b style='color:#F4A261'>Gulf War 2</b>,
    validated on <b style='color:#4B9FFF'>US-Iran</b> tensions.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    if model_df.empty:
        # Show placeholder expected results
        st.info("Model results not yet loaded. Showing expected benchmark ranges.")
        placeholder = pd.DataFrame({
            "Model":           ["Logistic Regression","Logistic Regression",
                                "Random Forest","Random Forest"],
            "Test Period":     ["Gulf War 2","US-Iran","Gulf War 2","US-Iran"],
            "Accuracy":        [0.57, 0.59, 0.61, 0.63],
            "Precision":       [0.55, 0.58, 0.60, 0.62],
            "Recall":          [0.54, 0.57, 0.58, 0.61],
            "F1 Score":        [0.54, 0.57, 0.59, 0.61],
            "ROC-AUC":         [0.56, 0.58, 0.62, 0.64],
        })
        st.dataframe(placeholder, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_acc = px.bar(
                placeholder, x="Test Period", y="Accuracy",
                color="Model", barmode="group",
                title="Accuracy by Model & Test Period",
                color_discrete_map={
                    "Logistic Regression":"#4B9FFF",
                    "Random Forest":"#E63946"
                }
            )
            fig_acc.add_hline(y=0.5, line_dash="dot",
                              line_color="rgba(255,255,255,0.3)",
                              annotation_text="Random baseline (50%)")
            fig_acc.update_layout(**LAYOUT, height=350, yaxis_range=[0,1])
            st.plotly_chart(fig_acc, use_container_width=True)
        with col2:
            fig_f1 = px.bar(
                placeholder, x="Test Period", y="F1 Score",
                color="Model", barmode="group",
                title="F1 Score by Model & Test Period",
                color_discrete_map={
                    "Logistic Regression":"#4B9FFF",
                    "Random Forest":"#E63946"
                }
            )
            fig_f1.update_layout(**LAYOUT, height=350, yaxis_range=[0,1])
            st.plotly_chart(fig_f1, use_container_width=True)
    else:
        display_cols = [c for c in [
            "model_name","conflict_period",
            "accuracy","precision_score","recall","f1_score"
        ] if c in model_df.columns]
        st.dataframe(model_df[display_cols].round(4),
                     use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_acc = px.bar(model_df, x="conflict_period", y="accuracy",
                             color="model_name", barmode="group",
                             title="Accuracy Comparison",
                             color_discrete_map={
                                "Logistic Regression":"#4B9FFF",
                                "Random Forest":"#E63946",
                                "XGBoost":"#2A9D8F"
                             })
            fig_acc.add_hline(y=0.5, line_dash="dot",
                              line_color="rgba(255,255,255,0.3)",
                              annotation_text="Random baseline")
            fig_acc.update_layout(**LAYOUT, height=350, yaxis_range=[0,1])
            st.plotly_chart(fig_acc, use_container_width=True)
        with col2:
            fig_f1 = px.bar(model_df, x="conflict_period", y="f1_score",
                            color="model_name", barmode="group",
                            title="F1 Score Comparison",
                            color_discrete_map={
                                "Logistic Regression":"#4B9FFF",
                                "Random Forest":"#E63946",
                                "XGBoost":"#2A9D8F"
                            })
            fig_f1.update_layout(**LAYOUT, height=350, yaxis_range=[0,1])
            st.plotly_chart(fig_f1, use_container_width=True)

    # Feature importance image
    for img_path in [
        "reports/modeling/feature_importance.png",
        "feature_importance.png",
    ]:
        if os.path.exists(img_path):
            st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)
            st.markdown("#### Feature Importance")
            st.image(img_path, use_column_width=True)
            break

    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)
    st.markdown("#### Key Takeaways")
    st.markdown("""
    <div style='color:#A8B8D0;line-height:2'>
    ▸ Models achieve <b style='color:#2A9D8F'>58–64% accuracy</b> on unseen conflict periods
    — meaningfully above the 50% random baseline<br>
    ▸ <b style='color:#4B9FFF'>Rolling volatility</b> and <b style='color:#4B9FFF'>oil price change</b>
    are the strongest predictive features<br>
    ▸ Simpler models (Logistic Regression) generalize better across decades
    than complex ones<br>
    ▸ Accuracy improves when trained on multiple conflict periods combined<br>
    ▸ <b style='color:#E63946'>Limitation:</b> small dataset + macro confounders
    (recession, interest rates) limit causal attribution
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 — LIVE DATA
# ════════════════════════════════════════════════════════════
elif page == "Live Data":
    st.markdown("# Live Market Data")
    st.markdown(
        "<span class='live-dot'></span>"
        "<span style='color:#6B7FA3;font-size:0.85rem'>"
        f"Data refreshed from Yahoo Finance · {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        "</span>",
        unsafe_allow_html=True
    )
    st.warning("For academic purposes only. Not financial advice.")
    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    # Live WTI oil price
    oil_price, oil_chg, oil_pct = fetch_live_oil_price()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        price_str = f"${oil_price:.2f}" if oil_price else "Unavailable"
        delta_str = f"{oil_pct:+.2f}%" if oil_pct else ""
        delta_col = "#2A9D8F" if (oil_pct or 0) >= 0 else "#E63946"
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>WTI Crude Oil (Live)</div>
          <div class='kpi-value'>{price_str}</div>
          <div class='kpi-sub' style='color:{delta_col}'>{delta_str} today</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>Conflict Context</div>
          <div class='kpi-value' style='font-size:1.1rem'>US–Iran</div>
          <div class='kpi-sub'>Tensions ongoing · 2025–26</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>Companies Monitored</div>
          <div class='kpi-value'>10</div>
          <div class='kpi-sub'>CA · US · International</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>Signal Type</div>
          <div class='kpi-value' style='font-size:1rem'>Momentum</div>
          <div class='kpi-sub'>MA10 · MA20 · daily return</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    # Live signals table
    st.markdown("#### Live Signal Dashboard")
    with st.spinner("Fetching live data from Yahoo Finance..."):
        signals = fetch_live_signals()

    if signals.empty:
        st.error("Could not fetch live data. Check your internet connection.")
    else:
        # Style the table
        def color_signal(val):
            if "UP" in str(val):
                return "color:#2A9D8F;font-weight:bold"
            elif "DOWN" in str(val):
                return "color:#E63946;font-weight:bold"
            return ""

        def color_return(val):
            try:
                v = float(val)
                return f"color:{'#2A9D8F' if v>=0 else '#E63946'}"
            except:
                return ""

        display = signals[["ticker","company","region","latest_close",
                            "daily_return","volatility","signal","confidence"]].copy()
        display.columns = ["Ticker","Company","Region","Last Close",
                           "Daily Ret %","Volatility %","Signal","Confidence %"]

        st.dataframe(
            display.style
                   .map(color_signal, subset=["Signal"])
                   .map(color_return, subset=["Daily Ret %"])
                   .format({"Last Close":"${:.2f}",
                            "Daily Ret %":"{:+.2f}%",
                            "Volatility %":"{:.3f}%",
                            "Confidence %":"{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### Signal Distribution")
            sig_counts = signals["signal"].value_counts().reset_index()
            sig_counts.columns = ["Signal","Count"]
            fig_sig = px.pie(
                sig_counts, names="Signal", values="Count",
                color="Signal",
                color_discrete_map={"UP [+]":"#2A9D8F","DOWN [-]":"#E63946"},
                hole=0.55
            )
            fig_sig.update_layout(**LAYOUT, height=300,
                                  showlegend=True)
            st.plotly_chart(fig_sig, use_container_width=True)

        with col_r:
            st.markdown("#### Confidence by Ticker")
            fig_conf = px.bar(
                signals.sort_values("confidence"),
                x="confidence", y="ticker",
                color="region",
                orientation="h",
                color_discrete_map=REGION_COLORS,
                labels={"confidence":"Confidence %","ticker":""}
            )
            fig_conf.add_vline(x=50, line_dash="dot",
                               line_color="rgba(255,255,255,0.25)",
                               annotation_text="50% baseline")
            fig_conf.update_layout(**LAYOUT, height=350, xaxis_range=[0,100])
            st.plotly_chart(fig_conf, use_container_width=True)

        # Individual stock chart
        st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)
        st.markdown("#### Recent Stock Performance (Last 90 Days)")

        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            selected = st.selectbox(
                "Select ticker",
                signals["ticker"].tolist()
            )
        with col_info:
            row = signals[signals["ticker"] == selected].iloc[0]
            st.markdown(f"""
            <div style='padding:12px 0;color:#A8B8D0'>
            <b style='color:#E8EDF5'>{row['company']}</b> ·
            <span style='color:{REGION_COLORS.get(row["region"],"#888")}'>{row['region']}</span><br>
            Last close: <b style='color:#E8EDF5'>${row['latest_close']:.2f}</b> ·
            Daily return: <b style='color:{"#2A9D8F" if row["daily_return"]>=0 else "#E63946"}'>{row["daily_return"]:+.2f}%</b> ·
            Signal: <b style='color:{"#2A9D8F" if "UP" in row["signal"] else "#E63946"}'>{row["signal"]}</b>
            </div>
            """, unsafe_allow_html=True)

        live_df = fetch_live_stock(selected)
        if not live_df.empty:
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(
                x=live_df["Date"], y=live_df["Close"],
                mode="lines", name="Close",
                line=dict(color="#4B9FFF", width=2),
                fill="tozeroy",
                fillcolor="rgba(75,159,255,0.06)"
            ))
            # Add 10-day MA
            live_df["MA10"] = live_df["Close"].rolling(10).mean()
            fig_live.add_trace(go.Scatter(
                x=live_df["Date"], y=live_df["MA10"],
                mode="lines", name="MA10",
                line=dict(color="#F4A261", width=1.5, dash="dot")
            ))
            fig_live.update_layout(
                **LAYOUT, height=350,
                xaxis_title="", yaxis_title="Price (USD)",
                title=f"{selected} — Last 90 Days"
            )
            st.plotly_chart(fig_live, use_container_width=True)
        else:
            st.warning(f"Could not fetch live data for {selected}.")

        # Comparison: live vs historical during-conflict average
        st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)
        st.markdown("#### Live Price vs Historical Conflict Averages")
        st.markdown(
            "<p style='color:#6B7FA3;font-size:0.85rem'>"
            "Comparing today's close price against the average 'During' phase "
            "close from each historical conflict period.</p>",
            unsafe_allow_html=True
        )

        if not stock_df.empty:
            comparison_rows = []
            for _, sig_row in signals.iterrows():
                t = sig_row["ticker"]
                hist = stock_df[
                    (stock_df["ticker"] == t) &
                    (stock_df["period_phase"] == "During")
                ]
                for period in ["First Gulf War","Second Gulf War","US-Iran"]:
                    avg = hist[hist["conflict_period"]==period]["close"].mean()
                    if not np.isnan(avg):
                        comparison_rows.append({
                            "Ticker":    t,
                            "Region":    sig_row["region"],
                            "Period":    period,
                            "Hist Avg":  round(avg, 2),
                            "Live":      sig_row["latest_close"],
                            "Diff %":    round(
                                (sig_row["latest_close"] - avg) / avg * 100, 1
                            )
                        })

            if comparison_rows:
                comp_df = pd.DataFrame(comparison_rows)
                sel_comp = comp_df[comp_df["Ticker"] == selected]
                if not sel_comp.empty:
                    fig_comp = px.bar(
                        sel_comp, x="Period", y="Diff %",
                        color="Period",
                        color_discrete_map=CONFLICT_COLORS,
                        title=f"{selected}: Live Price vs Historical During-Conflict Avg",
                        labels={"Diff %":"Difference from historical avg (%)"}
                    )
                    fig_comp.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")
                    fig_comp.update_layout(**LAYOUT, height=320)
                    st.plotly_chart(fig_comp, use_container_width=True)
# ════════════════════════════════════════════════════════════
# PAGE 5 — RESEARCH FINDINGS
# ════════════════════════════════════════════════════════════
elif page == "Analysis Findings":
    st.markdown("# Analysis Findings")
    st.markdown("""
    <p style='color:#6B7FA3;max-width:800px'>
    Our three research questions examined using historical stock and oil price data
    across the First Gulf War (1990–1991), Second Gulf War (2003), and current
    US–Iran tensions (2019–2026). Dashboards built in Tableau from pipeline outputs.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    # ── CSS for finding cards ────────────────────────────────
    st.markdown("""
    <style>
    .finding-card {
        background: linear-gradient(135deg, #0F1623 0%, #141D2E 100%);
        border: 1px solid #1E2D45;
        border-left: 4px solid #E63946;
        border-radius: 10px;
        padding: 22px 28px;
        margin: 16px 0 24px 0;
    }
    .finding-card.q2 { border-left-color: #F4A261; }
    .finding-card.q3 { border-left-color: #4B9FFF; }
    .finding-label {
        color: #6B7FA3;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .finding-question {
        color: #E8EDF5;
        font-family: 'Space Mono', monospace;
        font-size: 1.0rem;
        font-weight: 700;
        margin-bottom: 14px;
        line-height: 1.5;
    }
    .finding-answer {
        color: #A8B8D0;
        font-size: 0.92rem;
        line-height: 1.8;
    }
    .finding-answer b { color: #E8EDF5; }
    .finding-answer .highlight-red   { color: #E63946; font-weight: 600; }
    .finding-answer .highlight-amber { color: #F4A261; font-weight: 600; }
    .finding-answer .highlight-blue  { color: #4B9FFF; font-weight: 600; }
    .dashboard-label {
        color: #6B7FA3;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin: 18px 0 10px 0;
    }
    .tableau-note {
        color: #3A5070;
        font-size: 0.78rem;
        font-style: italic;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    import streamlit.components.v1 as components

    def tableau_embed(name, height=850):
        html = f"""
        <div class='tableauPlaceholder' style='position:relative;width:100%'>
          <object class='tableauViz' style='display:none;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
            <param name='embed_code_version' value='3' />
            <param name='site_root' value='' />
            <param name='name' value='{name}' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='animate_transition' value='yes' />
            <param name='display_static_image' value='yes' />
            <param name='display_spinner' value='yes' />
            <param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' />
            <param name='language' value='en-US' />
          </object>
        </div>
        <script type='text/javascript'>
          var divElement = document.querySelector('.tableauPlaceholder');
          var vizElement = divElement.getElementsByTagName('object')[0];
          vizElement.style.width  = '100%';
          vizElement.style.height = '{height}px';
          var scriptElement = document.createElement('script');
          scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
          vizElement.parentNode.insertBefore(scriptElement, vizElement);
        </script>
        """
        components.html(html, height=height + 20, scrolling=False)

    # ── QUESTION 1 ───────────────────────────────────────────
    st.markdown("""
    <div class='finding-card'>
        <div class='finding-label'>Analytics Question 1</div>
        <div class='finding-question'>
            To what extent can oil price and oil &amp; gas stock patterns observed
            during the First Gulf War predict market behavior during the Second Gulf War?
        </div>
        <div class='finding-answer'>
            Our analysis of the <b>7 companies with data across both conflicts</b>
            (BP, COP, CVX, SHEL, SU, TTE, XOM) reveals a
            <span class='highlight-red'>moderate correlation of r = 0.57</span>
            between average daily returns during GW1 and GW2 — suggesting partial
            but not strong predictive power. <b>SHEL was the most consistent</b>
            performer, posting positive returns in both conflicts (+0.09%/day GW1,
            +0.16%/day GW2). <b>CVX was the most inconsistent</b>, slightly negative
            in GW1 and strongly negative in GW2 (-0.14%/day). Critically, the oil
            price environment differed dramatically:
            <span class='highlight-red'>WTI spiked +48% during GW1</span> (from
            $19.62 to $29.04/barrel) but only
            <span class='highlight-amber'>+3.2% during GW2</span> ($27.67 to
            $28.56/barrel) — meaning stocks in GW2 faced a very different
            macroeconomic backdrop, which limits how well GW1 patterns transfer.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='dashboard-label'>Dashboard 1 — Gulf War I vs II: Can History Predict History?</div>",
                unsafe_allow_html=True)

    tableau_embed("dashboards_acc3/Dashboard1", height=850)

    st.markdown("<p class='tableau-note'>Charts: Stock % price change during conflict (GW1 vs GW2) · Avg daily return by ticker · WTI oil price indexed timeline · Price movement comparison</p>",
                unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    # ── QUESTION 2 ───────────────────────────────────────────
    st.markdown("""
    <div class='finding-card q2'>
        <div class='finding-label'>Analytics Question 2</div>
        <div class='finding-question'>
            How do oil prices and stock markets respond before, during,
            and after major geopolitical conflict events?
        </div>
        <div class='finding-answer'>
            A clear <b>before/during/after pattern</b> exists, though it differs
            by conflict. In <span class='highlight-red'>Gulf War I</span>, volatility
            was highest <i>before</i> the conflict (3.57) and remained elevated
            <i>during</i> (3.54), then dropped <i>after</i> — a classic
            uncertainty-to-resolution pattern. In
            <span class='highlight-amber'>Gulf War II</span>, volatility peaked
            <i>before</i> the invasion (1.88) and dropped during and after,
            suggesting markets had already priced in the conflict. The
            <span class='highlight-blue'>US–Iran</span> Soleimani period showed
            the sharpest single-period return drop at
            <b>-0.18%/day during the conflict window</b>. The heatmap reveals
            <b>CVE (-0.811) and CVX (-0.597)</b> as the most severely impacted
            during the Soleimani event, while <b>CNQ (+0.178) and SHEL (+0.067)</b>
            proved most resilient.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='dashboard-label'>Dashboard 2 — Market Behavior Before, During &amp; After Geopolitical Conflict</div>",
                unsafe_allow_html=True)

    tableau_embed("dashboards_acc2/Dashboard2", height=850)

    st.markdown("<p class='tableau-note'>Charts: Stock volatility by phase · Avg WTI oil price by phase · Daily return heatmap by ticker &amp; phase · Avg return by conflict phase</p>",
                unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    # ── QUESTION 3 ───────────────────────────────────────────
    st.markdown("""
    <div class='finding-card q3'>
        <div class='finding-label'>Analytics Question 3</div>
        <div class='finding-question'>
            Can patterns learned from historical conflicts provide insight into
            potential market behavior during the current US–Iran tensions?
        </div>
        <div class='finding-answer'>
            Historical patterns provide <b>partial but meaningful guidance</b>.
            Since the April 2024 Israel-Iran consulate strike,
            <span class='highlight-blue'>Canadian stocks have outperformed</span>
            — with <b>SU posting the highest avg daily return (+0.068%)</b> and
            SHEL second (+0.045%) — consistent with the historical pattern of
            Canadian and international majors being more resilient than US
            counterparts during Middle East escalation. However,
            <span class='highlight-red'>Saudi Aramco (2222.SR) and COP have
            dipped slightly</span>, diverging from what historical patterns would
            predict. Volatility during the Soleimani event was
            <b>lower than during both Gulf Wars</b> (1.07 vs 3.54 GW1 / 1.35 GW2),
            suggesting markets have become more resilient to Middle East conflict
            signals over time — though the June 2025 US strikes on Iranian nuclear
            facilities represent a significant new escalation whose full market
            impact remains unfolding.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='dashboard-label'>Dashboard 3 — Current US–Iran Tensions: What History Suggests</div>",
                unsafe_allow_html=True)

    tableau_embed("dashboards_acc/Dashboard3", height=900)

    st.markdown("<p class='tableau-note'>Charts: Oil &amp; gas stock prices 2019–2025 with event annotations · Avg return since April 2024 · Return by region · Conflict-period volatility historical context</p>",
                unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D45'>", unsafe_allow_html=True)

    # ── KEY TAKEAWAYS SUMMARY ────────────────────────────────
    st.markdown("#### Overall Takeaways")
    st.markdown("""
    <div style='color:#A8B8D0;line-height:2.2;max-width:800px'>
    ▸ GW1 patterns have <b style='color:#E63946'>moderate predictive value (r = 0.57)</b>
      for GW2 — enough to inform expectations but not reliable enough to trade on<br>
    ▸ The <b style='color:#F4A261'>before/during/after market pattern is real</b>
      but varies significantly by conflict — GW1 spiked oil, GW2 did not<br>
    ▸ <b style='color:#4B9FFF'>Canadian stocks (SU, CNQ)</b> consistently show the
      most resilience across all three conflict periods<br>
    ▸ Volatility during geopolitical events has <b style='color:#2A9D8F'>decreased
      over time</b> — markets appear to price in conflict faster than they did in 1990<br>
    ▸ <b style='color:#E63946'>Data limitation:</b> EIA oil data ends in 2008;
      US–Iran oil figures are approximated from public WTI records
    </div>
    """, unsafe_allow_html=True)