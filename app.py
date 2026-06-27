"""
Suvira Intelligence — Global Chemical Market Intelligence System
AI/ML-Driven Demand Forecasting and Price Trend Analysis
Suvira Energy
"""

import os
import sys
import io
import zipfile
import warnings
import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

from datetime import date
import feedparser
import urllib.parse
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

# from models.arima_model import run_arima_pipeline
# from models.prophet_model import run_prophet_pipeline
# from models.lstm_model import run_lstm_pipeline
from models.price_prediction import run_price_prediction_pipeline
from models.price_prediction2 import run_arima_pipeline
# from models.anomaly_detection import run_anomaly_pipeline, run_anomaly_all_chemicals
# from models.clustering import run_clustering_pipeline

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
_ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo copy.png")

st.set_page_config(
    page_title="Suvira Intelligence",
    page_icon=_ICON_PATH if os.path.exists(_ICON_PATH) else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Auto refresh per 24h
st_autorefresh(
    interval=300000,  #5min
    key="refresh"
)

st.markdown("""
<style>

/* reduce top padding of main page */
.block-container {
    padding-top: 0.5rem !important;
}

# /* optional: make title tighter */
# h1 {
#     margin-top: 0 !important;
# }

</style>
""", unsafe_allow_html=True)

#font style
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
/* Poppins */
body, h1, h2, h3, h4, h5, h6, p, label {
    font-family: 'Poppins', sans-serif !important;
}

/* Remove stray Streamlit icon label text */
button[kind="header"] span {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# WITH BORDERS 
# st.markdown("""
# <style>

# /* Actual sidebar button */
# section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
#     padding: 0px 6px !important;
#     min-height: 24px !important;
#     height: 24px !important;

#     border-radius: 5px !important;   /* square corners */
#     border: 1.5px solid #51A5DC !important;

#     background: transparent !important;
#     box-shadow: none !important;
# }

# /* text inside button */
# section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] p {
#     font-size: 11px !important;
#     line-height: 1 !important;
#     margin: 0 !important;

#     color: #2f3340 !important;       /* fixed text color */
#     font-family: 'Poppins', sans-serif !important;
# }

# /* hover = border only */
# section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
#     border-color: #76C442 !important;
#     background: transparent !important;
# }

# /* keep text same on hover */
# section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover p {
#     color: #2f3340 !important;
# }

# /* focus/click */
# section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:focus {
#     border-color: #76C442 !important;
#     box-shadow: none !important;
# }

# /* spacing */
# section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
#     margin-bottom: 2px !important;
# }

# </style>
# """, unsafe_allow_html=True)

st.markdown("""
<style>

/* ---------------- Sidebar buttons ---------------- */
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
    background: rgba(81,165,220,0.18) !important;

    border: none !important;
    box-shadow: none !important;

    padding: 0.08rem 0.18rem !important;

    min-height: 24px !important;
    height: 24px !important;

    border-radius: 8px !important;

    width: auto !important;
    min-width: 200px !important;
}

/* ---------------- Text ---------------- */
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] p {
    font-size: 12px !important;
    color: #2f3340 !important;

    margin: 0 !important;
    line-height: 1 !important;
}

/* ---------------- Hover ---------------- */
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
    background: rgba(118,196,66,0.22) !important;
}

/* keep text dark on hover */
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover p {
    color: #2f3340 !important;
}

/* ---------------- Selected / focus ---------------- */
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:focus {
    background: rgba(118,196,66,0.22) !important;
    box-shadow: none !important;
}

/* keep text dark when selected */
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:focus p {
    color: #2f3340 !important;
}

/* while clicking */
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:active p {
    color: #2f3340 !important;
}

/* ---------------- Row spacing ---------------- */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
    margin-bottom: 0px !important;
}

/* ---------------- Logo + button spacing ---------------- */
section[data-testid="stSidebar"] div[data-testid="column"] {
    gap: 0.05rem !important;

    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

</style>
""", unsafe_allow_html=True)


#colour of the buttons
st.markdown("""
<style>

/* Default button border = logo blue */
section[data-testid="stSidebar"] div.stButton > button {
    border: 1.5px solid #0D47C1 !important;
    border-radius: 12px !important;
}

/* Default text */
section[data-testid="stSidebar"] div.stButton button p {
    color: #2f3340 !important;
    font-size: 14px !important;
}

/* Hover = logo green */
section[data-testid="stSidebar"] div.stButton > button:hover {
    border-color: #76C442 !important;
}

section[data-testid="stSidebar"] div.stButton > button:hover p {
    color: #76C442 !important;
}

/* Keyboard focus / clicked */
section[data-testid="stSidebar"] div.stButton > button:focus {
    border-color: #76C442 !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] div.stButton > button:focus p {
    color: #76C442 !important;
}

</style>
""", unsafe_allow_html=True)


# selectbox styling
st.markdown("""
<style>
/* Selectbox border - blue */
[data-baseweb="select"] > div {
    border-color: #1B4FD8 !important;
}

/* Hover - multiple selectors to catch it */
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] li:hover,
[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] [aria-selected="false"]:hover,
ul[role="listbox"] li:hover {
    background-color: #D4EDD9 !important;
}
</style>
""", unsafe_allow_html=True)

#below dashboard heading
st.markdown("""
<style>

/* KPI label (Acetic Acid etc.) */
div[data-testid="stMetricLabel"] p {
    font-size: 14px !important;
    margin-bottom: 0 !important;
}

/* KPI value ($700.0) */
div[data-testid="stMetricValue"] {
    font-size: 20px !important;
    line-height: 1 !important;
}

/* KPI delta (+6.1%) */
div[data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* reduce spacing inside metric box */
div[data-testid="metric-container"] {
    padding: 0.2rem 0.3rem !important;
    margin: 0 !important;
}

/* optional: reduce gap between columns */
div[data-testid="column"] {
    padding-left: 0.15rem !important;
    padding-right: 0.15rem !important;
}

</style>
""", unsafe_allow_html=True)

#dashboard boxes
#dashboard boxes
st.markdown("""
<style>
.dash-tile {
    background: #f5f6f8;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 20px 22px 16px 22px;
    margin-bottom: 14px;
    transition: box-shadow 0.25s ease, border-color 0.25s ease;
}
.dash-tile:hover {
    box-shadow: 0 4px 18px rgba(0,0,0,0.07);
    border-color: #d0d4dc;
}
</style>
""", unsafe_allow_html=True)

#slider colour
st.markdown("""
<style>
/* Thumb dot - black */
.st-emotion-cache-1dj3ksd {
    background-color: #2f3340 !important;
    border-color: #2f3340 !important;
}
/* Filled track - black */
div.st-an.st-ap.st-aq.st-ao.st-ct {
    background: none !important;
    background-color: #2f3340 !important;
    background-image: none !important;
}
/* Unfilled track - keep grey */
div.st-an.st-ap.st-aq.st-ao.st-cu {
    background-color: rgba(151, 166, 195, 0.25) !important;
}
</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------------------------------
# DATA PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSET_DIR = os.path.join(BASE_DIR, "assets")

SENTIMENT_CSV = os.path.join(DATA_DIR, "FINAL_sentiment_macro.csv")
PRICE_CSV = os.path.join(DATA_DIR, "final_2020_2026.csv")

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
# @st.cache_data(show_spinner=False)
def load_data():
    df_sent = pd.read_csv(SENTIMENT_CSV, parse_dates=["date"])
    df_price = pd.read_csv(PRICE_CSV, parse_dates=["date"])

    df_sent = df_sent.sort_values(["chemical", "date"]).reset_index(drop=True)
    df_price = df_price.sort_values(["chemical", "date"]).reset_index(drop=True)

    return df_sent, df_price


#Auto-updating the dataset according to predicted price of current month
# Auto-updating the dataset according to predicted price of current month
def auto_update_monthly_prices(df_historical):
    """
    Appends predicted rows directly into FINAL_sentiment_macro.csv.
    - Uses each chemical's own latest date (fixes Dec cutoff bug)
    - Predicts one month at a time sequentially
    - Rolling window: keeps only last KEEP_YEARS of data
    - One dataset only — no separate predicted_prices.csv
    """
    KEEP_YEARS = 10

    today         = pd.Timestamp.today()
    last_complete = (pd.Timestamp(today.year, today.month, 1)
                     - pd.DateOffset(months=1))

    # Always re-read CSV fresh from disk to get latest saved state
    df = pd.read_csv(SENTIMENT_CSV, parse_dates=["date"])
    df = df.sort_values(["chemical", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    if "is_predicted" not in df.columns:
        df["is_predicted"] = False

    chemicals = sorted(df["chemical"].dropna().unique())

    all_up_to_date = all(
        df[df["chemical"] == chem]["date"].max() >= last_complete
        for chem in chemicals
        if not df[df["chemical"] == chem].empty
    )
    if all_up_to_date:
        return df

    min_latest = min(
        df[df["chemical"] == chem]["date"].max()
        for chem in chemicals
        if not df[df["chemical"] == chem].empty
    )
    total_months = (
        (last_complete.year  - min_latest.year)  * 12
        + last_complete.month - min_latest.month
    )

    new_records = []

    for m in range(1, total_months + 1):
        month_rows = []

        for chem in chemicals:
            chem_df      = df[df["chemical"] == chem]
            chem_latest  = chem_df["date"].max()
            next_month   = chem_latest + pd.DateOffset(months=1)
            target_month = pd.Timestamp(next_month.year, next_month.month, 1)

            if target_month > last_complete:
                continue

            if chem == "Sulphur":
                result = run_arima_pipeline(df, chem, steps=1)
            else:
                result = run_price_prediction_pipeline(df, chem, steps=1)
            forecast   = result["forecast"]
            latest_row = chem_df.iloc[-1]

            row = {
                "date":              target_month,
                "chemical":          chem,
                "price_usd_per_ton": forecast["predictions"][0],
                "netWgt":            latest_row.get("netWgt",            np.nan),
                "fobvalue":          latest_row.get("fobvalue",          np.nan),
                "sentiment_score":   latest_row.get("sentiment_score",   0),
                "crude_oil_price":   latest_row.get("crude_oil_price",   np.nan),
                "natural_gas_price": latest_row.get("natural_gas_price", np.nan),
                "is_predicted":      True,
            }
            month_rows.append(row)
            new_records.append(row)

        if month_rows:
            month_df = pd.DataFrame(month_rows)
            month_df["date"] = pd.to_datetime(month_df["date"])
            df = pd.concat([df, month_df], ignore_index=True)
            df = (df
                  .sort_values(["chemical", "date"])
                  .drop_duplicates(subset=["chemical", "date"], keep="last")
                  .reset_index(drop=True))

    if new_records:
        df_new = pd.DataFrame(new_records)
        df_new["date"] = pd.to_datetime(df_new["date"])

        existing_cols = pd.read_csv(SENTIMENT_CSV, nrows=0).columns.tolist()
        if "is_predicted" not in existing_cols:
            existing_full = pd.read_csv(SENTIMENT_CSV, parse_dates=["date"])
            existing_full["is_predicted"] = False
            existing_full.to_csv(SENTIMENT_CSV, index=False)
            existing_cols = existing_full.columns.tolist()

        for col in existing_cols:
            if col not in df_new.columns:
                df_new[col] = np.nan
        df_new = df_new[existing_cols]

        df_new.to_csv(SENTIMENT_CSV, mode="a", header=False, index=False)

        cutoff_date = pd.Timestamp(
            today.year - KEEP_YEARS,
            today.month,
            1
        )

        df_full = pd.read_csv(SENTIMENT_CSV, parse_dates=["date"])
        df_full["date"] = pd.to_datetime(df_full["date"])

        rows_before = len(df_full)
        df_full     = df_full[df_full["date"] >= cutoff_date]

        if len(df_full) < rows_before:
            df_full.to_csv(SENTIMENT_CSV, index=False)
            df = df[df["date"] >= cutoff_date].reset_index(drop=True)

    return df

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar(chemicals):
    with st.sidebar:
        logo_path = os.path.join(ASSET_DIR, "logo.png")
        if os.path.exists(logo_path):
            img = Image.open(logo_path)
            st.image(img, use_container_width=True)

        st.title("Suvira Intelligence")
        st.markdown(" *Global Chemical Market Intelligence System*")
        st.caption(f"## {datetime.date.today().strftime('%A | %d %b, %Y')}")

        st.markdown("---")

        pages = [
            "Dashboard",
            "Price Prediction",
            # "Demand Forecasting",
            # "Anomaly Detection",
            # "Sentiment & Macro",
            # "Clustering Analysis",
            "Market Insights",
            # "Download",
        ]

        if "page" not in st.session_state:
            st.session_state.page = pages[0]

        filled_logo = os.path.join(ASSET_DIR, "logo_filled.png")


        def set_page(page_name):
            st.session_state.page = page_name


        for p in pages:
            col1, col2 = st.columns([1, 8])

            with col1:
                st.image(filled_logo, width=18)

            with col2:
                st.button(
                    p,
                    key=f"nav_{p}",
                    use_container_width=True,
                    on_click=set_page,
                    args=(p,),
                )

        page = st.session_state.page

        st.markdown("---")

        selected_chemical = chemicals[0]
        steps = 6

        if page in ["Price Prediction", "Market Insights"]:

            st.markdown("**Select Chemical**")
            selected_chemical = st.selectbox(
                "Chemical",
                chemicals,
                label_visibility="collapsed",
            )

            st.markdown("**Forecast Horizon**")
            steps = st.slider(
                "Months ahead",
                min_value=1,
                max_value=12,
                value=6
            )

            st.markdown("---")
        # st.caption(f"Data: 2020 – {datetime.date.today().year}")
        st.caption("@Suvira Intelligence • 2026")

    return page, selected_chemical, steps


# ---------------------------------------------------------------------------
# COLOUR PALETTE
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#1565C0",
    "accent": "#43A047",
    "warn": "#F57C00",
    "danger": "#C62828",
    "neutral": "#546E7A",
    "bg": "#F4F6F9",
}

CHEM_COLORS = px.colors.qualitative.Set2


# ---------------------------------------------------------------------------
# PAGE: DASHBOARD
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_macro_live():
    import requests

    def _fetch_yahoo_monthly(ticker: str) -> pd.DataFrame:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "period1": int(pd.Timestamp("2020-01-01").timestamp()),
            "period2": int(pd.Timestamp.now().timestamp()),
            "interval": "1mo",
            "events":   "history",
        }
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SuviraBot/1.0)"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        result    = r.json()["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes     = result["indicators"]["quote"][0]["close"]
        dates      = (pd.to_datetime(timestamps, unit="s")
                        .to_period("M")
                        .to_timestamp())        
        return (pd.DataFrame({"date": dates, "close": closes})
                  .dropna()
                  .reset_index(drop=True))

    try:
        crude  = _fetch_yahoo_monthly("CL=F").rename(columns={"close": "crude_oil_price"})
        natgas = _fetch_yahoo_monthly("NG=F").rename(columns={"close": "natural_gas_price"})
        macro_df = (crude.merge(natgas, on="date", how="outer")
                         .sort_values("date")
                         .dropna()
                         .reset_index(drop=True))
        return macro_df, None
    except Exception as e:
        return None, str(e)

def page_dashboard(df, chemicals):
    st.title("Market Intelligence Dashboard")
    print("\n")

    cols = st.columns(len(chemicals))
    for i, chem in enumerate(chemicals):
        chem_df = df[df["chemical"] == chem].sort_values("date")
        if len(chem_df) < 2:
            continue  

        cur_val = chem_df.iloc[-1]["price_usd_per_ton"]
        prv_val = chem_df.iloc[-2]["price_usd_per_ton"]

        if prv_val == 0 or pd.isna(prv_val) or pd.isna(cur_val):
            continue  

        delta = ((cur_val - prv_val) / prv_val) * 100
        cols[i].metric(
            chem,
            f"${cur_val:,.1f}",
            f"{delta:+.1f}% MoM",
            delta_color="inverse"
        )

    ###live news updates
    # @st.cache_data(ttl=3600)
    @st.cache_data(ttl=21600, show_spinner=False)
    def fetch_market_news():

        import feedparser
        import urllib.parse

        search_terms = {

            # -------- Chemicals --------
            "sulphur price": "Sulphur",
            "sulphuric acid price": "Sulphuric Acid",
            "urea fertilizer price": "Urea",
            "potash price": "KCL",
            "acetic acid chemical price": "Acetic Acid",
            "poly aluminium chloride price": "PAC",
            "water treatment chemicals price": "Water Treatment Chemicals",
            "formic acid chemical price": "Formates",

            # -------- Energy --------
            "crude oil price": "Crude Oil",
            "opec oil supply": "Crude Oil",
            "natural gas price": "Natural Gas",

            # -------- Metals --------
            "gold price": "Gold",
            "silver price": "Silver",

            # -------- Macro --------
            "global shipping disruption": "Macro",
            "china chemical exports": "Macro",
            "india fertilizer subsidy": "Macro",
            "trump oil prices": "Macro",
        }

        news_items = []
        seen = set()

        for term, tag in search_terms.items():

            encoded = urllib.parse.quote(term)

            rss_url = (
                f"https://news.google.com/rss/search?q={encoded}+when:1d"
                "&hl=en-IN&gl=IN&ceid=IN:en"
            )

            feed = feedparser.parse(rss_url)

            
            entries = feed.entries[:2]

            for entry in entries:

                title = entry.title
                source = entry.link

                if title not in seen:

                    seen.add(title)

                    news_items.append({
                        "title": title,
                        "tag": tag,
                        "source": source,
                    })

        return news_items[:12]
    #####

    st.markdown("---")
    # left = chart | right = news tile
    # col1, col2 = st.columns([3.2, 1], gap="medium")
    # col1, spacer, col2 = st.columns([3.2, 0.25, 1])
    col1, spacer, col2 = st.columns([3.2, 0.2, 1])

    with spacer:
        st.markdown(
            '<div style="border-left: 1px solid #e5e7eb; height: 380px; margin-top: 38px;"></div>',
            unsafe_allow_html=True,
        )

    #automated price trends chart
    ##########To update

    with col1:
        st.markdown("#### Price Trends")

        # fig = go.Figure()

        # for i, chem in enumerate(chemicals):
        #     chem_df = df[df["chemical"] == chem].sort_values("date")
        #     # chem_df = build_plot_df(chem)

        #     # actual line
        #     fig.add_trace(go.Scatter(
        #     x=chem_df["date"],
        #     y=chem_df["price_usd_per_ton"],
        #     name=chem,
        #     mode="lines",

        #     line=dict(
        #         color=CHEM_COLORS[i % len(CHEM_COLORS)],
        #         width=2,
        #     ),

        #     legendgroup=chem,
        #     showlegend=False,

        #     hovertemplate=
        #         "%{fullData.name}: %{y:,.2f}<extra></extra>",
        # ))

        #     # square legend only
        #     fig.add_trace(go.Scatter(
        #         x=[None],
        #         y=[None],

        #         name=chem,
        #         mode="markers",

        #         marker=dict(
        #             symbol="square",
        #             size=12,
        #             color=CHEM_COLORS[i % len(CHEM_COLORS)],
        #         ),

        #         legendgroup=chem,

        #         showlegend=True,
        #         hoverinfo="skip",
        #     ))

        fig = go.Figure()

        for i, chem in enumerate(chemicals):
            chem_df = df[df["chemical"] == chem].sort_values("date")
            color = CHEM_COLORS[i % len(CHEM_COLORS)]

    
            if "is_predicted" in chem_df.columns:
                actual_df = chem_df[chem_df["is_predicted"] == False]
                pred_df   = chem_df[chem_df["is_predicted"] == True]
            else:
                actual_df = chem_df
                pred_df   = pd.DataFrame()


            fig.add_trace(go.Scatter(
                x=actual_df["date"],
                y=actual_df["price_usd_per_ton"],
                name=chem,
                mode="lines",
                line=dict(color=color, width=2),
                legendgroup=chem,
                showlegend=False,
                hovertemplate="%{fullData.name}: %{y:,.2f}<extra></extra>",
            ))


            if not pred_df.empty and not actual_df.empty:
                connector = pd.concat([actual_df.iloc[[-1]], pred_df])
                fig.add_trace(go.Scatter(
                    x=connector["date"],
                    y=connector["price_usd_per_ton"],
                    name=chem + " (est.)",
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    legendgroup=chem,
                    showlegend=False,
                    hovertemplate="Forecast: %{y:,.2f}<extra></extra>",
                ))

            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                name=chem, mode="markers",
                marker=dict(symbol="square", size=12, color=color),
                legendgroup=chem,
                showlegend=True,
                hoverinfo="skip",
            ))

        fig.update_layout(
            height=350,
            autosize=True,

            xaxis_title="Date",
            yaxis_title="Price (USD/ton)",


            hovermode="x unified",

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,


                itemclick="toggleothers",
                itemdoubleclick="toggle",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("###### Market News & Macro Drivers")

        news_items = fetch_market_news()
        if not news_items:
            news_items = [
                {
                    "title": "No major updates today",
                    "tag": "Market Updates",
                    "source": "",
                }
            ]

        with st.container(height=350, border=False):
            st.markdown("<div style='padding-right:8px;'>", unsafe_allow_html=True)
            for item in news_items:
                tag_html = (
                    f"<div style='font-size:11px; color:#64748b; margin-top:6px; font-weight:500; letter-spacing:0.3px;'>"
                    f"{item['tag']}</div>"
                    if item["tag"] not in ["Market Update", "Market Updates", ""]
                    else ""
                )

                st.markdown(
                    f"""
                    <a href="{item['source']}" target="_blank" style="text-decoration:none; color:inherit;">
                    <div style="
                        padding:12px 14px;
                        margin-bottom:10px;
                        margin-right:6px;
                        margin-left:6px;
                        border-radius:14px;
                        background:#f8fafc;
                        border:none;
                        box-shadow: 0 0 14px rgba(0,0,0,0.18);
                        cursor:pointer;
                    ">
                        <div style="font-size:14px; font-weight:600; color:#1e293b; line-height:1.4;">
                            {item['title']}
                        </div>
                        {tag_html}
                    </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
    # with col2:
    #     st.subheader("Latest Month Prices")
    #     latest_prices = (
    #         df[df["date"] == latest_date][["chemical", "price_usd_per_ton"]]
    #         .sort_values("price_usd_per_ton", ascending=True)
    #     )
    #     fig2 = go.Figure(go.Bar(
    #         x=latest_prices["price_usd_per_ton"],
    #         y=latest_prices["chemical"],
    #         orientation="h",
    #         marker_color=COLORS["primary"],
    #     ))
    #     fig2.update_layout(height=350, xaxis_title="USD/ton")
    #     st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    # col3, col4 = st.columns(2)

    
    st.markdown("#### Macro Drivers Over Time")

    macro_live, err = fetch_macro_live()

    if macro_live is not None and not macro_live.empty:
        macro_df   = macro_live
        src_caption = "" #Live · WTI Crude & Henry Hub Gas via Yahoo Finance"
    else:

        macro_df   = df.drop_duplicates("date").sort_values("date")[
            ["date", "crude_oil_price", "natural_gas_price"]
        ]
        st.caption("Source:Yahoo Finance • Updates daily")

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

    fig3.add_trace(
        go.Scatter(
            x=macro_df["date"],
            y=macro_df["crude_oil_price"],
            name="Crude Oil ($/bbl)",
            mode="lines",
            line=dict(color=COLORS["danger"], width=2),
            legendgroup="crude",
            showlegend=False,
            hovertemplate="Crude Oil: $%{y:.2f}/bbl<extra></extra>",
        ),
        secondary_y=False,
    )

    fig3.add_trace(
        go.Scatter(
            x=[None], y=[None],
            name="Crude Oil ($/bbl)",
            mode="markers",
            marker=dict(symbol="square", size=12, color=COLORS["danger"]),
            legendgroup="crude",
            showlegend=True,
            hoverinfo="skip",
        ),
        secondary_y=False,
    )


    fig3.add_trace(
        go.Scatter(
            x=macro_df["date"],
            y=macro_df["natural_gas_price"],
            name="Natural Gas ($/MMBtu)",
            mode="lines",
            line=dict(color=COLORS["accent"], width=2),
            legendgroup="natgas",
            showlegend=False,
            hovertemplate="Natural Gas: $%{y:.2f}/MMBtu<extra></extra>",
        ),
        secondary_y=True,
    )

    fig3.add_trace(
        go.Scatter(
            x=[None], y=[None],
            name="Natural Gas ($/MMBtu)",
            mode="markers",
            marker=dict(symbol="square", size=12, color=COLORS["accent"]),
            legendgroup="natgas",
            showlegend=True,
            hoverinfo="skip",
        ),
        secondary_y=True,
    )
    fig3.update_layout(
        height=280,
        hovermode="x unified",
        xaxis=dict(hoverformat="%b %Y"),
        legend=dict(
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            ),
        )
    fig3.update_yaxes(
        title_text="Crude Oil ($/bbl)",
        secondary_y=False,
        title_font=dict(color=COLORS["danger"]),
        tickfont=dict(color=COLORS["danger"]),
        gridcolor="rgba(198,40,40,0.2)",
        gridwidth=1,
        griddash="solid",
        autorange=True,    
    )
    fig3.update_yaxes(
        title_text="Natural Gas ($/MMBtu)",
        secondary_y=True,
        title_font=dict(color=COLORS["accent"]),
        tickfont=dict(color=COLORS["accent"]),
        showgrid=True,
        gridcolor="rgba(67,160,71,0.2)",
        gridwidth=1,
        griddash="dash",
        autorange=True,    
    )
    # 
    fig3_html = fig3.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "displayModeBar": False,
            "responsive": True,
        }
    )

    wrapper_html = f"""
    <!DOCTYPE html><html><body style="margin:0;padding:0;">
    {fig3_html}
    <script>
    (function() {{
        function init() {{
            var plots = document.querySelectorAll('.plotly-graph-div');
            if (plots.length === 0) {{ setTimeout(init, 100); return; }}
            var gd = plots[0];
            var highlighted = null;

            gd.on('plotly_legendclick', function(data) {{
                var idx = data.curveNumber;
                var isCrude = (idx === 0 || idx === 1);
                var clickedGroup = isCrude ? 'crude' : 'natgas';

                if (highlighted === clickedGroup) {{
                    // second click → reset everything
                    highlighted = null;
                    Plotly.restyle(gd, {{opacity: 1}}, [0, 1, 2, 3]);
                    Plotly.relayout(gd, {{
                        'yaxis.gridcolor':        'rgba(198,40,40,0.2)',
                        'yaxis.tickfont.color':   '#C62828',
                        'yaxis.title.font.color': '#C62828',
                        'yaxis2.gridcolor':        'rgba(67,160,71,0.2)',
                        'yaxis2.tickfont.color':   '#43A047',
                        'yaxis2.title.font.color': '#43A047',
                    }});
                }} else {{
                    highlighted = clickedGroup;
                    if (isCrude) {{
                        Plotly.restyle(gd, {{opacity: 1}},    [0, 1]);
                        Plotly.restyle(gd, {{opacity: 0.08}}, [2, 3]);
                        Plotly.relayout(gd, {{
                            'yaxis.gridcolor':        'rgba(198,40,40,0.2)',
                            'yaxis.tickfont.color':   '#C62828',
                            'yaxis.title.font.color': '#C62828',
                            'yaxis2.gridcolor':        'rgba(67,160,71,0.04)',
                            'yaxis2.tickfont.color':   'rgba(67,160,71,0.15)',
                            'yaxis2.title.font.color': 'rgba(67,160,71,0.15)',
                        }});
                    }} else {{
                        Plotly.restyle(gd, {{opacity: 0.08}}, [0, 1]);
                        Plotly.restyle(gd, {{opacity: 1}},    [2, 3]);
                        Plotly.relayout(gd, {{
                            'yaxis.gridcolor':        'rgba(198,40,40,0.04)',
                            'yaxis.tickfont.color':   'rgba(198,40,40,0.15)',
                            'yaxis.title.font.color': 'rgba(198,40,40,0.15)',
                            'yaxis2.gridcolor':        'rgba(67,160,71,0.2)',
                            'yaxis2.tickfont.color':   '#43A047',
                            'yaxis2.title.font.color': '#43A047',
                        }});
                    }}
                }}
                return false;
            }});
        }}
        init();
    }})();
    </script>
    </body></html>
    """

    components.html(wrapper_html, height=400, scrolling=False)
    st.caption(src_caption)

    macro_live, err = fetch_macro_live()

    # with col4:
    #     st.subheader("Market Sentiment by Chemical")
    #     sent_avg = (
    #         df.groupby("chemical")["sentiment_score"]
    #         .mean()
    #         .sort_values()
    #         .reset_index()
    #     )
    #     colors_sent = [COLORS["danger"] if v < 0 else COLORS["accent"]
    #                    for v in sent_avg["sentiment_score"]]
    #     fig4 = go.Figure(go.Bar(
    #         x=sent_avg["chemical"],
    #         y=sent_avg["sentiment_score"],
    #         marker_color=colors_sent,
    #     ))
    #     fig4.update_layout(height=280, yaxis_title="Avg Sentiment Score",
    #                        xaxis_tickangle=-30)
    #     st.plotly_chart(fig4, use_container_width=True)

    # st.markdown("---")
    # st.subheader("Price Correlation Heatmap")
    # pivot = df.pivot_table(index="date", columns="chemical", values="price_usd_per_ton")
    # corr = pivot.corr()
    # fig5 = px.imshow(
    #     corr, text_auto=".2f", color_continuous_scale="RdBu_r",
    #     zmin=-1, zmax=1, height=350,
    # )
    # fig5.update_layout(title="Chemical Price Correlation Matrix")
    # st.plotly_chart(fig5, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: PRICE PREDICTION
# ---------------------------------------------------------------------------
def page_price_prediction(df, chemical, steps):
    st.title("Price Prediction")
    # st.caption("Ensemble ML model (Random Forest + Gradient Boosting + Ridge) with macro drivers and sentiment")

    with st.spinner(f"Training ensemble model for {chemical}…"):
        if chemical == "Sulphur":
            result = run_arima_pipeline(df, chemical, steps=steps)
        else:
            result = run_price_prediction_pipeline(df, chemical, steps=steps)

    trained = result["trained"]
    forecast = result["forecast"]

    # col1, col2, col3, col4 = st.columns(4)
    # col1.metric("MAE", f"${trained['mae']:.2f}")
    # col2.metric("MAPE", f"{trained['mape']:.2f}%")
    # col3.metric("RMSE", f"${trained['rmse']:.2f}")
    # col4.metric("R²", f"{trained['r2']:.3f}")

    st.markdown("---")

    # ---------- Prediction Chart ----------
    st.markdown(f"#### Historical vs Predicted: {chemical}")

    hist = df[df["chemical"] == chemical].sort_values("date")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist["date"],
        y=hist["price_usd_per_ton"],
        name="Historical",
        mode="lines",
        line=dict(color=COLORS["primary"], width=2),
    ))

    fig.add_trace(go.Scatter(
        x=trained["dates_test"],
        y=trained["y_test"],
        name="Test Actual",
        mode="markers",
        marker=dict(color=COLORS["neutral"], size=8, symbol="circle"),
    ))

    fig.add_trace(go.Scatter(
        x=trained["dates_test"],
        y=trained["y_pred"],
        name="Test Predicted",
        mode="markers+lines",
        marker=dict(color=COLORS["accent"], size=8, symbol="diamond"),
        line=dict(color=COLORS["accent"], dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=forecast["dates"],
        y=forecast["predictions"],
        name=f"Forecast (+{steps}M)",
        mode="lines+markers",
        line=dict(color=COLORS["warn"], width=2.5),
        marker=dict(size=7),
    ))

    fig.add_trace(go.Scatter(
        x=list(forecast["dates"]) + list(forecast["dates"][::-1]),
        y=list(forecast["upper_ci"]) + list(forecast["lower_ci"][::-1]),
        fill="toself",
        fillcolor="rgba(245,124,0,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90% CI",
    ))

    fig.update_layout(
        height=380,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Price (USD/ton)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- Forecast Table ----------
    st.markdown("#### Forecast Table")

    fc_df = pd.DataFrame({
        "Month": [d.strftime("%b %Y") for d in forecast["dates"]],
        "Predicted ($/ton)": [f"${p:.2f}" for p in forecast["predictions"]],
        "Lower CI": [f"${p:.2f}" for p in forecast["lower_ci"]],
        "Upper CI": [f"${p:.2f}" for p in forecast["upper_ci"]],
    })

    rows_html = ""
    for _, row in fc_df.iterrows():
        rows_html += (
            "<tr style='border-bottom:1px solid #f1f5f9;'>"
            f"<td style='padding:10px 14px; color:#2f3340;'>{row['Month']}</td>"
            f"<td style='padding:10px 14px; color:#2f3340; font-weight:600; text-align:center;'>{row['Predicted ($/ton)']}</td>"
            f"<td style='padding:10px 14px; color:#64748b; text-align:center;'>{row['Lower CI']}</td>"
            f"<td style='padding:10px 14px; color:#64748b; text-align:center;'>{row['Upper CI']}</td>"
            "</tr>"
        )

    st.markdown(
        "<table style='width:100%; border-collapse:collapse; font-size:15px; margin-top:8px;'>"
        "<thead><tr style='background:#f1f5f9;'>"
        "<th style='text-align:left; padding:10px 14px; color:#64748b; font-weight:600; border-radius:8px 0 0 0;'>Month</th>"
        "<th style='text-align:center; padding:10px 14px; color:#64748b; font-weight:600;'>Predicted ($/ton)</th>"
        "<th style='text-align:center; padding:10px 14px; color:#64748b; font-weight:600;'>Lower CI</th>"
        "<th style='text-align:center; padding:10px 14px; color:#64748b; font-weight:600; border-radius:0 8px 0 0;'>Upper CI</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    last_hist = df[df["chemical"] == chemical]["price_usd_per_ton"].iloc[-1]
    last_fc = forecast["predictions"][-1]
    expected_change = ((last_fc - last_hist) / last_hist) * 100

    st.metric(
        f"Expected Price in {steps}M",
        f"${last_fc:.2f}",
        f"{expected_change:+.1f}% from current",
        delta_color="inverse"
    )

    # st.markdown("---")
    # col_c, col_d = st.columns(2)

    # with col_c:
    #     st.subheader("Feature Importance (Top 15)")
    #     fi = trained["feature_importance"].head(15)
    #     fig2 = go.Figure(go.Bar(
    #         x=fi.values, y=fi.index, orientation="h",
    #         marker_color=COLORS["primary"],
    #     ))
    #     fig2.update_layout(height=350, xaxis_title="Importance")
    #     st.plotly_chart(fig2, use_container_width=True)

    # with col_d:
    #     st.subheader("Residual Analysis")
    #     residuals = trained["y_test"].values - trained["y_pred"]
    #     fig3 = go.Figure()
    #     fig3.add_trace(go.Scatter(
    #         x=trained["dates_test"], y=residuals,
    #         mode="lines+markers",
    #         line=dict(color=COLORS["danger"]),
    #         name="Residuals",
    #     ))
    #     fig3.add_hline(y=0, line_dash="dash", line_color="gray")
    #     fig3.update_layout(height=350, yaxis_title="Residual (USD/ton)",
    #                        xaxis_title="Date")
    #     st.plotly_chart(fig3, use_container_width=True)

    # st.subheader("Cross-Validation Performance")
    # st.info(f"5-Fold Time-Series Cross-Validation MAPE: **{trained['cv_mape']:.2f}%**  "
    #         f"({'Excellent' if trained['cv_mape'] < 10 else 'Moderate' if trained['cv_mape'] < 20 else 'Needs Improvement'})")


# ---------------------------------------------------------------------------
# PAGE: DEMAND FORECASTING
# ---------------------------------------------------------------------------
# def page_demand_forecasting(df, chemical, steps):
#     st.title("Demand Forecasting")
#     st.caption("Multi-model forecasting: ARIMA · Prophet · Neural Network (LSTM-style MLP)")

#     model_tab = st.tabs(["ARIMA", "Prophet", "Neural Network (LSTM-style)", "Model Comparison"])

#     with model_tab[0]:
#         st.subheader(f"ARIMA Forecast — {chemical}")
#         with st.spinner("Fitting ARIMA model…"):
#             arima_result = run_arima_pipeline(df, chemical, steps=steps)

#         trained = arima_result["trained"]
#         col1, col2, col3 = st.columns(3)
#         col1.metric("MAE", f"${trained['mae']:.2f}")
#         col2.metric("MAPE", f"{trained['mape']:.2f}%")
#         col3.metric("ARIMA Order", str(trained["order"]))

#         fc = arima_result["forecast"]
#         hist = arima_result["historical_series"]

#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historical",
#                                  line=dict(color=COLORS["primary"], width=2)))
#         fig.add_trace(go.Scatter(x=trained["test"].index, y=trained["test_predictions"],
#                                  name="Test Fit", mode="lines",
#                                  line=dict(color=COLORS["accent"], dash="dot", width=2)))
#         fig.add_trace(go.Scatter(x=fc["dates"], y=fc["forecast"].values,
#                                  name="ARIMA Forecast", mode="lines+markers",
#                                  line=dict(color=COLORS["warn"], width=2.5)))
#         fig.add_trace(go.Scatter(
#             x=list(fc["dates"]) + list(fc["dates"][::-1]),
#             y=list(fc["upper_ci"]) + list(fc["lower_ci"][::-1]),
#             fill="toself", fillcolor="rgba(245,124,0,0.12)",
#             line=dict(color="rgba(255,255,255,0)"), name="95% CI",
#         ))
#         fig.update_layout(height=380, hovermode="x unified",
#                           xaxis_title="Date", yaxis_title="Price (USD/ton)")
#         st.plotly_chart(fig, use_container_width=True)

#         stat = arima_result["stationarity"]
#         st.info(f"ADF Stationarity Test: p-value = {stat['p_value']:.4f} — "
#                 f"{'Stationary' if stat['is_stationary'] else ' Non-stationary (differencing applied)'}")

#     with model_tab[1]:
#         st.subheader(f"Prophet Forecast — {chemical}")
#         with st.spinner("Fitting Prophet model with macro regressors…"):
#             prophet_result = run_prophet_pipeline(df, chemical, steps=steps)

#         trained_p = prophet_result["trained"]
#         col1, col2, col3 = st.columns(3)
#         col1.metric("MAE", f"${trained_p['mae']:.2f}")
#         col2.metric("MAPE", f"{trained_p['mape']:.2f}%")
#         regs = prophet_result["extra_regressors"]
#         col3.metric("Regressors", str(len(regs)) if regs else "0")

#         fc_p = prophet_result["forecast"]
#         pdf = prophet_result["prophet_df"]

#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=pdf["ds"], y=pdf["y"], name="Historical",
#                                  line=dict(color=COLORS["primary"], width=2)))
#         fig.add_trace(go.Scatter(x=fc_p["dates"], y=fc_p["forecast"],
#                                  name="Prophet Forecast", mode="lines+markers",
#                                  line=dict(color=COLORS["accent"], width=2.5)))
#         fig.add_trace(go.Scatter(
#             x=list(fc_p["dates"]) + list(fc_p["dates"][::-1]),
#             y=list(fc_p["upper_ci"]) + list(fc_p["lower_ci"][::-1]),
#             fill="toself", fillcolor="rgba(67,160,71,0.12)",
#             line=dict(color="rgba(255,255,255,0)"), name="95% CI",
#         ))
#         fig.update_layout(height=380, hovermode="x unified",
#                           xaxis_title="Date", yaxis_title="Price (USD/ton)")
#         st.plotly_chart(fig, use_container_width=True)

#         st.subheader("Prophet Components")
#         try:
#             comp = prophet_result["forecast"]["components"]
#             fig_comp = make_subplots(rows=2, cols=1, subplot_titles=["Trend", "Yearly Seasonality"])
#             if "trend" in comp.columns:
#                 fig_comp.add_trace(
#                     go.Scatter(x=comp["ds"], y=comp["trend"], name="Trend",
#                                line=dict(color=COLORS["primary"])), row=1, col=1)
#             if "yearly" in comp.columns:
#                 fig_comp.add_trace(
#                     go.Scatter(x=comp["ds"], y=comp["yearly"], name="Yearly",
#                                line=dict(color=COLORS["accent"])), row=2, col=1)
#             fig_comp.update_layout(height=380, showlegend=True)
#             st.plotly_chart(fig_comp, use_container_width=True)
#         except Exception:
#             st.info("Component decomposition not available for this chemical.")

#     with model_tab[2]:
#         st.subheader(f"Neural Network (LSTM-style MLP) Forecast — {chemical}")
#         with st.spinner("Training neural network…"):
#             lstm_result = run_lstm_pipeline(df, chemical, steps=steps)

#         trained_l = lstm_result["trained"]
#         col1, col2, col3 = st.columns(3)
#         col1.metric("MAE", f"${trained_l['mae']:.2f}")
#         col2.metric("MAPE", f"{trained_l['mape']:.2f}%")
#         col3.metric("RMSE", f"${trained_l['rmse']:.2f}")

#         fc_l = lstm_result["forecast"]
#         hist_l = lstm_result["historical_series"]

#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=hist_l.index, y=hist_l.values, name="Historical",
#                                  line=dict(color=COLORS["primary"], width=2)))
#         fig.add_trace(go.Scatter(
#             x=trained_l["dates_test"], y=trained_l["y_test_actual"],
#             name="Test Actual", mode="markers",
#             marker=dict(color=COLORS["neutral"], size=8, symbol="circle")))
#         fig.add_trace(go.Scatter(
#             x=trained_l["dates_test"], y=trained_l["y_test_predicted"],
#             name="Test Predicted", mode="markers+lines",
#             marker=dict(color=COLORS["accent"], size=8, symbol="diamond"),
#             line=dict(color=COLORS["accent"], dash="dash")))
#         fig.add_trace(go.Scatter(x=fc_l["dates"], y=fc_l["forecast"],
#                                  name="NN Forecast", mode="lines+markers",
#                                  line=dict(color=COLORS["danger"], width=2.5)))
#         fig.add_trace(go.Scatter(
#             x=list(fc_l["dates"]) + list(fc_l["dates"][::-1]),
#             y=list(fc_l["upper_ci"]) + list(fc_l["lower_ci"][::-1]),
#             fill="toself", fillcolor="rgba(198,40,40,0.10)",
#             line=dict(color="rgba(255,255,255,0)"), name="90% CI",
#         ))
#         fig.update_layout(height=380, hovermode="x unified",
#                           xaxis_title="Date", yaxis_title="Price (USD/ton)")
#         st.plotly_chart(fig, use_container_width=True)

#         st.subheader("Training Loss Curve")
#         loss_curve = trained_l["loss_curve"]
#         fig_loss = go.Figure(go.Scatter(y=loss_curve, mode="lines",
#                                         line=dict(color=COLORS["danger"], width=2),
#                                         name="Training Loss"))
#         fig_loss.update_layout(height=250, xaxis_title="Epoch",
#                                yaxis_title="Loss (MSE)")
#         st.plotly_chart(fig_loss, use_container_width=True)

#     with model_tab[3]:
#         st.subheader("Model Comparison")
#         st.write("Training all three models for comparison — this may take a moment…")

#         with st.spinner("Running all models…"):
#             arima_r = run_arima_pipeline(df, chemical, steps=steps)
#             prophet_r = run_prophet_pipeline(df, chemical, steps=steps)
#             lstm_r = run_lstm_pipeline(df, chemical, steps=steps)
#             ensemble_r = run_price_prediction_pipeline(df, chemical, steps=steps)

#         comparison = pd.DataFrame({
#             "Model": ["ARIMA", "Prophet", "Neural Network", "Ensemble"],
#             "MAE ($/ton)": [
#                 arima_r["trained"]["mae"],
#                 prophet_r["trained"]["mae"],
#                 lstm_r["trained"]["mae"],
#                 ensemble_r["trained"]["mae"],
#             ],
#             "MAPE (%)": [
#                 arima_r["trained"]["mape"],
#                 prophet_r["trained"]["mape"],
#                 lstm_r["trained"]["mape"],
#                 ensemble_r["trained"]["mape"],
#             ],
#             "RMSE ($/ton)": [
#                 arima_r["trained"]["rmse"],
#                 prophet_r["trained"]["rmse"],
#                 lstm_r["trained"]["rmse"],
#                 ensemble_r["trained"]["rmse"],
#             ],
#         })
#         st.dataframe(comparison.style.highlight_min(
#             subset=["MAE ($/ton)", "MAPE (%)", "RMSE ($/ton)"],
#             color="#c8e6c9"), use_container_width=True, hide_index=True)

#         fig_cmp = go.Figure()
#         months = [d.strftime("%b %Y") for d in ensemble_r["forecast"]["dates"]]
#         fig_cmp.add_trace(go.Scatter(x=months, y=arima_r["forecast"]["forecast"].values,
#                                      name="ARIMA", mode="lines+markers",
#                                      line=dict(color=COLORS["primary"])))
#         fig_cmp.add_trace(go.Scatter(x=months, y=prophet_r["forecast"]["forecast"],
#                                      name="Prophet", mode="lines+markers",
#                                      line=dict(color=COLORS["accent"])))
#         fig_cmp.add_trace(go.Scatter(x=months, y=lstm_r["forecast"]["forecast"],
#                                      name="Neural Network", mode="lines+markers",
#                                      line=dict(color=COLORS["danger"])))
#         fig_cmp.add_trace(go.Scatter(x=months, y=ensemble_r["forecast"]["predictions"],
#                                      name="Ensemble", mode="lines+markers",
#                                      line=dict(color=COLORS["warn"], width=3)))
#         fig_cmp.update_layout(height=350, xaxis_title="Month",
#                               yaxis_title="Predicted Price (USD/ton)",
#                               hovermode="x unified")
#         st.plotly_chart(fig_cmp, use_container_width=True)

#         best = comparison.loc[comparison["MAPE (%)"].idxmin(), "Model"]
#         st.success(f"Best model by MAPE: **{best}**")


# ---------------------------------------------------------------------------
# PAGE: ANOMALY DETECTION
# ---------------------------------------------------------------------------
# def page_anomaly(df, chemical):
#     st.title("Anomaly Detection")
#     st.caption("Isolation Forest + Statistical Z-score anomaly detection for price irregularities")

#     contamination = st.slider("Anomaly sensitivity (contamination rate)", 0.05, 0.20, 0.10, 0.01)

#     with st.spinner("Detecting anomalies…"):
#         result = run_anomaly_pipeline(df, chemical, contamination=contamination)

#     col1, col2, col3 = st.columns(3)
#     total = len(result["isolation_forest"])
#     col1.metric("Total Observations", total)
#     col2.metric("Isolation Forest Anomalies", result["n_iso_anomalies"])
#     col3.metric("Statistical Anomalies", result["n_stat_anomalies"])

#     st.markdown("---")
#     tab1, tab2 = st.tabs(["Isolation Forest", "Statistical (Z-score)"])

#     with tab1:
#         iso = result["isolation_forest"]
#         normal = iso[iso["anomaly"] == 0]
#         anomaly = iso[iso["anomaly"] == 1]

#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=normal["date"], y=normal["price_usd_per_ton"],
#             mode="lines", name="Normal",
#             line=dict(color=COLORS["primary"], width=2),
#         ))
#         fig.add_trace(go.Scatter(
#             x=anomaly["date"], y=anomaly["price_usd_per_ton"],
#             mode="markers", name="Anomaly",
#             marker=dict(color=COLORS["danger"], size=10, symbol="x"),
#         ))
#         fig.update_layout(height=380, xaxis_title="Date", yaxis_title="Price (USD/ton)",
#                           title=f"Isolation Forest Anomalies — {chemical}")
#         st.plotly_chart(fig, use_container_width=True)

#         st.subheader("Anomaly Score Distribution")
#         fig2 = go.Figure()
#         fig2.add_trace(go.Histogram(x=iso["anomaly_score"], nbinsx=30,
#                                     marker_color=COLORS["neutral"], name="Anomaly Score"))
#         fig2.add_vline(x=iso[iso["anomaly"] == 1]["anomaly_score"].max(),
#                        line_dash="dash", line_color=COLORS["danger"],
#                        annotation_text="Threshold")
#         fig2.update_layout(height=250, xaxis_title="Anomaly Score", yaxis_title="Count")
#         st.plotly_chart(fig2, use_container_width=True)

#         if len(anomaly) > 0:
#             st.subheader("Detected Anomaly Events")
#             disp = anomaly[["date", "price_usd_per_ton", "anomaly_score"]].copy()
#             disp["date"] = disp["date"].dt.strftime("%Y-%m")
#             disp.columns = ["Month", "Price (USD/ton)", "Anomaly Score"]
#             st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

#     with tab2:
#         stat = result["statistical"]
#         stat_anom = result["stat_anomalies"]

#         fig3 = go.Figure()
#         fig3.add_trace(go.Scatter(x=stat["date"], y=stat["price_usd_per_ton"],
#                                   name="Price", line=dict(color=COLORS["primary"], width=2)))
#         fig3.add_trace(go.Scatter(x=stat["date"], y=stat["rolling_mean"],
#                                   name="Rolling Mean (6M)", line=dict(color=COLORS["accent"],
#                                   dash="dash", width=1.5)))
#         fig3.add_trace(go.Scatter(
#             x=list(stat["date"]) + list(stat["date"][::-1]),
#             y=list(stat["upper_bound"]) + list(stat["lower_bound"].values[::-1]),
#             fill="toself", fillcolor="rgba(84,110,122,0.12)",
#             line=dict(color="rgba(255,255,255,0)"), name="±2.5σ Band",
#         ))
#         if len(stat_anom):
#             fig3.add_trace(go.Scatter(
#                 x=stat_anom["date"], y=stat_anom["price_usd_per_ton"],
#                 mode="markers", name="Z-score Anomaly",
#                 marker=dict(color=COLORS["warn"], size=10, symbol="triangle-up"),
#             ))
#         fig3.update_layout(height=380, xaxis_title="Date", yaxis_title="Price (USD/ton)",
#                            title=f"Statistical Anomalies — {chemical}")
#         st.plotly_chart(fig3, use_container_width=True)

#     st.markdown("---")
#     st.subheader("Anomaly Map — All Chemicals")
#     with st.spinner("Running cross-chemical anomaly scan…"):
#         all_anom = run_anomaly_all_chemicals(df, contamination=contamination)

#     if len(all_anom):
#         anom_counts = all_anom.groupby("chemical").size().reset_index(name="n_anomalies")
#         fig4 = go.Figure(go.Bar(
#             x=anom_counts["chemical"], y=anom_counts["n_anomalies"],
#             marker_color=COLORS["danger"],
#         ))
#         fig4.update_layout(height=250, xaxis_title="Chemical", yaxis_title="# Anomalies",
#                            xaxis_tickangle=-30)
#         st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: SENTIMENT & MACRO
# ---------------------------------------------------------------------------
# def page_sentiment(df, chemical):
    # st.title("Sentiment & Macro Drivers")
    # st.caption("Real sentiment scores and macro-economic drivers: Crude Oil & Natural Gas")

    # chem_df = df[df["chemical"] == chemical].sort_values("date")

    # col1, col2, col3 = st.columns(3)
    # col1.metric("Avg Sentiment Score", f"{chem_df['sentiment_score'].mean():.4f}")
    # col2.metric("Latest Crude Oil", f"${chem_df['crude_oil_price'].iloc[-1]:.2f}/bbl")
    # col3.metric("Latest Natural Gas", f"${chem_df['natural_gas_price'].iloc[-1]:.3f}/MMBtu")

    # st.markdown("---")
    # col_a, col_b = st.columns(2)

    # with col_a:
    #     st.subheader(f"Sentiment Score vs Price — {chemical}")
    #     fig = make_subplots(specs=[[{"secondary_y": True}]])
    #     fig.add_trace(go.Scatter(x=chem_df["date"], y=chem_df["price_usd_per_ton"],
    #                              name="Price (USD/ton)", line=dict(color=COLORS["primary"], width=2)),
    #                   secondary_y=False)
    #     fig.add_trace(go.Bar(x=chem_df["date"], y=chem_df["sentiment_score"],
    #                          name="Sentiment Score",
    #                          marker_color=[COLORS["accent"] if v >= 0 else COLORS["danger"]
    #                                        for v in chem_df["sentiment_score"]],
    #                          opacity=0.7),
    #                   secondary_y=True)
    #     fig.update_yaxes(title_text="Price (USD/ton)", secondary_y=False)
    #     fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    #     fig.update_layout(height=350, hovermode="x unified")
    #     st.plotly_chart(fig, use_container_width=True)

    # with col_b:
    #     st.subheader(f"Crude Oil vs Price — {chemical}")
    #     fig2 = go.Figure()
    #     fig2.add_trace(go.Scatter(
    #         x=chem_df["crude_oil_price"], y=chem_df["price_usd_per_ton"],
    #         mode="markers", text=chem_df["date"].dt.strftime("%Y-%m"),
    #         marker=dict(color=chem_df["date"].astype(np.int64),
    #                     colorscale="Blues", size=7, showscale=True,
    #                     colorbar=dict(title="Time →")),
    #     ))
    #     corr = chem_df["crude_oil_price"].corr(chem_df["price_usd_per_ton"])
    #     fig2.update_layout(height=350, xaxis_title="Crude Oil ($/bbl)",
    #                        yaxis_title="Price (USD/ton)",
    #                        title=f"Pearson r = {corr:.3f}")
    #     st.plotly_chart(fig2, use_container_width=True)

    # st.markdown("---")
    # st.subheader("Macro Driver Correlations — All Chemicals")
    # corr_data = []
    # for chem in df["chemical"].unique():
    #     c = df[df["chemical"] == chem]
    #     corr_data.append({
    #         "Chemical": chem,
    #         "Sentiment × Price": c["sentiment_score"].corr(c["price_usd_per_ton"]),
    #         "Crude Oil × Price": c["crude_oil_price"].corr(c["price_usd_per_ton"]),
    #         "Natural Gas × Price": c["natural_gas_price"].corr(c["price_usd_per_ton"]),
    #     })
    # corr_df = pd.DataFrame(corr_data).set_index("Chemical")
    # fig3 = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
    #                  zmin=-1, zmax=1, height=320)
    # fig3.update_layout(title="Correlation Matrix: Macro Drivers × Chemical Prices")
    # st.plotly_chart(fig3, use_container_width=True)

    # st.markdown("---")
    # st.subheader("Sentiment Trend — All Chemicals")
    # sent_pivot = df.pivot_table(index="date", columns="chemical",
    #                             values="sentiment_score")
    # fig4 = go.Figure()
    # for i, col in enumerate(sent_pivot.columns):
    #     fig4.add_trace(go.Scatter(x=sent_pivot.index, y=sent_pivot[col],
    #                               name=col, mode="lines",
    #                               line=dict(color=CHEM_COLORS[i % len(CHEM_COLORS)], width=1.5)))
    # fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    # fig4.update_layout(height=300, hovermode="x unified",
    #                    xaxis_title="Date", yaxis_title="Sentiment Score")
    # st.plotly_chart(fig4, use_container_width=True)

    # st.markdown("---")
    # st.subheader("Lagged Sentiment Effect on Price")
    # max_lag = 6
    # lag_corrs = {}
    # for lag in range(0, max_lag + 1):
    #     shifted_sent = chem_df["sentiment_score"].shift(lag)
    #     lag_corrs[f"Lag {lag}M"] = shifted_sent.corr(chem_df["price_usd_per_ton"])

    # lag_df = pd.DataFrame(list(lag_corrs.items()), columns=["Lag", "Correlation"])
    # fig5 = go.Figure(go.Bar(x=lag_df["Lag"], y=lag_df["Correlation"],
    #                         marker_color=[COLORS["accent"] if v >= 0 else COLORS["danger"]
    #                                       for v in lag_df["Correlation"]]))
    # fig5.update_layout(height=250, yaxis_title="Pearson Correlation",
    #                    title=f"Sentiment Lag Correlations — {chemical}")
    # st.plotly_chart(fig5, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: CLUSTERING
# ---------------------------------------------------------------------------
# def page_clustering(df):
#     st.title("Clustering Analysis")
#     st.caption("K-Means clustering to identify chemicals with similar price behavior patterns")

#     with st.spinner("Running clustering analysis…"):
#         result = run_clustering_pipeline(df)

#     kmeans = result["kmeans"]
#     feat_df = kmeans["result_df"]
#     best_k = result["best_k"]

#     col1, col2, col3 = st.columns(3)
#     col1.metric("Optimal Clusters", best_k)
#     col2.metric("Silhouette Score", f"{kmeans['silhouette_score']:.3f}")
#     col3.metric("PCA Variance Explained",
#                 f"{sum(kmeans['pca_explained_variance'])*100:.1f}%")

#     st.markdown("---")
#     col_a, col_b = st.columns(2)

#     with col_a:
#         st.subheader("Cluster Map (PCA Projection)")
#         fig = px.scatter(
#             feat_df, x="pca_x", y="pca_y", color="cluster",
#             text="chemical", size="mean_price",
#             color_continuous_scale="Set1",
#             labels={"pca_x": "PC1", "pca_y": "PC2"},
#             hover_data=["mean_price", "volatility", "trend"],
#             height=380,
#         )
#         fig.update_traces(textposition="top center", marker=dict(line=dict(width=1)))
#         fig.update_layout(coloraxis_showscale=False)
#         st.plotly_chart(fig, use_container_width=True)

#     with col_b:
#         st.subheader("Cluster Composition")
#         for i, profile in enumerate(result["cluster_profiles"]):
#             chems = ", ".join(profile["chemicals"])
#             avg_vol = profile.get("avg_volatility", 0) * 100
#             st.markdown(
#                 f"**Cluster {i}**: {chems}  \n"
#                 f"Avg Price: ${profile['avg_mean_price']:,.1f}/ton | "
#                 f"Volatility: {avg_vol:.2f}%"
#             )
#             st.markdown("---")

#     st.subheader("Optimal Cluster Selection (Elbow + Silhouette)")
#     opt = result["optimal"]
#     fig2 = make_subplots(specs=[[{"secondary_y": True}]])
#     fig2.add_trace(
#         go.Scatter(x=opt["k_range"], y=opt["inertias"], name="Inertia (Elbow)",
#                    mode="lines+markers", line=dict(color=COLORS["primary"])),
#         secondary_y=False)
#     fig2.add_trace(
#         go.Scatter(x=opt["k_range"], y=opt["silhouettes"], name="Silhouette Score",
#                    mode="lines+markers", line=dict(color=COLORS["accent"])),
#         secondary_y=True)
#     fig2.add_vline(x=best_k, line_dash="dash", line_color=COLORS["warn"],
#                    annotation_text=f"Optimal k={best_k}")
#     fig2.update_layout(height=280)
#     st.plotly_chart(fig2, use_container_width=True)

#     st.subheader("Chemical Feature Comparison")
#     display_cols = ["chemical", "mean_price", "std_price", "cv", "volatility",
#                     "trend", "avg_pct_change", "cluster"]
#     display_cols = [c for c in display_cols if c in feat_df.columns]
#     show_df = feat_df[display_cols].copy()
#     show_df.columns = [c.replace("_", " ").title() for c in show_df.columns]
#     for col in show_df.select_dtypes(include=[float]).columns:
#         show_df[col] = show_df[col].round(4)
#     st.dataframe(show_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# HELPER: PERIOD-OVER-PERIOD % CHANGE (MoM /YoY)
# ---------------------------------------------------------------------------
def calc_period_change(chem_df, current_price, months_back, tolerance_days=10):
    """
    % change of current_price vs. the price ~months_back months ago.
    Uses the closest available date within `tolerance_days` of the target
    date to tolerate slight date-alignment drift (e.g. predicted rows,
    missing months). Returns None if no usable reference point exists.
    """
    target_date = chem_df["date"].max() - pd.DateOffset(months=months_back)
    window = chem_df[
        (chem_df["date"] >= target_date - pd.Timedelta(days=tolerance_days)) &
        (chem_df["date"] <= target_date + pd.Timedelta(days=tolerance_days))
    ]
    if window.empty:
        return None

    closest_idx = (window["date"] - target_date).abs().idxmin()
    ref_price = window.loc[closest_idx, "price_usd_per_ton"]

    if pd.isna(ref_price) or ref_price == 0:
        return None

    return (current_price - ref_price) / ref_price * 100

# ---------------------------------------------------------------------------
# PAGE: MARKET INSIGHTS
# ---------------------------------------------------------------------------
def page_market_insights(df, df_price, chemical):
    st.title("Market Insights")
    st.caption("Daily brief, procurement recommendations, and risk assessment")

    chem_df = df[df["chemical"] == chemical].sort_values("date")
    latest = chem_df.iloc[-1]
    prev = chem_df.iloc[-2] if len(chem_df) > 1 else chem_df.iloc[-1]

    mom_change = calc_period_change(chem_df, latest["price_usd_per_ton"],
                                     months_back=1, tolerance_days=10)
    yoy_change = calc_period_change(chem_df, latest["price_usd_per_ton"],
                                     months_back=12, tolerance_days=15)

    mom_display = f"{mom_change:+.2f}%" if mom_change is not None else "N/A"
    yoy_display = f"{yoy_change:+.2f}%" if yoy_change is not None else "N/A"

    st.markdown(f"#### Morning Intelligence Brief: {chemical}")
    # st.markdown(f"**Date:** {datetime.date.today().strftime('%d %b, %Y')}  |  "
    #             f"**Price:** ${latest['price_usd_per_ton']:,.2f}/ton"
    #             )

    # # sentiment_level = "Positive" if latest["sentiment_score"] > 0.05 else \
    # #     "Negative" if latest["sentiment_score"] < -0.05 else "Neutral"
    # st.markdown(
    #     f"**MoM Change:** {mom_change:+.2f}%  "
    #     + (f"|  **YoY Change:** {yoy_change:+.2f}%" if yoy_change is not None else "")
    # )

    # st.markdown(
    #     f"**Crude Oil:** \\${latest['crude_oil_price']:.2f}/bbl  | "
    #     f"**Natural Gas:** \\${latest['natural_gas_price']:.3f}/MMBtu"
    # )

    #col wise
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.markdown(f"**Date:** {datetime.date.today().strftime('%d %b, %Y')}")
    #     st.markdown(f"**MoM Change:** {mom_change:+.2f}%")
    #     st.markdown(f"**Crude Oil:** \\${latest['crude_oil_price']:.2f}/bbl")
    # with col2:
    #     st.markdown(f"**Price:** ${latest['price_usd_per_ton']:,.2f}/ton")
    #     st.markdown(f"**YoY Change:** {yoy_change:+.2f}%")
    #     st.markdown(f"**Natural Gas:** \\${latest['natural_gas_price']:.3f}/MMBtu")
    
    #html
    # st.markdown(f"""
    #     <div style="display:flex; gap:4rem; margin-top:0.25rem;">
    #     <div style="display:flex; flex-direction:column; gap:0.5rem;">
    #         <span><strong>Date:</strong> {datetime.date.today().strftime('%d %b, %Y')}</span>
    #         <span><strong>MoM Change:</strong> {mom_change:+.2f}%</span>
    #         <span><strong>Crude Oil:</strong> ${latest['crude_oil_price']:.2f}/bbl</span>
    #     </div>
    #     <div style="display:flex; flex-direction:column; gap:0.5rem;">
    #         <span><strong>Price:</strong> ${latest['price_usd_per_ton']:,.2f}/ton</span>
    #         <span><strong>YoY Change:</strong> {yoy_change:+.2f}%</span>
    #         <span><strong>Natural Gas:</strong> ${latest['natural_gas_price']:.3f}/MMBtu</span>
    #     </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    #updated corrected macro
    # Fetch live macro prices (reuses cached result from dashboard — no extra network call)
    _macro_live, _macro_err = fetch_macro_live()
    if _macro_live is not None and not _macro_live.empty:
        _macro_latest   = _macro_live.iloc[-1]
        _crude_display  = f"${_macro_latest['crude_oil_price']:.2f}/bbl"
        _gas_display    = f"${_macro_latest['natural_gas_price']:.3f}/MMBtu"
    else:
        _crude_display  = f"${latest['crude_oil_price']:.2f}/bbl"
        _gas_display    = f"${latest['natural_gas_price']:.3f}/MMBtu"

    st.markdown(f"""
        <div style="display:flex; gap:4rem; margin-top:0.25rem;">
        <div style="display:flex; flex-direction:column; gap:0.5rem;">
            <span><strong>Date:</strong> {datetime.date.today().strftime('%d %b, %Y')}</span>
            <span><strong>MoM Change:</strong> {mom_display}</span>
            <span><strong>Crude Oil:</strong> {_crude_display}</span>
        </div>
        <div style="display:flex; flex-direction:column; gap:0.5rem;">
            <span><strong>Price:</strong> ${latest['price_usd_per_ton']:,.2f}/ton</span>
            <span><strong>YoY Change:</strong> {yoy_display}</span>
            <span><strong>Natural Gas:</strong> {_gas_display}</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, divider, col2 = st.columns([1, 0.05, 1])

    with divider:
        st.markdown(
            '<div style="border-left:1px solid #e5e7eb; height:100%; min-height:300px; margin-top:40px;"></div>',
            unsafe_allow_html=True
        )

    with col1:
        st.markdown("#### Risk Assessment Matrix")

        price_vol = chem_df["price_usd_per_ton"].pct_change().std() * 100


        _macro_live, _macro_err = fetch_macro_live()
        if _macro_live is not None and not _macro_live.empty:
            _chem_merge  = chem_df[["date", "price_usd_per_ton"]].copy()
            _macro_merge = _macro_live[["date", "crude_oil_price", "natural_gas_price"]].copy()

            _chem_merge["ym"]  = _chem_merge["date"].dt.to_period("M")
            _macro_merge["ym"] = _macro_merge["date"].dt.to_period("M")

            _corr_df = _chem_merge.merge(
                _macro_merge[["ym", "crude_oil_price", "natural_gas_price"]],
                on="ym", how="inner"
            )
        else:
            _corr_df = chem_df[["date", "price_usd_per_ton", "crude_oil_price", "natural_gas_price"]].copy()

        _crude_corr = abs(_corr_df["crude_oil_price"].corr(_corr_df["price_usd_per_ton"]))
        _gas_corr   = abs(_corr_df["natural_gas_price"].corr(_corr_df["price_usd_per_ton"]))

        risk_data = {
            "Risk Factor": ["Price Volatility", "Crude Oil Sensitivity", "Natural Gas Sensitivity"],
            "Level": [
                "High" if price_vol > 5 else "Medium" if price_vol > 2 else "Low",
                "High" if _crude_corr > 0.6 else "Medium" if _crude_corr > 0.3 else "Low",
                "High" if _gas_corr   > 0.6 else "Medium" if _gas_corr   > 0.3 else "Low",
            ],
            "Score": [
                min(price_vol / 10, 1),
                _crude_corr,
                _gas_corr,
            ],
        }
        risk_df = pd.DataFrame(risk_data)

        def _pill(level):
            bg  = "#ffcdd2" if level == "High" else "#fff9c4" if level == "Medium" else "#c8e6c9"
            tc  = "#b71c1c" if level == "High" else "#f57f17" if level == "Medium" else "#2e7d32"
            return (
                f'<span style="background:{bg}; color:{tc}; padding:3px 12px; '
                f'border-radius:12px; font-weight:600; font-size:14px;">{level}</span>'
            )

        st.markdown(f"""
            <table style="width:100%; border-collapse:collapse; font-size:16px; margin-top:8px;">
                <thead>
                    <tr style="background:#f1f5f9;">
                        <th style="text-align:left; padding:10px 14px; color:#64748b; font-weight:600; border-radius:8px 0 0 0;">Risk Factor</th>
                        <th style="text-align:left; padding:10px 14px; color:#64748b; font-weight:600; border-radius:0 8px 0 0;">Level</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid #f1f5f9;">
                        <td style="padding:10px 14px; color:#2f3340;">Price Volatility</td>
                        <td style="padding:10px 14px; text-align:center;">{_pill(risk_df.loc[0,'Level'])}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #f1f5f9;">
                        <td style="padding:10px 14px; color:#2f3340;">Crude Oil Sensitivity</td>
                        <td style="padding:10px 14px; text-align:center;">{_pill(risk_df.loc[1,'Level'])}</td>
                    </tr>
                    <tr>
                        <td style="padding:10px 14px; color:#2f3340;">Natural Gas Sensitivity</td>
                        <td style="padding:10px 14px; text-align:center;">{_pill(risk_df.loc[2,'Level'])}</td>
                    </tr>
                </tbody>
            </table>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Procurement Recommendations")

        trend_6m = np.polyfit(range(min(6, len(chem_df))),
                            chem_df["price_usd_per_ton"].tail(6).values, 1)[0]

        _macro_live, _macro_err = fetch_macro_live()
        if _macro_live is not None and len(_macro_live) >= 2:
            _crude_trend   = _macro_live["crude_oil_price"].iloc[-1] - _macro_live["crude_oil_price"].iloc[-2]
            _gas_trend     = _macro_live["natural_gas_price"].iloc[-1] - _macro_live["natural_gas_price"].iloc[-2]
            _energy_rising  = (_crude_trend > 0) or (_gas_trend > 0)
            _energy_falling = (_crude_trend < 0) and (_gas_trend < 0)
        else:
            _energy_rising  = False
            _energy_falling = False

        if trend_6m > 0 and latest["sentiment_score"] < 0:
            rec    = "BUY NOW"
            detail = "Price trending up but sentiment is negative: short window before market correction. Lock in current prices."
            color  = "🔴"
            bg     = "#fff0f0"
            border = "#ef4444"
            tc     = "#991b1b"

        elif trend_6m < 0 and latest["sentiment_score"] > 0:
            rec    = "WAIT"
            detail = "Price declining with positive sentiment: further drops likely. Delay procurement by 1-2 months."
            color  = "🟡"
            bg     = "#fefce8"
            border = "#eab308"
            tc     = "#854d0e"

        elif trend_6m > 0 and mom_change is not None and mom_change > 5 and _energy_rising:
            rec    = "STRATEGIC BUY"
            detail = "Strong upward momentum with rising energy costs: prices likely to increase further. Build inventory now."
            color  = "🟠"
            bg     = "#fff7ed"
            border = "#f97316"
            tc     = "#9a3412"

        elif trend_6m > 0 and mom_change is not None and mom_change > 5:
            rec    = "STRATEGIC BUY"
            detail = "Strong upward price momentum. Build inventory now at current prices to hedge against further increases."
            color  = "🟠"
            bg     = "#fff7ed"
            border = "#f97316"
            tc     = "#9a3412"

        elif trend_6m > 0 and _energy_rising:
            rec    = "STRATEGIC BUY"
            detail = "Price trending up and energy costs rising: input cost pressure likely to push prices higher. Consider buying ahead."
            color  = "🟠"
            bg     = "#fff7ed"
            border = "#f97316"
            tc     = "#9a3412"

        elif trend_6m < 0 and _energy_falling:
            rec    = "WAIT"
            detail = "Price declining and energy costs falling: further price drops expected. Delay procurement by 1-2 months."
            color  = "🟡"
            bg     = "#fefce8"
            border = "#eab308"
            tc     = "#854d0e"

        else:
            rec    = "MONITOR"
            detail = "Price stable. No urgent procurement action required. Review in next monthly cycle."
            color  = "🟢"
            bg     = "#f0fdf4"
            border = "#22c55e"
            tc     = "#166534"

        #Recommendation label
        st.markdown(f"""
            <div style="font-size:17px; font-weight:700; margin:4px 0 8px 0;">
                {color} {rec}
            </div>
        """, unsafe_allow_html=True)

        #Detail box
        #addons: border:1.5px solid {border};
        st.markdown(f"""
            <div style="
                background:{bg};
                border-radius:8px;
                padding:12px 16px;
                font-size:15px;
                color:{tc};
                margin-bottom:12px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.25);
            ">
                {detail}
            </div>
        """, unsafe_allow_html=True)

        #Percentile box
        price_hist = chem_df["price_usd_per_ton"]
        p25 = price_hist.quantile(0.25)
        p75 = price_hist.quantile(0.75)
        cur = latest["price_usd_per_ton"]

        if cur < p25:
            pct_bg     = "#f0fdf4"
            pct_border = "#22c55e"
            pct_tc     = "#166534"
            pct_text   = f"Current price (${cur:,.1f}) is in the <b>bottom 25th percentile</b>: historically cheap"
        elif cur > p75:
            pct_bg     = "#fefce8"
            pct_border = "#eab308"
            pct_tc     = "#854d0e"
            pct_text   = f"Current price (${cur:,.1f}) is in the <b>top 25th percentile</b>: historically expensive"
        else:
            pct_bg     = "#eff6ff"
            pct_border = "#3b82f6"
            pct_tc     = "#1e40af"
            pct_text   = f"Current price (${cur:,.1f}) is at <b>mid-range</b> historically"

        #Percentile box
        #Addons: border:1.5px solid {pct_border};
        st.markdown(f"""
            <div style="
                background:{pct_bg};
                border-radius:8px;
                padding:12px 16px;
                font-size:15px;
                color:{pct_tc};
                box-shadow: 0 4px 16px rgba(0,0,0,0.25);
            ">
                {pct_text}
            </div>
        """, unsafe_allow_html=True)

    # st.markdown("---")
    # st.subheader("Price Statistics")
    # stats = chem_df["price_usd_per_ton"].describe()
    # stat_df = pd.DataFrame({
    #     "Metric": ["Count", "Mean", "Std Dev", "Min", "25th Pct", "Median", "75th Pct", "Max"],
    #     "Value": [
    #         f"{int(stats['count'])} months",
    #         f"${stats['mean']:,.2f}",
    #         f"${stats['std']:,.2f}",
    #         f"${stats['min']:,.2f}",
    #         f"${stats['25%']:,.2f}",
    #         f"${stats['50%']:,.2f}",
    #         f"${stats['75%']:,.2f}",
    #         f"${stats['max']:,.2f}",
    #     ],
    # })
    # st.dataframe(stat_df, use_container_width=True, hide_index=True)

    # st.markdown("---")
    # st.subheader("Weekly/Monthly Alert System")
    # threshold_pct = st.slider("Price alert threshold (%)", 1, 20, 5)
    # recent_change = abs(mom_change)
    # if recent_change > threshold_pct:
    #     st.error(f"PRICE ALERT: {chemical} moved {mom_change:+.2f}% last month — "
    #              f"exceeds {threshold_pct}% threshold!")
    # else:
    #     st.success(f"No alert: {chemical} moved only {mom_change:+.2f}% last month — "
    #                f"within {threshold_pct}% threshold")


# ---------------------------------------------------------------------------
# PAGE: DOWNLOAD
# ---------------------------------------------------------------------------
# def page_download(df):
#     st.title("Download")
#     st.caption("Download all code files, ML models, and data as a ready-to-run package")

#     st.markdown("""
#     ### What's included in the download

#     | File | Description |
#     |------|-------------|
#     | `app.py` | Main Streamlit application |
#     | `models/arima_model.py` | ARIMA time series forecasting |
#     | `models/prophet_model.py` | Prophet model with macro regressors |
#     | `models/lstm_model.py` | Neural network (LSTM-style MLP) forecasting |
#     | `models/price_prediction.py` | Ensemble price prediction model |
#     | `models/anomaly_detection.py` | Isolation Forest anomaly detection |
#     | `models/clustering.py` | K-Means clustering analysis |
#     | `data/FINAL_sentiment_macro.csv` | Price + sentiment + macro data |
#     | `data/final_2020_2026.csv` | Base price data 2020–2026 |
#     | `requirements.txt` | Python dependencies |
#     | `README.md` | Setup and usage instructions |
#     """)

#     st.markdown("---")

#     readme_content = """# Suvira Intelligence — Chemical Market Intelligence System

# ## Setup

# ```bash
# pip install -r requirements.txt
# streamlit run app.py
# ```

# ## Structure

# ```
# suvira_intelligence/
# ├── app.py                   # Main Streamlit app
# ├── requirements.txt         # Python dependencies
# ├── .streamlit/
# │   └── config.toml          # Streamlit server config
# ├── data/
# │   ├── FINAL_sentiment_macro.csv   # Price + sentiment + macro
# │   └── final_2020_2026.csv         # Base price data
# ├── assets/
# │   └── logo.png
# └── models/
#     ├── arima_model.py       # ARIMA forecasting
#     ├── prophet_model.py     # Prophet forecasting
#     ├── lstm_model.py        # Neural network forecasting
#     ├── price_prediction.py  # Ensemble price prediction
#     ├── anomaly_detection.py # Anomaly detection
#     └── clustering.py        # Market clustering
# ```

# ## ML Models

# | Model | Purpose | Method |
# |-------|---------|--------|
# | ARIMA | Time-series forecasting | statsmodels ARIMA with auto-order selection |
# | Prophet | Seasonal forecasting | Facebook Prophet with macro regressors |
# | Neural Network | Deep learning forecasting | MLPRegressor (LSTM-style) |
# | Ensemble | Price prediction | RF + GradientBoosting + Ridge voting |
# | Anomaly Detection | Outlier identification | Isolation Forest + Z-score |
# | Clustering | Market segmentation | K-Means with PCA visualization |

# ## Features

# - Dashboard with KPIs, price trends, correlation heatmap
# - Multi-model demand forecasting with confidence intervals
# - Ensemble price prediction with sentiment & macro features
# - Anomaly detection (Isolation Forest + Statistical)
# - Sentiment analysis and macro driver correlation
# - K-Means clustering for chemical market segmentation
# - Market insights with procurement recommendations
# - Downloadable reports

# ## Data

# Chemicals covered: Acetic Acid, Formates, KCL, PAC, Sulphur, Sulphuric Acid, Urea, Water Treatment Chemicals
# Time range: January 2020 — 2026
# Features: Price (USD/ton), Net Weight, FOB Value, Sentiment Score, Crude Oil Price, Natural Gas Price

# ## KPIs

# - Forecast MAPE target: < 10% for 30-day forecasts
# - Coverage: 8 chemicals
# - Models: ARIMA, Prophet, Neural Network, Ensemble

# ## Prepared by
# Suvira Energy — Market Intelligence & Analytics
# """

#     def build_zip():
#         buffer = io.BytesIO()
#         app_dir = BASE_DIR

#         with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
#             files_to_include = [
#                 ("app.py", "app.py"),
#                 ("requirements.txt", "requirements.txt"),
#                 (".streamlit/config.toml", ".streamlit/config.toml"),
#                 ("models/__init__.py", "models/__init__.py"),
#                 ("models/arima_model.py", "models/arima_model.py"),
#                 ("models/prophet_model.py", "models/prophet_model.py"),
#                 ("models/lstm_model.py", "models/lstm_model.py"),
#                 ("models/price_prediction.py", "models/price_prediction.py"),
#                 ("models/anomaly_detection.py", "models/anomaly_detection.py"),
#                 ("models/clustering.py", "models/clustering.py"),
#                 ("data/FINAL_sentiment_macro.csv", "data/FINAL_sentiment_macro.csv"),
#                 ("data/final_2020_2026.csv", "data/final_2020_2026.csv"),
#             ]

#             for rel_path, zip_name in files_to_include:
#                 full_path = os.path.join(app_dir, rel_path)
#                 if os.path.exists(full_path):
#                     zf.write(full_path, f"suvira_intelligence/{zip_name}")

#             logo_path = os.path.join(app_dir, "assets", "logo.png")
#             if os.path.exists(logo_path):
#                 zf.write(logo_path, "suvira_intelligence/assets/logo.png")

#             zf.writestr("suvira_intelligence/README.md", readme_content)

#         buffer.seek(0)
#         return buffer

#     with st.spinner("Building download package…"):
#         zip_buffer = build_zip()

#     st.download_button(
#         label="⬇️ Download Suvira Intelligence Package (.zip)",
#         data=zip_buffer,
#         file_name="suvira_intelligence.zip",
#         mime="application/zip",
#         type="primary",
#         use_container_width=True,
#     )

#     st.markdown("---")
#     st.subheader("Export Current Data View")
#     chemical_filter = st.multiselect("Select chemicals to export", df["chemical"].unique().tolist(),
#                                      default=df["chemical"].unique().tolist())
#     export_df = df[df["chemical"].isin(chemical_filter)].copy()
#     csv_data = export_df.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         label="⬇️ Download Filtered Data as CSV",
#         data=csv_data,
#         file_name="suvira_data_export.csv",
#         mime="text/csv",
#         use_container_width=True,
#     )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    with st.spinner("Loading data…"):
        df_sent, df_price = load_data()
        
        # auto-fill missing months
        df_sent = auto_update_monthly_prices(df_sent)


    chemicals = sorted(df_sent["chemical"].unique().tolist())

    page, selected_chemical, steps = render_sidebar(chemicals)

    if page == "Dashboard":
        page_dashboard(df_sent, chemicals)

    elif page == "Price Prediction":
        page_price_prediction(df_sent, selected_chemical, steps)

    # elif page == "Demand Forecasting":
    #     page_demand_forecasting(df_sent, selected_chemical, steps)

    # elif page == "Anomaly Detection":
    #     page_anomaly(df_sent, selected_chemical)

    # elif page == "Sentiment & Macro":
    #     page_sentiment(df_sent, selected_chemical)

    # elif page == "Clustering Analysis":
    #     page_clustering(df_sent)

    elif page == "Market Insights":
        page_market_insights(df_sent, df_price, selected_chemical)

    # elif page == "Download":
    #     page_download(df_sent)


if __name__ == "__main__":
    main()
