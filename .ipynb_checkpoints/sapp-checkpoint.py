import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.markdown("""
<style>

/* Adaptive background */
.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F5F7FA;
}

/* Headings */
h1, h2, h3, h4 {
    color: inherit !important;
}

/* Dropdown */
div[data-baseweb="select"] {
    max-width: 400px;
}

/* Borders */
div[data-baseweb="select"] > div,
input {
    border: 1px solid #2FA4D6 !important;
}

/* Buttons */
.stButton > button {
    border: 1px solid #2FA4D6;
    color: inherit;
}

.stButton > button:hover {
    border: 1px solid #76C043;
    color: #76C043;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: inherit;
}

/* Remove gap */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem;
    margin-top: 0rem;
}

</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("### SUVIRA")
st.sidebar.markdown("<small style='color: gray;'>Intelligent Chemical Analytics Platform</small>", unsafe_allow_html=True)

# SESSION STATE
if "page" not in st.session_state:
    st.session_state.page = "Market Trends"

# NAVIGATION
st.sidebar.markdown("### Navigation")

# ---- MARKET TRENDS ----
col1, col2 = st.sidebar.columns([1, 6])

with col1:
    if st.session_state.page == "Market Trends":
        st.image("logo_filled.png", width=18)
    else:
        st.image("logo_outline.png", width=18)

with col2:
    if st.button("Market Trends", key="nav_trends"):
        st.session_state.page = "Market Trends"

# ---- PREDICTION ----
col1, col2 = st.sidebar.columns([1, 6])

with col1:
    if st.session_state.page == "Prediction":
        st.image("logo_outline.png", width=18)
    else:
        st.image("logo_filled.png", width=18) 

with col2:
    if st.button("Prediction", key="nav_pred"):
        st.session_state.page = "Prediction"

page = st.session_state.page

# ---------- DATA ----------
df = pd.read_csv("outputs/final_prices_v4_before_SentimentAnalysis.csv")
df_hist = pd.read_csv("data/final_2020_2026.csv")
comparison_df = pd.read_csv("outputs/model_comparison.csv")
df_hist["date"] = pd.to_datetime(df_hist["date"])


#page1

if page == "Market Trends":

    st.title("SUVIRA Intelligence")
    st.subheader("Chemical price trends and decision support system")

    chemicals = df["Chemical"].unique()
    selected_chemical = st.selectbox("Select Chemical", chemicals)

    filtered_df = df[df["Chemical"] == selected_chemical]
    hist_filtered = df_hist[df_hist["chemical"] == selected_chemical].sort_values("date")

    predicted_price = filtered_df["Predicted Price (USD/ton)"].values[0]
    manual_price = filtered_df["Manual Price (USD/ton)"].values[0]
    final_price = filtered_df["Final Price (USD/ton)"].values[0]

    st.markdown("### Price Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Price", f"${predicted_price:.2f}")
    col2.metric("Manual Price", f"${manual_price}" if pd.notna(manual_price) else "Not Set")
    col3.metric("Final Price", f"${final_price:.2f}")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Historical Price Trend")
        st.line_chart(hist_filtered.set_index("date")["price_usd_per_ton"])

    with col_right:
        st.markdown("### Manual Price Adjustment")

        key_name = f"manual_input_{selected_chemical}"
        current_manual_price = filtered_df["Manual Price (USD/ton)"].values[0]

        if key_name not in st.session_state:
            st.session_state[key_name] = None if pd.isna(current_manual_price) else float(current_manual_price)

        manual_input = st.number_input(
            "Enter Manual Price (USD/ton)",
            min_value=0.0,
            step=1.0,
            key=key_name,
            placeholder="Enter price"
        )

        if manual_input is not None:
            if pd.isna(current_manual_price) or manual_input != float(current_manual_price):
                df.loc[df["Chemical"] == selected_chemical, "Manual Price (USD/ton)"] = manual_input
                df.loc[df["Chemical"] == selected_chemical, "Final Price (USD/ton)"] = manual_input
                df.to_csv("outputs/final_prices_v3.csv", index=False)
                st.rerun()

        updated_final_price = manual_input if manual_input is not None else predicted_price
        st.metric("Updated Final Price", f"${updated_final_price:.2f}")

        if st.button("Reset to Model Prediction"):
            df.loc[df["Chemical"] == selected_chemical, "Manual Price (USD/ton)"] = None
            df.loc[df["Chemical"] == selected_chemical, "Final Price (USD/ton)"] = predicted_price
            df.to_csv("outputs/final_prices_v4_before_SentimentAnalysis.csv", index=False)


            if key_name in st.session_state:
                del st.session_state[key_name]

            st.rerun()

        st.markdown("### Market Insights")

        price_series = hist_filtered["price_usd_per_ton"].dropna()

        if len(price_series) > 1:
            latest_price = price_series.iloc[-1]
            earliest_price = price_series.iloc[0]
            percent_change = None if earliest_price == 0 else ((latest_price - earliest_price) / earliest_price) * 100

            if percent_change is None:
                trend = "Not Available"
            elif percent_change > 0:
                trend = "Increasing"
            elif percent_change < 0:
                trend = "Decreasing"
            else:
                trend = "Stable"
        else:
            trend = "Not Available"
            percent_change = None

        max_price = hist_filtered["price_usd_per_ton"].max()
        min_price = hist_filtered["price_usd_per_ton"].min()

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Trend", trend)
            st.metric("Price Change", f"{percent_change:.2f}%" if percent_change is not None else "Not Available")

        with c2:
            st.metric("Highest Price", f"${max_price:.2f}")
            st.metric("Lowest Price", f"${min_price:.2f}")

#page2
elif page == "Prediction":

    st.title("Price Prediction Engine")
    st.subheader("ML-based forecasting")

    chemical = st.selectbox("Select Chemical", df["Chemical"].unique())
    best_model = comparison_df[comparison_df["Chemical"] == chemical]["Best Model"].values[0]

    # if st.button("Run Prediction"):
    #     st.info("Model integration using pickle will be added here.")
    
    if st.button("Run Prediction"):    
        predicted_price = df[df["Chemical"] == chemical]["Final Price (USD/ton)"].values[0]      # Get predicted value (already computed)
        st.success(f"Predicted Price for {chemical}: ${predicted_price:.2f}") #Backend:best_model is already selected, switch to A/S .pkl models