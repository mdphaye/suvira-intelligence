import streamlit as st
import pandas as pd

# TITLE
st.title("SUVIRA Intelligence")
st.subheader("An AI-powered chemical price prediction and decision support system")

# LOAD DATA
df = pd.read_csv("/Users/mangalyaphaye/Desktop/suvira_intelligence/outputs/final_prices_v3.csv")
df_hist = pd.read_csv("/Users/mangalyaphaye/Desktop/suvira_intelligence/data/master_dataset_2026.csv")
df_hist["date"] = pd.to_datetime(df_hist["date"])

# DROPDOWN
chemicals = df["Chemical"].unique()
selected_chemical = st.selectbox("Select Chemical", chemicals)

# FILTER DATA
filtered_df = df[df["Chemical"] == selected_chemical]
hist_filtered = df_hist[df_hist["chemical"] == selected_chemical].sort_values("date")

# EXTRACT VALUES
predicted_price = filtered_df["Predicted Price (USD/ton)"].values[0]
manual_price = filtered_df["Manual Price (USD/ton)"].values[0]
final_price = filtered_df["Final Price (USD/ton)"].values[0]

# KPI SECTION
st.markdown("### Price Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Price", f"${predicted_price:.2f}")
col2.metric("Manual Price", f"${manual_price}" if pd.notna(manual_price) else "Not Set")
col3.metric("Final Price", f"${final_price:.2f}")

# MANUAL INPUT
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

# APPLY OVERRIDE
if manual_input is not None:
    if pd.isna(current_manual_price) or manual_input != float(current_manual_price):
        df.loc[df["Chemical"] == selected_chemical, "Manual Price (USD/ton)"] = manual_input
        df.loc[df["Chemical"] == selected_chemical, "Final Price (USD/ton)"] = manual_input
        df.to_csv("/Users/mangalyaphaye/Desktop/suvira_intelligence/outputs/final_prices_v3.csv", index=False)
        st.rerun()

# UPDATED FINAL PRICE
updated_final_price = manual_input if manual_input is not None else predicted_price

st.subheader("Updated Final Price")
st.metric("Final Price (Adjusted)", f"${updated_final_price:.2f}")

# RESET
if st.button("Reset to Model Prediction"):
    df.loc[df["Chemical"] == selected_chemical, "Manual Price (USD/ton)"] = None
    df.loc[df["Chemical"] == selected_chemical, "Final Price (USD/ton)"] = predicted_price
    df.to_csv("/Users/mangalyaphaye/Desktop/suvira_intelligence/outputs/final_prices_v3.csv", index=False)

    if key_name in st.session_state:
        del st.session_state[key_name]

    st.rerun()

# GRAPH
st.markdown("### Historical Price Trend")
st.line_chart(hist_filtered.set_index("date")["price_usd_per_ton"])

# INSIGHTS
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

    st.write(f"Trend: {trend}")
    st.write(f"Price Change: {percent_change:.2f}%" if percent_change is not None else "Price Change: Not Available")

else:
    st.write("Trend: Not Available")
    st.write("Price Change: Not Available")

# HIGH / LOW
max_price = hist_filtered["price_usd_per_ton"].max()
min_price = hist_filtered["price_usd_per_ton"].min()

st.write(f"Highest Price: ${max_price:.2f}")
st.write(f"Lowest Price: ${min_price:.2f}")