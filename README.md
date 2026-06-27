# Suvira Intelligence: Chemical Market Intelligence System

A data-driven market intelligence platform for monitoring, forecasting, and analysing global chemical commodity prices. Built for procurement teams and market analysts to make informed sourcing decisions.

## Live App

**App:** [suvira-intelligence.streamlit.app](https://suvira-intelligence.streamlit.app)
Built and deployed using Streamlit Community Cloud.

## Overview

Suvira Intelligence tracks and forecasts prices for 8 key industrial chemicals imported into India:

- Sulphur
- Sulphuric Acid
- Urea
- KCL (Potassium Chloride)
- PAC (Polyaluminium Chloride)
- Acetic Acid
- Formates
- Water Treatment Chemicals

The system combines historical trade data, macroeconomic indicators (crude oil, natural gas), and market sentiment scores to generate monthly price forecasts and procurement recommendations.

## Features

### Dashboard
- Live price KPI cards for all chemicals with Month-over-Month (MoM) change
- Historical price trend charts, with actual data shown as a solid line and predicted data as a dotted line (driven by the `is_predicted` flag)
- Macro driver analysis (Crude Oil & Natural Gas), pulled live from Yahoo Finance
- Live market news feed via Google News RSS

### Price Prediction
- **Sulphur:** ARIMA(2,1,2) time-series model — captures the structural price shock following geopolitical disruption, which tree-based models can't extrapolate beyond their training range
- **All other chemicals:** Ensemble model (Random Forest + Gradient Boosting + Ridge, via `VotingRegressor`) + ARIMA
- Adjustable forecast horizon (1–12 months ahead)
- 90% confidence intervals on all forecasts
- Historical vs. predicted visualisation with a forecast table (price, lower CI, upper CI)

### Market Insights
- Chemical-specific deep dive
- MoM and YoY price change (date-tolerant window calculation)
- 6-month trend analysis
- AI-driven procurement recommendation (Buy Now /Wait /Strategic Buy /Hold) based on price momentum, energy costs, and sentiment
- Price percentile positioning vs. historical range

### Auto-Update
- On load, automatically predicts and appends each chemical's missing month(s) up to the last completed month
- Predicts one month at a time per chemical, using that chemical's own latest available date, then re-runs the model on the growing dataset for subsequent months
- New rows are flagged `is_predicted = True` and appended directly to `FINAL_sentiment_macro.csv`
- Keeps a rolling 10-year window of data, dropping anything older

---

## Models

### Ensemble Model (all chemicals except Sulphur)
- **Algorithm:** `VotingRegressor` (Random Forest + Gradient Boosting + Ridge Regression)
- **Features:** Price lags (1, 2, 3, 6 months), rolling averages (3M, 6M), rolling std, % change, sentiment lags, crude oil lags, natural gas lags, volume lags, seasonal features (month, quarter, year)
- **Validation:** TimeSeriesSplit (5-fold cross-validation), held-out test window for MAE/MAPE/RMSE
- **Limitation:** Tree-based models can't extrapolate beyond historically observed price ranges; best suited to chemicals with stable, trend-following price behaviour

### ARIMA Model (Sulphur)
- **Algorithm:** ARIMA(2,1,2) via `statsmodels`
- **Reason:** Sulphur experienced a structural price shock following geopolitical disruption (US–Iran conflict, Strait of Hormuz disruptions), spiking from historically stable levels to a much higher range. Differencing (d=1) lets the model follow trend momentum and project beyond historical bounds, which tree-based models cannot do.
- **Confidence Interval:** 90% CI via `get_forecast()`
- Forecasts are floored at zero to avoid unrealistic negative dips on volatile series

---

## Tech Stack

- **Frontend/UI:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Random Forest, Gradient Boosting, Ridge, VotingRegressor)
- **Time Series:** Statsmodels (ARIMA)
- **Visualisation:** Plotly
- **News Feed:** Feedparser (Google News RSS)
- **Macro Data:** Yahoo Finance (direct `requests` call to the Yahoo Finance chart API, fetched live on each session and cached for 24h)

---

## Project Structure

```
suvira_intelligence/
├── app.py                          # Main Streamlit application
├── fix_macro_data.py               # Backfills crude oil / natural gas values via Yahoo Finance
├── requirements.txt                # Python dependencies
├── README.md
│
├── assets/
│   ├── logo.png
│   ├── logo_filled.png
│   ├── logo_hollow.png
│   └── logo copy.png
│
├── data/
│   ├── FINAL_sentiment_macro.csv        # Master dataset (prices + sentiment + macro)
│   ├── final_2020_2026.csv              # Base price data
│   ├── predicted_prices.csv
│   └── FINAL_sentiment_macro_BACKUP.csv
│
├── models/
│   ├── __init__.py
│   ├── price_prediction.py         # Ensemble ML model (all chemicals except Sulphur)
│   ├── price_prediction2.py        # ARIMA model (Sulphur)
│   ├── model_manifest.json
│   ├── sulphur_arima.pkl
│   ├── sulphuric_acid_arima.pkl
│   ├── urea_arima.pkl
│   ├── kcl_arima.pkl
│   ├── pac_arima.pkl
│   ├── acetic_acid_arima.pkl
│   ├── formates_arima.pkl
│   └── water_treatment_chemicals_arima.pkl
│
├── development_history
    ├── raw_datasets/                   # Source/working data — not used directly by the app
    │   ├── merged/
    │   ├── processed/
    │   ├── raw/
    │   ├── 0_master_data_chemicals_2025.csv
    │   ├── 1_master_dataset_2026.csv
    │   ├── crude_oil_daily.csv
    │   ├── natural_gas_daily.csv
    │   ├── final_2020_2026.csv
    │   ├── final_2020_2026_with_research_adjustment.csv
    │   ├── final_2020_2026_with_sentiment_CUSTOM.csv
    │   └── FINAL_sentiment+macro.csv
    │
    ├── Notebooks/                       # Exploratory / model-development notebooks
    │   ├── 01_trade_price_processing.ipynb
    │   ├── 02_combine_datasets.ipynb
    │   ├── 03_arima_model_v1.ipynb
    │   ├── 05_manual_adjustment_v3.ipynb
    │   ├── 06_new_training_code_final_DS.ipynb
    │   ├── 07_updated_04_sarimax_v2.ipynb
    │   ├── 08_final_model_comparison.ipynb
    │   ├── 09_final_predictions_after_revision.ipynb
    │   ├── 10_sentiment_analysis_NEWS.ipynb
    │   ├── 11_real_news_sentiment.ipynb
    │   ├── 12_sarimax_with_sentiment.ipynb
    │   ├── sapp_old.py
    │   └── previous_sapp.py
    │
    └── test_outputs/                   # Model evaluation / experiment outputs
        ├── arima_predictions_v1.csv
        ├── base_final_prices_v3.csv
        ├── final_arima_results.csv
        ├── final_prices_v4_before_SARIMAX_analysis.csv
        ├── final_sarimax_results.csv
        ├── model_comparison.csv
        ├── sarimax_predictions_v2.csv
        └── sarimax_results_sentiment_CUSTOM.csv
```

---

## Dataset

The master dataset (`FINAL_sentiment_macro.csv`) contains monthly data from January 2020 to present, including:

| Column | Description |
|---|---|
| `date` | Month start date (YYYY-MM-01) |
| `chemical` | Chemical name |
| `price_usd_per_ton` | Average monthly price (USD/MT) |
| `netWgt` | Import volume (kg) — from trade data |
| `fobvalue` | FOB value (USD) — from trade data |
| `sentiment_score` | Market sentiment score (-1 to +1) |
| `crude_oil_price` | Crude oil price (USD/barrel) |
| `natural_gas_price` | Natural gas price (USD/MMBtu) |
| `is_predicted` | `True` if the value is model-predicted, `False` if actual/observed |

> **Note:** Trade volume data (`netWgt`, `fobvalue`) typically lags 3–6 months from customs/DGFT publication. Recent months use forward-filled values for volume features while price and macro data are kept current.

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- pip
- Git

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/suvira-intelligence.git
   cd suvira-intelligence
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   `http://localhost:8501`

---

## Known Limitations

- **Trade volume data lag:** Real import/export volumes (`netWgt`, `fobvalue`) from DGFT/customs typically take 3–6 months to be published. Recent months use forward-filled values for volume features.
- **Sentiment scores:** Currently static for predicted months (carry forward the last known value). A live NLP sentiment pipeline from news feeds would improve future forecast accuracy, especially during volatile periods.
- **Auto-update persistence:** On cloud deployment (e.g. Streamlit Community Cloud), the CSV write in `auto_update_monthly_prices()` is ephemeral and does not persist across app restarts. For production use, this should be connected to a cloud database (e.g. Supabase, AWS S3, Google Sheets).
- **Black swan events:** No historical ML model can predict sudden geopolitical price shocks (e.g. the US–Iran conflict, Strait of Hormuz disruptions). The system is designed to adapt once post-event data becomes available — typically 3–6 months after the event.

---

## Data Sources

- **Trade data:** DGFT / Indian Customs import-export records (UNComtrade as a supplementary source), Ex-im
- **Macro data:** Yahoo Finance (crude oil — `CL=F`, natural gas — `NG=F`)
- **Sentiment scores:** Derived from market news NLP analysis
- **Price benchmarks (for validation):** ICIS, Argus Media, TradingEconomics, IMARC, Ex-im

---

## Disclaimer

This application is built for academic and portfolio demonstration purposes. Price forecasts are model-generated and should not be used as the **sole** basis for commercial procurement decisions. Always cross-reference with licensed price reporting agencies (ICIS, Platts, Argus) for actual trading decisions.

---

## Author

**Mangalya D. Phaye** **|**
Summer Intern (Jan–Jun 2026) — [Suvira Energy](https://suvira.com/) **|**
mphaye05@gmail.com