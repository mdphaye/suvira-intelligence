"""
Price Prediction Model - ARIMA (for chemicals with structural price shocks)
Suvira Energy - Market Intelligence System

This model is used specifically for chemicals like Sulphur that experienced
a sudden structural price shift (e.g. geopolitical disruption) which
tree-based ensemble models cannot extrapolate beyond their training range.
ARIMA models the trend/momentum directly and can project forward beyond
historically observed price levels.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


def build_series(df: pd.DataFrame, chemical: str) -> pd.Series:
    """
    Build a clean, monthly-indexed price series for the given chemical,
    ready to feed into ARIMA.
    """
    chem_df = df[df["chemical"] == chemical].copy()
    chem_df["date"] = pd.to_datetime(chem_df["date"])
    chem_df = chem_df.sort_values("date").reset_index(drop=True)

    series = chem_df.set_index("date")["price_usd_per_ton"]
    series = series.asfreq("MS")          
    series = series.interpolate()          

    return series


def train_arima(df: pd.DataFrame, chemical: str, test_size: int = 2,
                 order: tuple = (2, 1, 2)) -> dict:
    """
    Fit an ARIMA model on the chemical's full price history and
    evaluate it on a held-out test window.

    order=(p,d,q):
      p = autoregressive terms (how many past prices to look at)
      d = differencing (1 = model the month-to-month change, helps
          the model react to trend shifts like the Sulphur shock)
      q = moving average terms (how many past errors to factor in)
    """
    series = build_series(df, chemical)

    if len(series) <= test_size + 5:
        # not enough data to hold out a test set meaningfully
        test_size = max(1, len(series) // 5)

    train_series = series.iloc[:-test_size]
    test_series = series.iloc[-test_size:]

    model = ARIMA(train_series, order=order)
    fitted = model.fit()

    test_pred = fitted.forecast(steps=test_size)

    mae = np.mean(np.abs(test_series.values - test_pred.values))
    mape = np.mean(np.abs((test_series.values - test_pred.values) / test_series.values)) * 100
    rmse = np.sqrt(np.mean((test_series.values - test_pred.values) ** 2))

    full_model = ARIMA(series, order=order)
    full_fitted = full_model.fit()

    return {
        "fitted": full_fitted,
        "series": series,
        "dates_test": test_series.index,
        "y_test": test_series,
        "y_pred": test_pred.values,
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
    }


def predict_next_months(trained: dict, steps: int = 6) -> dict:
    fitted = trained["fitted"]
    series = trained["series"]

    forecast_result = fitted.get_forecast(steps=steps)
    predictions = forecast_result.predicted_mean.values

    # 90% confidence interval
    conf_int = forecast_result.conf_int(alpha=0.10)
    lower_ci = conf_int.iloc[:, 0].values
    upper_ci = conf_int.iloc[:, 1].values

    last_date = series.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq="MS")[1:]

    predictions = np.maximum(predictions, 0)
    lower_ci = np.maximum(lower_ci, 0)

    return {
        "dates": forecast_dates,
        "predictions": predictions,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }


def run_arima_pipeline(df: pd.DataFrame, chemical: str, steps: int = 6) -> dict:
    trained = train_arima(df, chemical)
    forecast = predict_next_months(trained, steps=steps)

    return {
        "chemical": chemical,
        "trained": trained,
        "forecast": forecast,
    }
