"""
Price Prediction Model: Ensemble (Random Forest + Gradient Boosting)
Incorporates sentiment scores and macro drivers
Suvira Energy: Market Intelligence System
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


def build_features(df: pd.DataFrame, chemical: str) -> pd.DataFrame:
    chem_df = df[df["chemical"] == chemical].copy()
    chem_df["date"] = pd.to_datetime(chem_df["date"])
    chem_df = chem_df.sort_values("date").reset_index(drop=True)

    chem_df["month"] = chem_df["date"].dt.month
    chem_df["quarter"] = chem_df["date"].dt.quarter
    chem_df["year"] = chem_df["date"].dt.year

    for lag in [1, 2, 3, 6]:
        chem_df[f"price_lag_{lag}"] = chem_df["price_usd_per_ton"].shift(lag)

    chem_df["price_ma_3"] = chem_df["price_usd_per_ton"].rolling(3).mean()
    chem_df["price_ma_6"] = chem_df["price_usd_per_ton"].rolling(6).mean()
    chem_df["price_std_3"] = chem_df["price_usd_per_ton"].rolling(3).std()
    chem_df["price_pct_change_1"] = chem_df["price_usd_per_ton"].pct_change(1)
    chem_df["price_pct_change_3"] = chem_df["price_usd_per_ton"].pct_change(3)

    if "netWgt" in chem_df.columns:
        chem_df["netWgt"] = chem_df["netWgt"].ffill()
        for lag in [1, 2, 3]:
            chem_df[f"volume_lag_{lag}"] = chem_df["netWgt"].shift(lag)
        chem_df["volume_ma_3"] = chem_df["netWgt"].rolling(3).mean()

    if "sentiment_score" in chem_df.columns:
        for lag in [1, 2]:
            chem_df[f"sentiment_lag_{lag}"] = chem_df["sentiment_score"].shift(lag)
        chem_df["sentiment_ma_3"] = chem_df["sentiment_score"].rolling(3).mean()

    if "crude_oil_price" in chem_df.columns:
        for lag in [1, 2, 3]:
            chem_df[f"crude_lag_{lag}"] = chem_df["crude_oil_price"].shift(lag)
        chem_df["crude_ma_3"] = chem_df["crude_oil_price"].rolling(3).mean()

    if "natural_gas_price" in chem_df.columns:
        for lag in [1, 2]:
            chem_df[f"gas_lag_{lag}"] = chem_df["natural_gas_price"].shift(lag)

    chem_df = chem_df.drop(columns=["fobvalue"], errors="ignore")
    chem_df = chem_df.dropna().reset_index(drop=True)
    return chem_df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"date", "chemical", "price_usd_per_ton", "fobvalue",
               "netWgt", "sentiment_score", "crude_oil_price", "natural_gas_price"}
    return [c for c in df.columns if c not in exclude]


def train_ensemble(df: pd.DataFrame, chemical: str, test_size: int = 2) -> dict:
    feat_df = build_features(df, chemical)

    feature_cols = get_feature_columns(feat_df)
    X = feat_df[feature_cols]
    y = feat_df["price_usd_per_ton"]
    dates = feat_df["date"]

    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    dates_test = dates.iloc[-test_size:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                   random_state=42)
    ridge = Ridge(alpha=1.0)

    ensemble = VotingRegressor(estimators=[("rf", rf), ("gb", gb), ("ridge", ridge)])
    ensemble.fit(X_train_scaled, y_train)

    rf.fit(X_train_scaled, y_train)
    gb.fit(X_train_scaled, y_train)
    ridge.fit(X_train_scaled, y_train)

    y_pred = ensemble.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = ensemble.score(X_test_scaled, y_test)

    rf_importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(ensemble, scaler.transform(X), y,
                                cv=tscv, scoring="neg_mean_absolute_percentage_error")

    return {
        "ensemble": ensemble,
        "rf": rf,
        "gb": gb,
        "ridge": ridge,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "dates_test": dates_test,
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
        "r2": r2,
        "feature_importance": rf_importance,
        "cv_mape": -cv_scores.mean() * 100,
        "feat_df": feat_df,
    }


def predict_next_months(trained: dict, df: pd.DataFrame, chemical: str,
                        steps: int = 6) -> dict:
    feat_df = trained["feat_df"].copy()
    feature_cols = trained["feature_cols"]
    scaler = trained["scaler"]
    ensemble = trained["ensemble"]

    last_row = feat_df.iloc[-1:][feature_cols]
    last_price = feat_df["price_usd_per_ton"].iloc[-1]
    last_date = feat_df["date"].iloc[-1]

    predictions = []
    forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq="MS")[1:]

    current_features = last_row.copy()
    last_prices = list(feat_df["price_usd_per_ton"].tail(6).values)

    for i in range(steps):
        pred = ensemble.predict(scaler.transform(current_features))[0]
        predictions.append(pred)
        last_prices.append(pred)

        if "price_lag_1" in feature_cols:
            current_features["price_lag_1"] = last_prices[-2]
        if "price_lag_2" in feature_cols:
            current_features["price_lag_2"] = last_prices[-3]
        if "price_lag_3" in feature_cols:
            current_features["price_lag_3"] = last_prices[-4]
        if "price_ma_3" in feature_cols:
            current_features["price_ma_3"] = np.mean(last_prices[-3:])
        if "price_ma_6" in feature_cols:
            current_features["price_ma_6"] = np.mean(last_prices[-6:])
        if "month" in feature_cols:
            current_features["month"] = forecast_dates[i].month
        if "quarter" in feature_cols:
            current_features["quarter"] = forecast_dates[i].quarter
        if "year" in feature_cols:
            current_features["year"] = forecast_dates[i].year

    predictions = np.array(predictions)
    std_error = trained["rmse"]
    lower_ci = predictions - 1.645 * std_error
    upper_ci = predictions + 1.645 * std_error

    return {
        "dates": forecast_dates,
        "predictions": predictions,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }


def run_price_prediction_pipeline(df: pd.DataFrame, chemical: str,
                                  steps: int = 6) -> dict:
    trained = train_ensemble(df, chemical)
    forecast = predict_next_months(trained, df, chemical, steps=steps)

    return {
        "chemical": chemical,
        "trained": trained,
        "forecast": forecast,
    }
