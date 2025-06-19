import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

def _calendar_cols(month_series: pd.Series) -> pd.DataFrame:
    """Return sin/cos month embeddings + quarter/holiday flags."""
    month_num = month_series.dt.month
    return pd.DataFrame({
        "month_num":     month_num,
        "month_sin":     np.sin(2 * np.pi * month_num / 12),
        "month_cos":     np.cos(2 * np.pi * month_num / 12),
        "quarter":       month_series.dt.quarter,
        "is_holiday":    month_num.isin([11, 12]).astype(int)
    }, index=month_series.index)

def build_forecast_rows(df: pd.DataFrame, forecast_month) -> pd.DataFrame:
    # -------- sanity checks ------------------------------------------------
    forecast_month = pd.Timestamp(forecast_month).to_period("M").to_timestamp()
    latest_month = pd.Timestamp(df["month"].max())

    if forecast_month <= latest_month:
        raise ValueError(f"`forecast_month` must be > last month in df ({latest_month.date()}).")

    forecast_offset = (forecast_month.to_period("M") - latest_month.to_period("M")).n

    last_rows = df[df["month"] == latest_month].copy()

    new = last_rows.copy()
    new["month"] = forecast_month
    new["year_month"] = new["month"].dt.to_period("M")

    new[_calendar_cols(new["month"]).columns] = _calendar_cols(new["month"])

    new["crime_count_lag_1m"] = last_rows["crime_count"]

    m_minus_3 = forecast_month - pd.DateOffset(months=3 + forecast_offset - 1)
    lookup3 = df[df["month"] == m_minus_3].set_index("lsoa_code")["crime_count"]
    new["crime_count_lag_3m"] = lookup3.reindex(new["lsoa_code"]).fillna(0).values

    for lag in [1, 3, 6, 12]:
        actual_lag = lag + forecast_offset - 1
        col = f"crime_count_pct_change_{lag}m"
        prev_val = df[df["month"] == forecast_month - pd.DateOffset(months=actual_lag)]
        prev_val = prev_val.set_index("lsoa_code")["crime_count"].reindex(new["lsoa_code"])
        new[col] = (new["crime_count_lag_1m"] - prev_val) / prev_val.replace(0, np.nan)
        new[col] = new[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    vol_start = forecast_month - pd.DateOffset(months=forecast_offset + 3 - 1)
    vol_end = forecast_month - pd.DateOffset(months=forecast_offset)
    vol_window = df[df["month"].between(vol_start, vol_end)]
    vol = vol_window.groupby("lsoa_code")["crime_count"].std()
    new["crime_volatility_3m"] = vol.reindex(new["lsoa_code"]).fillna(0).values

    new["months_since_burglary"] = last_rows["months_since_burglary"] + forecast_offset
    new.loc[last_rows["burglary_count"] > 0, "months_since_burglary"] = 0

    new["lag1_crime_x_pop"] = new["crime_count_lag_1m"] * new["population"]
    new["lag3_crime_x_imd"] = new["crime_count_lag_3m"] * new["imd_decile_2019"].astype(float)

    new["crime_entropy"] = last_rows["crime_entropy"]
    new["lag1_x_entropy"] = new["crime_count_lag_1m"] * new["crime_entropy"]
    new["lag3_x_entropy"] = new["crime_count_lag_3m"] * new["crime_entropy"]
    new["entropy_x_sin"] = new["crime_entropy"] * new["month_sin"]
    new["entropy_x_cos"] = new["crime_entropy"] * new["month_cos"]
    new["entropy_x_imd2019"] = new["crime_entropy"] * new["imd_decile_2019"].astype(float)

    new["volatility_x_sin"] = new["crime_volatility_3m"] * new["month_sin"]
    new["volatility_x_cos"] = new["crime_volatility_3m"] * new["month_cos"]

    new["stop_x_imd2019"] = last_rows["stop_and_search_count"] * new["imd_decile_2019"].astype(float)
    new["imd2019_x_msb"] = new["imd_decile_2019"].astype(float) * new["months_since_burglary"]

    new["crime_count"] = np.nan
    new["burglary_count"] = np.nan

    return new[df.columns]

def save_prediction(model: XGBRegressor, scaler: RobustScaler, month):
    df = pd.read_csv("../data/XGBoost_ready_dataset_with_features.csv", parse_dates=["month", "year_month"])

    features = scaler.feature_names_in_

    month = pd.Timestamp(month)
    next_rows = build_forecast_rows(df, month)

    X_next = scaler.transform(next_rows[features])
    next_rows["predicted_burglary"] = model.predict(X_next)
    next_rows["predicted_burglary"] = next_rows["predicted_burglary"].clip(lower=0).round().astype(int)

    # Save or concatenate with history
    next_rows[["lsoa_code", "year_month", "predicted_burglary"]].to_csv(
        "../data/burglary_next_month_forecast.csv", index=False
    )