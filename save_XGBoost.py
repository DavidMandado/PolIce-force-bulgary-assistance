import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import entropy
from pandas.tseries.offsets import MonthBegin
import joblib
import os

# load the dataset
df = pd.read_csv("data/XGBoost_ready_dataset.csv", low_memory=False)
df["month"] = pd.to_datetime(df["month"])
df["year_month"] = df["month"].dt.to_period("M")
df = df[df["burglary_count"].notna() & (df["burglary_count"] >= 0)].copy()
df.sort_values(["lsoa_code", "month"], inplace=True)
df.reset_index(drop=True, inplace=True)

# add time features
for lag in [1, 3, 6, 12]:
    col = f"crime_count_pct_change_{lag}m"
    df[col] = df.groupby("lsoa_code")["crime_count"].pct_change(lag)
    df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
# Derived features
df["delta_lag"] = df["lag_1"] - df["lag_2"]
df["momentum"] = df["lag_1"] - df["lag_3"]
df["stop_rate"] = df["stop_and_search_count"] / (df["population"] + 1)
df["log_pop"] = np.log1p(df["population"])
df["crime_per_capita"] = df["lag_1"] / (df["population"] + 1)

df["month_num"] = df["month"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
df["quarter"] = df["month"].dt.quarter
df["is_holiday_season"] = df["month_num"].isin([11, 12]).astype(int)

if "crime_count_lag_1m" not in df.columns:
    df["crime_count_lag_1m"] = df.groupby("lsoa_code")["crime_count"].shift(1).fillna(0)

if "crime_count_lag_3m" not in df.columns:
    df["crime_count_lag_3m"] = df.groupby("lsoa_code")["crime_count"].shift(3).fillna(0)

# add interaction features
df["lag1_crime_x_pop"] = df["crime_count_lag_1m"] * df["population"]
df["lag3_crime_x_imd"] = df["crime_count_lag_3m"] * df["imd_decile_2019"].astype(float)

df["crime_volatility_3m"] = (
    df.groupby("lsoa_code")["crime_count"]
    .transform(lambda x: x.rolling(3, min_periods=1).std())
    .fillna(0)
)

crime_types = [c for c in df.columns if c.startswith("crime_") and c.endswith("_count") and c != "burglary_count"]
crime_data = df[crime_types].to_numpy()

# Normalize rows (avoid division by zero)
row_sums = crime_data.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # to avoid division by zero
prob_matrix = crime_data / row_sums

# Compute entropy for each row
df["crime_entropy"] = entropy(prob_matrix.T, base=np.e)

# months since last burglary
def time_since_burglary(series):
    last_seen = -1
    result = []
    for val in series:
        if val > 0:
            last_seen = 0
        elif last_seen >= 0:
            last_seen += 1
        result.append(last_seen if last_seen >= 0 else np.nan)
    return result

df["months_since_burglary"] = df.groupby("lsoa_code")["burglary_count"].transform(time_since_burglary).fillna(100)

# extended interactions
df["lag1_x_entropy"] = df["crime_count_lag_1m"] * df["crime_entropy"]
df["lag3_x_entropy"] = df["crime_count_lag_3m"] * df["crime_entropy"]

df["entropy_x_sin"] = df["crime_entropy"] * df["month_sin"]
df["entropy_x_cos"] = df["crime_entropy"] * df["month_cos"]

df["entropy_x_imd2019"] = df["crime_entropy"] * df["imd_decile_2019"].astype(float)
df["volatility_x_sin"] = df["crime_volatility_3m"] * df["month_sin"]
df["volatility_x_cos"] = df["crime_volatility_3m"] * df["month_cos"]

df["stop_x_imd2019"] = df["stop_and_search_count"] * df["imd_decile_2019"].astype(float)
df["imd2019_x_msb"] = df["imd_decile_2019"].astype(float) * df["months_since_burglary"]

# feature selection
exclude_cols = {
    "lsoa_code", "month", "year_month", "crime_type",
    "latitude", "longitude", "burglary_count", "crime_count"
}
features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.int64, np.float64]]
print("Features:", features)
# scaling
scaler = RobustScaler()
X = scaler.fit_transform(df[features])
y = df["burglary_count"]

# splits
train = df["month"] < "2023-01-01"
val = (df["month"] >= "2023-01-01") & (df["month"] < "2024-01-01")
test = df["month"] >= "2024-01-01"
X_train, X_val, X_test = X[train], X[val], X[test]
y_train, y_val, y_test = y[train], y[val], y[test]

# model training
best_params = {
    'n_estimators': 500,
    'learning_rate': 0.2194440144154437,
    'max_depth': 5,
    'min_child_weight': 3,
    'subsample': 0.718124691088876,
    'colsample_bytree': 0.7743729096902522,
    'gamma': 3.361223663481572,
    'alpha': 4.578509071143608,
    'reg_lambda': 1.7898710081745761,
    'random_state': 42,
    'eval_metric': 'rmse',
    'verbosity': 1
}

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# evaluation
def evaluate(name, X, y):
    pred = final_model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    print(f"{name} → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    return pred

evaluate("Train", X_train, y_train)
evaluate("Validation", X_val, y_val)
evaluate("Test", X_test, y_test)

# feature importance
feature_names = df[features].columns.tolist()
importances = final_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)
print(feat_imp_df.head(20))

# === SAVE MODEL & SCALER ===
MODEL_PATH = "models/xgb_burglary_model.pkl"
SCALER_PATH = "models/robust_scaler.pkl"
os.makedirs("models", exist_ok=True)

joblib.dump(final_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Model saved to {MODEL_PATH}\nScaler saved to {SCALER_PATH}")