import pandas as pd

df = pd.read_csv("data/burglary_next_month_forecast.csv")
df["predicted_burglary"] = pd.to_numeric(df["predicted_burglary"], errors="coerce").fillna(0)

# normalize
min_val = df["predicted_burglary"].min()
max_val = df["predicted_burglary"].max()

if max_val != min_val:
    df["predicted_burglary_norm"] = (df["predicted_burglary"] - min_val) / (max_val - min_val)
else:
    df["predicted_burglary_norm"] = 0

# save
df.to_csv("data/burglary_next_month_forecast_normalized.csv", index=False)
print(df.head(20))