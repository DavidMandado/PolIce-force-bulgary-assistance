import pandas as pd
import numpy as np
import geopandas as gpd
import os
import glob
from shapely.geometry import Point

# Setup
os.makedirs("data", exist_ok=True)
crime_files = glob.glob("../PolIce-force-bulgary-assistance/data/2019-to-2025/*.csv")
print(f"Found {len(crime_files)} crime files to combine.")

# Combine crime data
combined_data = pd.concat([pd.read_csv(f) for f in crime_files], ignore_index=True)
combined_data.to_csv("data/combined_crime_2019-2025.csv", index=False, encoding="utf-8-sig")
combined_data = pd.read_csv("data/combined_crime_2019-2025.csv")

# Load and clean stop and search data
stop_and_search_data = pd.read_csv("../PolIce-force-bulgary-assistance/data/stopandsearch2019.csv", skiprows=2, on_bad_lines='skip', engine="python")
stop_and_search_data.columns = stop_and_search_data.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)
stop_and_search_data["date"] = pd.to_datetime(stop_and_search_data["date"], errors="coerce")
stop_and_search_data.dropna(subset=["date", "longitude", "latitude"], inplace=True)
stop_and_search_data["month"] = stop_and_search_data["date"].dt.to_period("M").dt.to_timestamp()

# Attach LSOA to stop and search
lsoa_gdf = gpd.read_file("../PolIce-force-bulgary-assistance/data/LSOAs.geojson")
stop_and_search_data["geometry"] = stop_and_search_data.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
stop_gdf = gpd.GeoDataFrame(stop_and_search_data, geometry="geometry", crs="EPSG:4326").to_crs(lsoa_gdf.crs)
stop_with_lsoa = gpd.sjoin(stop_gdf, lsoa_gdf[["LSOA11CD", "geometry"]], how="left", predicate="within")
stop_with_lsoa.rename(columns={"LSOA11CD": "lsoa_code"}, inplace=True)

# Aggregate stop and search
stop_search_counts = stop_with_lsoa.dropna(subset=["lsoa_code"]).groupby(["lsoa_code", "month"]).size().reset_index(name="stop_and_search_count")

# Clean and process crime data
combined_data.columns = combined_data.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)
combined_data.drop(columns=["reported_by", "falls_within", "context"], errors='ignore', inplace=True)
combined_data["month"] = pd.to_datetime(combined_data["month"], format="%Y-%m")
combined_data.dropna(subset=["lsoa_code", "month", "crime_type"], inplace=True)
combined_data["crime_type"] = combined_data["crime_type"].str.lower()

# Create complete grid
all_lsoas = combined_data["lsoa_code"].unique()
all_months = pd.date_range(combined_data["month"].min(), combined_data["month"].max(), freq="MS")
full_index = pd.MultiIndex.from_product([all_lsoas, all_months], names=["lsoa_code", "month"])
full_df = pd.DataFrame(index=full_index).reset_index()

# Merge population early
pop = pd.read_csv("data/Mid-2021-LSOA-2021.csv", delimiter=";")
pop.columns = pop.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)
pop = pop.rename(columns={"lsoa_2021_code": "lsoa_code", "total": "population"})
full_df = full_df.merge(pop[["lsoa_code", "population"]], on="lsoa_code", how="left")
if "population" not in full_df.columns:
    raise KeyError("Column 'population' is missing after merge. Please check the population CSV structure.")

# Crime counts
burglary_counts = combined_data[combined_data["crime_type"] == "burglary"].groupby(["lsoa_code", "month"]).size().reset_index(name="burglary_count")
crime_counts_total = combined_data.groupby(["lsoa_code", "month"]).size().reset_index(name="crime_count")

# Merge counts
full_df = full_df.merge(burglary_counts, on=["lsoa_code", "month"], how="left")
full_df = full_df.merge(crime_counts_total, on=["lsoa_code", "month"], how="left")
full_df[["burglary_count", "crime_count"]] = full_df[["burglary_count", "crime_count"]].fillna(0).astype(int)

# Other crimes
other_crimes = combined_data[combined_data["crime_type"] != "burglary"].groupby(["lsoa_code", "month", "crime_type"]).size().reset_index(name="count")
other_pivot = other_crimes.pivot(index=["lsoa_code", "month"], columns="crime_type", values="count").fillna(0).reset_index()
full_df = full_df.merge(other_pivot, on=["lsoa_code", "month"], how="left")
full_df.fillna(0, inplace=True)

# Coordinates
lsoa_coords = combined_data.dropna(subset=["longitude", "latitude"]).groupby("lsoa_code")[["longitude", "latitude"]].mean().reset_index()
full_df = full_df.merge(lsoa_coords, on="lsoa_code", how="left")

# Merge stop and search
full_df = full_df.merge(stop_search_counts, on=["lsoa_code", "month"], how="left")
full_df["stop_and_search_count"] = full_df["stop_and_search_count"].fillna(0)

# Lags and rolling stats
full_df.sort_values(["lsoa_code", "month"], inplace=True)
grouped = full_df.groupby("lsoa_code")
for lag in [1, 2, 3, 6, 12]:
    full_df[f"lag_{lag}"] = grouped["burglary_count"].shift(lag)
for window in [3, 6, 12]:
    full_df[f"rolling_mean_{window}"] = grouped["burglary_count"].shift(1).rolling(window).mean()
    full_df[f"rolling_std_{window}"] = grouped["burglary_count"].shift(1).rolling(window).std()
    full_df[f"rolling_sum_{window}"] = grouped["burglary_count"].shift(1).rolling(window).sum()

# Derived features
full_df["delta_lag"] = full_df["lag_1"] - full_df["lag_2"]
full_df["momentum"] = full_df["lag_1"] - full_df["lag_3"]
full_df["stop_rate"] = full_df["stop_and_search_count"] / (full_df["population"] + 1)
full_df["log_pop"] = np.log1p(full_df["population"])
full_df["crime_per_capita"] = full_df["lag_1"] / (full_df["population"] + 1)

# Time features
full_df["month_num"] = full_df["month"].dt.month
full_df["quarter"] = full_df["month"].dt.quarter
full_df["month_sin"] = np.sin(2 * np.pi * full_df["month_num"] / 12)
full_df["month_cos"] = np.cos(2 * np.pi * full_df["month_num"] / 12)
full_df["is_winter"] = full_df["month_num"].isin([12, 1, 2]).astype(int)
full_df["is_holiday_season"] = full_df["month_num"].isin([11, 12]).astype(int)

# Merge IMD
imd = pd.read_csv("data/id-2019-for-london.csv", delimiter=";")
imd.columns = imd.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)
imd = imd.rename(columns={
    "lsoa_code_(2011)": "lsoa_code",
    "index_of_multiple_deprivation_imd_decile_where_1_is_most_deprived_10_of_lsoas": "imd_decile_2019",
    "income_decile_where_1_is_most_deprived_10_of_lsoas": "income_decile_2019",
    "employment_decile_where_1_is_most_deprived_10_of_lsoas": "employment_decile_2019",
    "crime_decile_where_1_is_most_deprived_10_of_lsoas": "crime_decile_2019",
    "health_deprivation_and_disability_decile_where_1_is_most_deprived_10_of_lsoas": "health_decile_2019"
})
full_df = full_df.merge(imd, on="lsoa_code", how="left")

# Export
full_df.to_csv("data/XGBoost_ready_dataset.csv", index=False, encoding="utf-8-sig")
print("Final row count:", full_df.shape[0])