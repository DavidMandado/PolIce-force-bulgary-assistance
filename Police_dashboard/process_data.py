import os
import glob
import argparse

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
MONTHLY_FOLDER  = os.path.join(DATA_DIR, "2019-to-2025")
LSOA_GEOJSON    = os.path.join(DATA_DIR, "LSOAs.geojson")
WARD_GEOJSON    = os.path.join(DATA_DIR, "wards.geojson")
IMD_CSV_PATH    = os.path.join(DATA_DIR, "id-2019-for-london.csv")
POP_CSV_PATH    = os.path.join(DATA_DIR, "Mid-2021-LSOA-2021.csv")
OUTPUT_MASTER   = os.path.join(DATA_DIR, "crime_fixed_dataset.csv")

# ─── Helper: Clean column names ────────────────────────────────────────────────
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return df

# ─── Part A: Process one single‐month CSV ─────────────────────────────────────
def process_single_month(input_csv: str, output_csv: str):
    """
    Reads a single‐month raw crime CSV, cleans it, filters to London LSOAs,
    and writes out a “combined_crime_YYYY-MM.csv” that has the same columns
    (lsoa_code, month, crime_type, longitude, latitude, etc.) as the original
    monthly files in data/2019-to-2025/.
    
    You can then copy/move the output_csv into data/2019-to-2025/ to include it
    in your next full combine.
    """
    # 1) Load the raw CSV
    df = pd.read_csv(input_csv)
    df = clean_column_names(df)
    
    # 2) Filter to London LSOAs (those that start with “E01”)
    df = df[df["lsoa_code"].astype(str).str.startswith("E01")]
    
    # 3) Parse “month” column if it isn’t already a datetime; expect 'YYYY-MM' format
    #    If raw CSV has a “month” column, great. If it has a “date” column, adjust accordingly.
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    elif "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    else:
        raise ValueError("Input CSV has neither a 'month' nor a 'date' column.")
    
    # 4) Drop rows missing essential fields
    df.dropna(subset=["lsoa_code", "month", "crime_type"], inplace=True)
    df["crime_type"] = df["crime_type"].str.lower()
    
    # 5) Keep only columns needed for “monthly combine” stage:
    #    We assume the original monthly CSVs all had at least:
    #      lsoa_code, month, crime_type, longitude, latitude, plus any other raw fields.
    #    We’ll preserve: ['lsoa_code','month','crime_type','longitude','latitude', …]
    #    If your downstream combine script expects fewer or more columns, adjust here.
    cols_to_keep = [
        "lsoa_code", "month", "crime_type",
        "longitude", "latitude"
    ]
    # Also keep any extra columns that were in the original “data/2019-to-2025” CSVs:
    for c in ["falls_within", "context", "reported_by", "lsoa_code"]:  # just in case
        if c in df.columns and c not in cols_to_keep:
            cols_to_keep.append(c)
    df = df[[c for c in cols_to_keep if c in df.columns]]
    
    # 6) Write out the cleaned single‐month file
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Processed single month → saved to {output_csv}")
    print("Rows:", len(df), "| Unique LSOAs:", df["lsoa_code"].nunique())

# ─── Part B: Combine all monthly CSVs into one “XGBoost_ready_dataset.csv” ────
def combine_all_months_and_build_master():
    """
    Assumes that every CSV inside data/2019-to-2025/ is already ‘cleaned’—i.e.,
    it has standardized column names, a proper datetime “month” column, a valid
    “lsoa_code”, and so on. This function concatenates them all, builds one big
    full_df with burglary counts, features, and exports XGBoost_ready_dataset.csv.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    monthly_csvs = glob.glob(os.path.join(MONTHLY_FOLDER, "*.csv"))
    print(f"Found {len(monthly_csvs)} CSV(s) in {MONTHLY_FOLDER} to combine.")

    if len(monthly_csvs) == 0:
        print("No files to combine. Exiting.")
        return

    # 1) Read + concat all monthly files
    combined_data = pd.concat([pd.read_csv(f) for f in monthly_csvs], ignore_index=True)
    combined_data = clean_column_names(combined_data)

    # 2) Filter again to London LSOAs (in case any slipped in)
    combined_data = combined_data[combined_data["lsoa_code"].astype(str).str.startswith("E01")]

    # 3) If there’s a “date” column instead of “month”, convert:
    if "date" in combined_data.columns and "month" not in combined_data.columns:
        combined_data["month"] = pd.to_datetime(combined_data["date"], errors="coerce")
    # If “month” came in as string, ensure datetime
    if "month" in combined_data.columns:
        combined_data["month"] = pd.to_datetime(combined_data["month"], errors="coerce")

    combined_data.dropna(subset=["lsoa_code", "month", "crime_type"], inplace=True)
    combined_data["crime_type"] = combined_data["crime_type"].str.lower()

    # 4) Create a “full grid” of (lsoa_code × all months)
    all_lsoas = combined_data["lsoa_code"].unique()
    all_months = pd.date_range(combined_data["month"].min(), combined_data["month"].max(), freq="MS")
    full_index = pd.MultiIndex.from_product([all_lsoas, all_months], names=["lsoa_code", "month"])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # 5) Compute burglary_counts and total crime_counts
    burglary_counts = (
        combined_data[combined_data["crime_type"] == "burglary"]
        .groupby(["lsoa_code", "month"])
        .size()
        .reset_index(name="burglary_count")
    )
    crime_counts_total = (
        combined_data.groupby(["lsoa_code", "month"])
        .size()
        .reset_index(name="crime_count")
    )
    full_df = full_df.merge(burglary_counts, on=["lsoa_code", "month"], how="left")
    full_df = full_df.merge(crime_counts_total, on=["lsoa_code", "month"], how="left")
    full_df[["burglary_count", "crime_count"]] = (
        full_df[["burglary_count", "crime_count"]].fillna(0).astype(int)
    )

    # 6) Other crime types (pivot)
    other_crimes = (
        combined_data[combined_data["crime_type"] != "burglary"]
        .groupby(["lsoa_code", "month", "crime_type"])
        .size()
        .reset_index(name="count")
    )
    other_pivot = (
        other_crimes.pivot(index=["lsoa_code", "month"], columns="crime_type", values="count")
        .fillna(0)
        .reset_index()
    )
    full_df = full_df.merge(other_pivot, on=["lsoa_code", "month"], how="left")
    full_df.fillna(0, inplace=True)

    # 7) LSOA coordinates (mean of points)
    coords_df = (
        combined_data.dropna(subset=["longitude", "latitude"])
        .groupby("lsoa_code")[["longitude", "latitude"]]
        .mean()
        .reset_index()
    )
    full_df = full_df.merge(coords_df, on="lsoa_code", how="left")

    # 8) Stop-and-search aggregation (if present)
    #    If you already produced stop_search_counts separately, read that:
    stop_ss_path = os.path.join(DATA_DIR, "stop_search_counts.csv")
    if os.path.exists(stop_ss_path):
        stop_search_counts = pd.read_csv(stop_ss_path)
        stop_search_counts["month"] = pd.to_datetime(stop_search_counts["month"])
        full_df = full_df.merge(stop_search_counts, on=["lsoa_code", "month"], how="left")
        full_df["stop_and_search_count"] = full_df["stop_and_search_count"].fillna(0)
    else:
        full_df["stop_and_search_count"] = 0

    # 9) Compute lags & rolling stats on burglary_count
    full_df.sort_values(["lsoa_code", "month"], inplace=True)
    grouped = full_df.groupby("lsoa_code")
    for lag in [1, 2, 3, 6, 12]:
        full_df[f"lag_{lag}"] = grouped["burglary_count"].shift(lag)
    for window in [3, 6, 12]:
        full_df[f"rolling_mean_{window}"] = (
            grouped["burglary_count"].shift(1).rolling(window).mean()
        )
        full_df[f"rolling_std_{window}"] = (
            grouped["burglary_count"].shift(1).rolling(window).std()
        )
        full_df[f"rolling_sum_{window}"] = (
            grouped["burglary_count"].shift(1).rolling(window).sum()
        )

    # 10) Time features
    full_df["month_num"] = full_df["month"].dt.month
    full_df["quarter"] = full_df["month"].dt.quarter
    full_df["month_sin"] = np.sin(2 * np.pi * full_df["month_num"] / 12)
    full_df["month_cos"] = np.cos(2 * np.pi * full_df["month_num"] / 12)
    full_df["is_winter"] = full_df["month_num"].isin([12, 1, 2]).astype(int)
    full_df["is_holiday_season"] = full_df["month_num"].isin([11, 12]).astype(int)

    # 11) Merge IMD and population
    imd = pd.read_csv(IMD_CSV_PATH, delimiter=";")
    imd = clean_column_names(imd)
    imd = imd.rename(
        columns={
            "lsoa_code_(2011)": "lsoa_code",
            "index_of_multiple_deprivation_imd_decile_where_1_is_most_deprived_10_of_lsoas": "imd_decile_2019",
            "income_decile_where_1_is_most_deprived_10_of_lsoas": "income_decile_2019",
            "employment_decile_where_1_is_most_deprived_10_of_lsoas": "employment_decile_2019",
            "crime_decile_where_1_is_most_deprived_10_of_lsoas": "crime_decile_2019",
            "health_deprivation_and_disability_decile_where_1_is_most_deprived_10_of_lsoas": "health_decile_2019"
        }
    )
    full_df = full_df.merge(imd, on="lsoa_code", how="left")

    pop = pd.read_csv(POP_CSV_PATH, delimiter=";")
    pop = clean_column_names(pop)
    pop = pop.rename(columns={"lsoa_2021_code": "lsoa_code", "total": "population"})
    full_df = full_df.merge(pop[["lsoa_code", "population"]], on="lsoa_code", how="left")

    # 12) Derived features
    full_df["log_pop"] = np.log1p(full_df["population"])
    full_df["crime_per_capita"] = full_df["lag_1"] / (full_df["population"] + 1)
    full_df["stop_rate"] = full_df["stop_and_search_count"] / (full_df["population"] + 1)
    full_df["imd_pop_interaction"] = full_df["imd_decile_2019"] * full_df["log_pop"]
    fill_cols = [c for c in full_df.columns if c.startswith(("lag_", "rolling_"))]
    full_df[fill_cols] = full_df[fill_cols].fillna(0)

    # 13) Additional stop‐and‐search features
    #    If you have access to the full stop_with_lsoa GeoDataFrame (with object_of_search etc),
    #    you could repeat the weapon/drug counts here. For now, we check if those columns exist:
    if "weapon_search_count" in full_df.columns and "drug_search_count" in full_df.columns:
        pass
    else:
        full_df["weapon_search_count"] = 0
        full_df["drug_search_count"] = 0

    # 14) Clean up and drop unnecessary columns
    drop_cols = [col for col in full_df.columns if "rolling_sum_" in col] + ["month_num", "stop_rate"]
    full_df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # 15) IMD interactions
    imd_cols = [
        "imd_decile_2019", "income_decile_2019", "employment_decile_2019",
        "crime_decile_2019", "health_decile_2019"
    ]
    for col in imd_cols:
        full_df[col] = full_df[col].astype("category")
        full_df[f"{col}_x_sin"] = full_df[col].cat.codes * full_df["month_sin"]
        full_df[f"{col}_x_cos"] = full_df[col].cat.codes * full_df["month_cos"]
        full_df[f"{col}_x_quarter"] = full_df[col].cat.codes * full_df["quarter"]

    # 16) Export final dataset
    full_df.to_csv(OUTPUT_MASTER, index=False, encoding="utf-8-sig")
    print("Wrote full dataset to", OUTPUT_MASTER)
    print("Final row count:", full_df.shape[0])

# ─── Main entrypoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Either process a single‐month crime CSV or combine all into XGBoost_ready_dataset.csv"
    )
    parser.add_argument(
        "--single",
        help="Path to one raw month CSV. The script will clean it and output a cleaned version.",
        required=False
    )
    args = parser.parse_args()

    if args.single:
        # Process that single file only.
        infile = args.single
        # Create an output name in the same folder, e.g. “cleaned_2025-06.csv”
        base = os.path.basename(infile).replace(".csv", "")
        outname = f"processed_{base}.csv"
        outpath = os.path.join(os.path.dirname(infile), outname)
        process_single_month(infile, outpath)
    else:
        # No --single argument: combine everything in data/2019-to-2025/ and build master
        combine_all_months_and_build_master()
