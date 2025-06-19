import os
import json
import base64
import io
import numpy as np
import traceback

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from flask import Flask, send_from_directory

import pandas as pd
import geopandas as gpd
import plotly.express as px
from shapely import Point
from shapely.geometry import shape

import joblib
from scipy.stats import entropy
from helper import save_prediction

import random

# ─── Paths ────────────────────────────────────────────────────────────────────

# Centralized data folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMMUNITY_DIR = os.path.join(BASE_DIR, "community-tool")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# The “master” CSV with all LSOA × month burglary counts and features
MASTER_CSV_PATH  = os.path.join(DATA_DIR, "crime_fixed_data.csv")

# GeoJSON files (make sure these exist inside DATA_DIR)
WARD_GEOJSON     = os.path.join(DATA_DIR, "wards.geojson")
LSOA_GEOJSON     = os.path.join(DATA_DIR, "LSOAs.geojson")

# (Leave these for your predicted/allocation modes later)
PRED_CSV_PATH    = os.path.join(DATA_DIR, "burglary_next_month_forecast.csv")
ALLOC_CSV_PATH   = os.path.join(DATA_DIR, "abbey_ward_patrol_schedule.csv")
PRED_CSV_NORM_PATH = os.path.join(DATA_DIR, "burglary_next_month_forecast_normalized.csv")
PERC_CSV_NORM_PATH = os.path.join(DATA_DIR, "survey_mean_normalized.csv")

ID_DATA_PATH     = os.path.join(DATA_DIR, "ID-2019-for-London.csv")
STOP_SEARCH_PATH = os.path.join(DATA_DIR, "stopandsearch2019.csv")
MID_LSOA_PATH    = os.path.join(DATA_DIR, "Mid-2021-LSOA-2021.csv")

# model paths
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_burglary_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "robust_scaler.pkl")

# get model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ─── 1) Read both GeoJSONs into Python dicts ─────────────────────────────────

ward_gdf = gpd.read_file(WARD_GEOJSON).to_crs(epsg=4326)
# convert back to GeoJSON dict for Plotly
ward_geo = json.loads(ward_gdf.to_json())

# ─── Read & reproject LSOA boundaries into EPSG:4326 ────────────────────────
lsoa_gdf = gpd.read_file(LSOA_GEOJSON).to_crs(epsg=4326)
lsoa_geo = json.loads(lsoa_gdf.to_json())


print("── Sample ward_geo.properties keys:", ward_geo["features"][0]["properties"].keys())
print("── Sample ward_geo.properties (first feature):", ward_geo["features"][0]["properties"])

lsoa_to_ward = {}

# First, build a list of (ward_code, shapely_polygon) for all wards
ward_polygons = [
    (
        feat["properties"]["GSS_Code"],
        shape(feat["geometry"])
    )
    for feat in ward_geo["features"]
]

# Now, for each LSOA, find its centroid and test which ward polygon contains it
for feat in lsoa_geo["features"]:
    lsoa_code = feat["properties"]["LSOA11CD"]
    lsoa_centroid = shape(feat["geometry"]).centroid

    # Look for the ward that contains this centroid
    for ward_code, ward_poly in ward_polygons:
        if ward_poly.contains(lsoa_centroid):
            lsoa_to_ward[lsoa_code] = ward_code
            break

# (Optional debug print: how many LSOAs mapped successfully)
print(f"▶ Precomputed mapping for {len(lsoa_to_ward)} LSOAs → ward codes.")


# ─── 3) Build a ward_code ⇄ ward_name dictionary (for “search by name”) ─────
ward_mapping = {
    feat["properties"]["GSS_Code"]: feat["properties"]["Name"]
    for feat in ward_geo["features"]
}
name_to_code = {name.lower(): code for code, name in ward_mapping.items()}


# ─── 4) Start Dash App ──────────────────────────────────────────────────────
server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/police-dashboard/',
    requests_pathname_prefix='/police-dashboard/'
)

@server.route("/")
def server_index():
    return send_from_directory(COMMUNITY_DIR, "index.html")

@server.route('/<path:path>')
def serve_community_static(path):
    return send_from_directory(COMMUNITY_DIR, path)

@server.route("/police-dashboard/api/crime-data")
def crime_data():
    return send_from_directory(DATA_DIR, "crime_fixed_data.csv")

@server.route("/police-dashboard/api/lookup")
def lookup():
    return send_from_directory(DATA_DIR, "lsoa_to_ward.json")

# CSS styles
SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "300px", "padding": "20px", "background-color": "#f8f9fa",
    "overflow": "auto", "transition": "transform 0.3s ease"
}
SIDEBAR_HIDDEN = {**SIDEBAR_STYLE, "transform": "translateX(-100%)"}

CONTENT_STYLE   = {"margin-left": "320px", "margin-right": "20px", "padding": "20px"}
MAP_CONTAINER   = {"display": "flex", "flexDirection": "row"}
HALF_MAP_STYLE  = {"width": "50%", "height": "80vh"}
FULL_MAP_STYLE  = {"width": "100%", "height": "80vh"}


app.layout = html.Div([

    dcc.Store(id="selected-ward", data=None),
    dcc.Store(id="sidebar-open", data=True),
    dcc.Store(id="show-perception", data=False),

    # ── Toggle filters button ─────────────────────────────────────────────────
    html.Button(
        "☰ Filters",
        id="btn-toggle",
        n_clicks=0,
        style={"position": "fixed", "top": "10px", "left": "10px", "zIndex": 1000}
    ),

    # ── Sidebar ────────────────────────────────────────────────────────────────
    html.Div(
        id="sidebar",
        children=[

            html.H2("Filters", style={"margin-top": "20px"}),
            html.Hr(),

            # Data View: Past / Predicted / Allocation
            html.Label("Data View"),
            dcc.RadioItems(
                id="data-mode",
                options=[
                    {"label": "Past Data",      "value": "past"},
                    {"label": "Predicted Data", "value": "pred"},
                    {"label": "Predicted vs Perceived", "value": "pred_vs_perceived"},
                ],
                value="past",
                labelStyle={"display": "block"}
            ),
            html.Br(),

            # View Level (Ward vs LSOA)
            html.Label("View Level"),
            dcc.Dropdown(
                id="level",
                options=[
                    {"label": "Ward", "value": "ward"},
                    {"label": "LSOA", "value": "lsoa"},
                ],
                value="ward",
                clearable=False
            ),
            html.Br(),

            # ── Past Controls (only when “Past Data” is chosen) ───────────────────
            html.Div(
                id="past-controls",
                children=[
                    html.Label("Date Range (Year)"),
                    dcc.RangeSlider(
                        id="past-range",
                        min=2021,
                        max=2025,
                        step=1,
                        marks={y: str(y) for y in range(2021, 2026)},
                        value=[2021, 2025]
                    ),
                ]
            ),

            # ── Predicted Controls (only when “Predicted Data”) ─────────────────

            html.Div(
                id="pred-controls",
                children=[
                    html.Br(),
                    html.Label("Upload New Monthly CSV:"),               # 1️⃣ Upload first
                    dcc.Upload(
                        id="upload-file",
                        children=html.Div(["Drag & Drop or ", html.A("Select CSV")]),
                        style={
                            "width": "100%", "height": "40px",
                            "borderWidth": "1px", "borderStyle": "dashed",
                            "borderRadius": "5px", "textAlign": "center"
                        }
                    ),
                    dcc.Loading(
                        id="loading-upload",
                        type="dot",
                        children=html.Div(id="upload-status")
                    ),
                    dcc.Store(id="upload-done", data=False),
                    html.Br(), html.Br(),
                    html.Button(                                        # 2️⃣ Then Predict
                        "Predict Next Month",
                        id="predict-button",
                        n_clicks=0,
                        style={"width": "100%", "background-color": "#007bff", "color": "white"}
                    ),
                    html.Br(), html.Br(),
                    html.Button(
                        "Download Schedule CSV",
                        id="Schedule Button",
                        n_clicks=0,
                        style={"width": "100%"}
                    ),
                    dcc.Download(id="download-schedule"),
                ],
                style={"display": "none"}
            ),

            # ── Allocation Controls ──────────────────────────────────────────────
            html.Div(
                id="alloc-controls",
                children=[
                    html.P("Allocation view will appear below as a table.")
                ],
                style={"display": "none"}
            ),

            html.Hr(),

            # ── Search by Ward Name or Code ────────────────────────────────────
            html.Label("Search Ward by Name or Code"),
            dcc.Input(
                id="ward-search-input",
                type="text",
                placeholder="e.g. Camden Town or E05000405",
                style={"width": "70%"}
            ),
            html.Button(
                "Go",
                id="ward-search-button",
                n_clicks=0,
                style={"margin-left": "10px"}
            ),
            html.Br(), html.Br(),

            # ── Back Button (when a ward is selected) ──────────────────────────
            html.Button(
                "← Back to Wards",
                id="back-button",
                n_clicks=0,
                style={"display": "none", "width": "100%"}
            ),
            html.Br(), html.Br(),

            # ── Apply Button ───────────────────────────────────────────────────
            html.Button(
                "Apply",
                id="apply-button",
                n_clicks=0,
                style={"width": "100%"}
            ),
            
            html.Button("Perception Analysis",
                        id="btn-perception",
                        n_clicks=0,
                        style={"width":"100%", "marginBottom":"1em"}),
        ],
        style=SIDEBAR_STYLE
    ),

    # ── Main content: maps + (optional) allocation table ───────────────────────
    html.Div(
        id="page-content",
        style=CONTENT_STYLE,
        children=[
            html.Div(
                id="map-container",
                style=MAP_CONTAINER,
                children=[
                    dcc.Graph(id="map-ward", style=FULL_MAP_STYLE),
                    dcc.Graph(id="map-lsoa", style={**HALF_MAP_STYLE, "display": "none"})
                ]
            ),
            html.Div(
                id="allocation-table-container",
                style={"margin-top": "20px"}
            )
        ]
    ),
    
        # ─── Perception Analysis “modal” ────────────────────────────────────────
    html.Div(
        id="perception-window",
    style={
        "position": "fixed",
        "top": "5%",
        "left": "10%",
        "width": "80%",
        "height": "90vh",              # full viewport height
        "backgroundColor": "white",
        "zIndex": 2000,
        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
        "display": "none",             # still hidden by default
        "padding": "1rem",

        # make this a flex container
        # "display": "flex",
        # "flexDirection": "column",
    },
        children=[
            html.Button("Close", id="close-perception", style={"float":"right"}),
            html.H3("Mean Sentiment Score by Topic"),
            dcc.Graph(id="perception-graph")
        ]
    )
])

# ─── 5) Callbacks ─────────────────────────────────────────────────────────────

# 5.1) Show/hide Past / Predicted / Allocation controls
@app.callback(
    Output("past-controls", "style"),
    Output("pred-controls", "style"),
    Output("alloc-controls", "style"),
    Input("data-mode", "value")
)
def toggle_mode_controls(mode):
    if mode == "past":
        return {}, {"display": "none"}, {"display": "none"}
    elif mode == "pred":
        return {"display": "none"}, {}, {"display": "none"}
    elif mode == "pred_vs_perceived":
        return {"display": "none"}, {}, {"display": "none"}
    elif mode == "alloc":
        return {"display": "none"}, {"display": "none"}, {}
    return {}, {"display": "none"}, {"display": "none"}


# 5.2) Handle upload of a new monthly CSV (just append raw rows)
@app.callback(
    Output("upload-file", "children"),
    Output("upload-file", "contents"),
    Output("upload-status", "children"),
    Input("upload-file", "contents"),
    State("upload-file", "filename"),
)
def handle_upload(contents, filename):    
    if contents is None:
        raise PreventUpdate
    if not filename.endswith(".csv"):
        return html.Div("Please upload a valid CSV file."), None,  ""

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df_new = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        print("df_new created")
        clean_df = clean_new_dataset(df_new)
        print("clean_df created")
        
        # Append to master CSV
        if not os.path.exists(MASTER_CSV_PATH):
            return html.Div(f"Master CSV not found at {MASTER_CSV_PATH}."), None, ""
        
        df_master = pd.read_csv(MASTER_CSV_PATH)
        clean_df = clean_df[df_master.columns]
        
        print("Difference: ", set(df_master.columns) - set(clean_df.columns))
        if sorted(df_master.columns) != sorted(clean_df.columns):
            return html.Div("Uploaded CSV columns do not match master CSV columns."), None, ""

        # Ensure no duplicates
        clean_df["month"] = pd.to_datetime(clean_df["month"])
        df_master["month"] = pd.to_datetime(df_master["month"])

        prev_len_clean = len(clean_df)

        # Remove rows from clean_df that already exist in df_master (by lsoa_code + month)
        existing_index = df_master.set_index(["lsoa_code", "month"]).index
        clean_df = clean_df[~clean_df.set_index(["lsoa_code", "month"]).index.isin(existing_index)]
        # print if any rows were removed
        if len(clean_df) < prev_len_clean:
            return html.Div("Data already exists, no new rows added."), None, ""
        
        update_model_with_new_data(clean_df)
        print("model updated")
        df_master = pd.concat([df_master, clean_df], ignore_index=True)
        df_master.to_csv(MASTER_CSV_PATH, index=False, mode='w')

        return html.Div("New data uploaded successfully."), None, ""
    except Exception as e:
        print("Error details:", e)
        traceback.print_exc()
        return html.Div("Error: could not read uploaded CSV."), None, ""

@app.callback(
    Output("sidebar", "style"),
    Output("page-content", "style"),
    Input("btn-toggle", "n_clicks"),
    State("sidebar-open", "data")
)
def toggle_sidebar(n, open_):
    if n:
        if open_:
            return SIDEBAR_HIDDEN, {**CONTENT_STYLE, "margin-left": "20px"}
        else:
            return SIDEBAR_STYLE, CONTENT_STYLE
    return SIDEBAR_STYLE, CONTENT_STYLE

@app.callback(
    Output("sidebar-open", "data"),
    Input("sidebar", "style")
)
def store_sidebar_state(style):
    return style.get("transform") != "translateX(-100%)"

@app.callback(
    Output("selected-ward", "data"),
    Output("back-button", "style"),
    Input("map-ward", "clickData"),
    Input("back-button", "n_clicks"),
    Input("ward-search-button", "n_clicks"),
    State("ward-search-input", "value"),
    State("data-mode", "value"),
)
def handle_selection(map_click, back_click, search_click, search_value, mode):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered == "map-ward" and map_click:
        code = map_click["points"][0]["location"]
        return {"code": code, "mode": mode}, {"display": "block", "width": "100%"}

    if triggered == "back-button":
        return None, {"display": "none"}

    if triggered == "ward-search-button" and search_value:
        query = search_value.strip()
        if query.upper() in ward_mapping:
            code = query.upper()
        else:
            code = name_to_code.get(query.lower())
        if not code:
            raise PreventUpdate
        return {"code": code, "mode": mode}, {"display": "block", "width": "100%"}

    raise PreventUpdate


@app.callback(
    Output("show-perception", "data"),
    Input("btn-perception", "n_clicks"),
    State("show-perception", "data"),
    prevent_initial_call=True
)
def toggle_perception(n, showing):
    if n:
        # flip it on click
        return not showing
    return showing

# ─── Toggle the Perception window open/closed ─────────────────────────────
@app.callback(
    Output("perception-window", "style"),
    Input("btn-perception", "n_clicks"),
    Input("close-perception", "n_clicks"),
    State("perception-window", "style"),
)
def toggle_perception_window(open_clicks, close_clicks, current_style):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    style = current_style.copy()
    if trigger == "btn-perception":
        style["display"] = "block"
    else:
        style["display"] = "none"
    return style

@app.callback(
    Output("predict-button", "n_clicks"),  # reset the button
    Input("predict-button", "n_clicks"),
)
def predict_month(n_clicks):
    if n_clicks == 0:
        raise PreventUpdate

    try:
        month = (pd.Timestamp.now() + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        print("Predicting for month:", month)
        save_prediction(model, scaler, month)
        return 0
    except Exception as e:
        print("Prediction error:", e)
        return 0
    
@app.callback(
    Output("perception-graph", "figure"),
    Input("btn-perception","n_clicks"),
    Input("show-perception", "data"),
)
def perception_callback(n_clicks, show_perc):
    if n_clicks == 0:
        raise PreventUpdate
    if not show_perc:
        raise PreventUpdate

    try:
        fig = build_perception_figure()
        return fig
    except Exception as e:
        print("Error building perception figure:", e)
        return px.Figure()

@app.callback(
    Output("map-ward", "figure"),
    Output("map-lsoa", "figure"),
    Output("map-ward", "style"),
    Output("map-lsoa", "style"),
    Output("allocation-table-container", "children"),
    Input("apply-button", "n_clicks"),
    Input("predict-button", "n_clicks"),
    Input("data-mode", "value"),
    Input("selected-ward", "data"),
    State("level", "value"),
    State("past-range", "value"),
)
def unified_map_callback(apply_clicks, predict_clicks, mode, selected_ward, level, past_range):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if mode == "past" and trigger_id in ["apply-button", "data-mode", "selected-ward"]:
        return generate_map("past", selected_ward, level, past_range)

    if mode == "pred" and trigger_id in ["data-mode", "predict-button", "selected-ward"]:
        return generate_map("pred", selected_ward, level)
    
    if mode == "pred_vs_perceived":
        return generate_map("pred_vs_perceived", selected_ward, level)

    raise PreventUpdate

def build_perception_figure():
    sentiment_summary = pd.read_csv(os.path.join(DATA_DIR, "topic_sentiment_summary.csv"))
    
    topic_sentiment = (
        sentiment_summary.groupby('matched_topics')
        .agg(mean_sentiment=('avg_sentiment', 'mean'))
        .reset_index()
        .sort_values(by='mean_sentiment', ascending=True)
    )
    
    fig = px.bar(
        topic_sentiment,
        x='matched_topics',
        y='mean_sentiment',
        title='Mean Sentiment Score by Topic',
        labels={'matched_topics': 'Topic', 'mean_sentiment': 'Mean Sentiment'}
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title='Mean Sentiment Score',
        height=500,
        margin={"r":0,"t":60,"l":0,"b":0},
        template="plotly_white",

    )
    
    return fig


def update_model_with_new_data(new_df: pd.DataFrame):
    exclude_cols = {
        "lsoa_code", "month", "year_month", "crime_type",
        "latitude", "longitude", "burglary_count", "crime_count",
        'crime_score', 'index_of_multiple_deprivation_imd_score', 'employment_score_rate',
        'local_authority_district_code_2019', 'local_authority_district_name_2019',
        'income_score_rate', 'barriers_to_housing_and_services_score', 'education_skills_and_training_score',
        'quarter', 'month_num', 'lsoa_name', 'health_deprivation_and_disability_score', 'living_environment_score',
    }
    features = [c for c in new_df.columns if c not in exclude_cols]
    target_col = "burglary_count"
    
    trained_features = scaler.feature_names_in_.tolist()
    X_sorted_cols = new_df[trained_features]

    X_new = scaler.transform(X_sorted_cols)
    y_new = new_df[target_col].values
    model.fit(X_new, y_new, xgb_model=model)
    joblib.dump(model, MODEL_PATH)
    print(f"Model updated and saved to {MODEL_PATH}")

def clean_new_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    # Filter for London LSOAs
    df = df[df["lsoa_code"].astype(str).str.startswith("E01")]
    # print("Filtered to London LSOAs:", df["lsoa_code"].nunique(), "unique LSOAs remaining")

    # Load and clean stop and search data
    stop_and_search_data = pd.read_csv(STOP_SEARCH_PATH, skiprows=2, on_bad_lines='skip', engine="python")
    stop_and_search_data.columns = stop_and_search_data.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)
    stop_and_search_data["date"] = pd.to_datetime(stop_and_search_data["date"], errors="coerce")
    stop_and_search_data.dropna(subset=["date", "longitude", "latitude"], inplace=True)
    stop_and_search_data["month"] = stop_and_search_data["date"].dt.to_period("M").dt.to_timestamp()

    # Attach LSOA to stop and search
    lsoa_gdf = gpd.read_file(LSOA_GEOJSON)
    stop_and_search_data["geometry"] = stop_and_search_data.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    stop_gdf = gpd.GeoDataFrame(stop_and_search_data, geometry="geometry", crs="EPSG:4326").to_crs(lsoa_gdf.crs)
    stop_with_lsoa = gpd.sjoin(stop_gdf, lsoa_gdf[["LSOA11CD", "geometry"]], how="left", predicate="within")
    stop_with_lsoa.rename(columns={"LSOA11CD": "lsoa_code"}, inplace=True)

    # Aggregate stop and search
    stop_search_counts = stop_with_lsoa.dropna(subset=["lsoa_code"]).groupby(["lsoa_code", "month"]).size().reset_index(name="stop_and_search_count")

    # Clean and process crime data
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)
    df.drop(columns=["reported_by", "falls_within", "context"], errors='ignore', inplace=True)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df.dropna(subset=["lsoa_code", "month", "crime_type"], inplace=True)
    df["crime_type"] = df["crime_type"].str.lower()

    # Create complete grid
    all_lsoas = df["lsoa_code"].unique()
    all_months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    full_index = pd.MultiIndex.from_product([all_lsoas, all_months], names=["lsoa_code", "month"])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Crime counts
    burglary_counts = df[df["crime_type"] == "burglary"].groupby(["lsoa_code", "month"]).size().reset_index(name="burglary_count")
    crime_counts_total = df.groupby(["lsoa_code", "month"]).size().reset_index(name="crime_count")

    # Merge population early
    pop = pd.read_csv(MID_LSOA_PATH, delimiter=";")
    pop.columns = pop.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)
    pop = pop.rename(columns={"lsoa_2021_code": "lsoa_code", "total": "population"})
    full_df = full_df.merge(pop[["lsoa_code", "population"]], on="lsoa_code", how="left")
    if "population" not in full_df.columns:
        raise KeyError("Column 'population' is missing after merge. Please check the population CSV structure.")

    # Merge counts
    full_df = full_df.merge(burglary_counts, on=["lsoa_code", "month"], how="left")
    full_df = full_df.merge(crime_counts_total, on=["lsoa_code", "month"], how="left")
    full_df[["burglary_count", "crime_count"]] = full_df[["burglary_count", "crime_count"]].fillna(0).astype(int)

    # Other crimes
    other_crimes = df[df["crime_type"] != "burglary"].groupby(["lsoa_code", "month", "crime_type"]).size().reset_index(name="count")
    other_pivot = other_crimes.pivot(index=["lsoa_code", "month"], columns="crime_type", values="count").fillna(0).reset_index()
    full_df = full_df.merge(other_pivot, on=["lsoa_code", "month"], how="left")
    full_df.fillna(0, inplace=True)

    # Coordinates
    lsoa_coords = df.dropna(subset=["longitude", "latitude"]).groupby("lsoa_code")[["longitude", "latitude"]].mean().reset_index()
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

    # Merge IMD and population
    imd = pd.read_csv(ID_DATA_PATH, delimiter=";")
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

    # Derived features
    full_df["imd_pop_interaction"] = full_df["imd_decile_2019"] * full_df["log_pop"]
    fill_cols = [c for c in full_df.columns if c.startswith(("lag_", "rolling_", "delta_", "momentum"))]
    full_df[fill_cols] = full_df[fill_cols].fillna(0)

    # Object of search feature engineering
    stop_with_lsoa["object_of_search"] = stop_with_lsoa["object_of_search"].str.lower()
    weapon_counts = stop_with_lsoa[stop_with_lsoa["object_of_search"].str.contains("weapon", na=False)].groupby(["lsoa_code", "month"]).size().reset_index(name="weapon_search_count")
    drug_counts = stop_with_lsoa[stop_with_lsoa["object_of_search"].str.contains("drug", na=False)].groupby(["lsoa_code", "month"]).size().reset_index(name="drug_search_count")
    full_df = full_df.merge(weapon_counts, on=["lsoa_code", "month"], how="left")
    full_df = full_df.merge(drug_counts, on=["lsoa_code", "month"], how="left")
    full_df[["weapon_search_count", "drug_search_count"]] = full_df[["weapon_search_count", "drug_search_count"]].fillna(0).astype(int)

    # Drop unnecessary columns
    # full_df.drop(columns=[col for col in full_df.columns if "rolling_sum_" in col] + ["month_num"], inplace=True, errors="ignore")

    # IMD interactions
    imd_cols = ["imd_decile_2019", "income_decile_2019", "employment_decile_2019", "crime_decile_2019", "health_decile_2019"]
    for col in imd_cols:
        full_df[col] = full_df[col].astype("category")
        full_df[f"{col}_x_sin"] = full_df[col].cat.codes * full_df["month_sin"]
        full_df[f"{col}_x_cos"] = full_df[col].cat.codes * full_df["month_cos"]
        full_df[f"{col}_x_quarter"] = full_df[col].cat.codes * full_df["quarter"]
        
    full_df["month"] = pd.to_datetime(full_df["month"])
    full_df["year_month"] = full_df["month"].dt.to_period("M")
    full_df = full_df[full_df["burglary_count"].notna() & (full_df["burglary_count"] >= 0)].copy()
    full_df.sort_values(["lsoa_code", "month"], inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    # add time features
    for lag in [1, 3, 6, 12]:
        col = f"crime_count_pct_change_{lag}m"
        full_df[col] = full_df.groupby("lsoa_code")["crime_count"].pct_change(lag)
        full_df[col] = full_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    if "crime_count_lag_1m" not in full_df.columns:
        full_df["crime_count_lag_1m"] = full_df.groupby("lsoa_code")["crime_count"].shift(1).fillna(0)

    if "crime_count_lag_3m" not in full_df.columns:
        full_df["crime_count_lag_3m"] = full_df.groupby("lsoa_code")["crime_count"].shift(3).fillna(0)

    # add interaction features
    full_df["lag1_crime_x_pop"] = full_df["crime_count_lag_1m"] * full_df["population"]
    full_df["lag3_crime_x_imd"] = full_df["crime_count_lag_3m"] * full_df["imd_decile_2019"].astype(float)

    full_df["crime_volatility_3m"] = (
        full_df.groupby("lsoa_code")["crime_count"]
        .transform(lambda x: x.rolling(3, min_periods=1).std())
        .fillna(0)
    )

    crime_types = [c for c in full_df.columns if c.startswith("crime_") and c.endswith("_count") and c != "burglary_count"]
    crime_data = full_df[crime_types].to_numpy()

    # Normalize rows (avoid division by zero)
    row_sums = crime_data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # to avoid division by zero
    prob_matrix = crime_data / row_sums

    # Compute entropy for each row
    full_df["crime_entropy"] = entropy(prob_matrix.T, base=np.e)

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

    full_df["months_since_burglary"] = full_df.groupby("lsoa_code")["burglary_count"].transform(time_since_burglary).fillna(100)

    # extended interactions
    full_df["lag1_x_entropy"] = full_df["crime_count_lag_1m"] * full_df["crime_entropy"]
    full_df["lag3_x_entropy"] = full_df["crime_count_lag_3m"] * full_df["crime_entropy"]

    full_df["entropy_x_sin"] = full_df["crime_entropy"] * full_df["month_sin"]
    full_df["entropy_x_cos"] = full_df["crime_entropy"] * full_df["month_cos"]

    full_df["entropy_x_imd2019"] = full_df["crime_entropy"] * full_df["imd_decile_2019"].astype(float)
    full_df["volatility_x_sin"] = full_df["crime_volatility_3m"] * full_df["month_sin"]
    full_df["volatility_x_cos"] = full_df["crime_volatility_3m"] * full_df["month_cos"]

    full_df["stop_x_imd2019"] = full_df["stop_and_search_count"] * full_df["imd_decile_2019"].astype(float)
    full_df["imd2019_x_msb"] = full_df["imd_decile_2019"].astype(float) * full_df["months_since_burglary"]

    return full_df

def generate_map(mode, selected_ward, level, past_range=None):
    df = pd.read_csv(MASTER_CSV_PATH, parse_dates=["month"])

    # resolve selected ward object
    if isinstance(selected_ward, dict):
        selected_code = selected_ward.get("code")
        selected_mode = selected_ward.get("mode")
        if selected_mode != mode:
            selected_code = None
    else:
        selected_code = selected_ward


    if mode == "past":
        y0, y1 = int(past_range[0]), int(past_range[1])
        df = df[(df.month.dt.year >= y0) & (df.month.dt.year <= y1)]

    df = df[df.lsoa_code.isin(lsoa_to_ward)]
    if df.empty:
        blank = px.choropleth_map(
            pd.DataFrame({"code":[], "count":[]}),
            geojson=ward_geo, featureidkey="properties.GSS_Code",
            locations="code", color="count",
            map_style="open-street-map",
            center={"lat":51.5074,"lon":-0.1278}, zoom=10
        )
        blank.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return blank, blank, FULL_MAP_STYLE, {"display":"none"}, html.Div()

    if mode == "pred":
        df_pred = pd.read_csv(PRED_CSV_PATH)

        if level == "ward":
            df_pred["ward_code"] = df_pred.lsoa_code.map(lsoa_to_ward)
            wc = (
                df_pred.groupby("ward_code")["predicted_burglary"]
                .sum().reset_index(name="count")
            )
            all_w = [f["properties"]["GSS_Code"] for f in ward_geo["features"]]
            dfw = pd.DataFrame({"code": all_w}).merge(
                wc.rename(columns={"ward_code":"code"}), on="code", how="left"
            )
            dfw["count"] = dfw["count"].fillna(0).astype(int)

            ward_fig = px.choropleth_map(
                dfw, geojson=ward_geo, featureidkey="properties.GSS_Code",
                locations="code", color="count",
                color_continuous_scale="oryel", opacity=0.7,
                map_style="open-street-map",
                center={"lat":51.5074, "lon":-0.1278}, zoom=10,
                labels={"count":"Predicted Burglaries"},
            )
            ward_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

            if not selected_code:
                return ward_fig, ward_fig, FULL_MAP_STYLE, {"display":"none"}, html.Div()

        # if selected
        ward_geom = next(
            f["geometry"] for f in ward_geo["features"]
            if f["properties"]["GSS_Code"] == selected_code
        )
        feats = [
            f for f in lsoa_geo["features"]
            if shape(ward_geom).contains(shape(f["geometry"]).centroid)
        ]
        geo_l = {"type":"FeatureCollection","features":feats}
        lcodes = [f["properties"]["LSOA11CD"] for f in feats]
        fl = (
            df_pred[df_pred.lsoa_code.isin(lcodes)]
            .groupby("lsoa_code")["predicted_burglary"]
            .sum().reset_index(name="count")
        ).rename(columns={"lsoa_code":"code"})

        minx, miny, maxx, maxy = shape(ward_geom).bounds
        center_l = {"lat":(miny+maxy)/2, "lon":(minx+maxx)/2}

        lsoa_fig = px.choropleth_map(
            fl, geojson=geo_l, featureidkey="properties.LSOA11CD",
            locations="code", color="count", opacity=0.7,
            color_continuous_scale="oryel",
            map_style="open-street-map",
            center=center_l, zoom=12,
            labels={"count":"Burglary Count"},
        )
        lsoa_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        alloc_path = os.path.join(DATA_DIR, "allocations", f"All_wards_patrol_schedule.csv")
        if os.path.exists(alloc_path):
            df_alloc = get_ward_schedule(map_wards_function(selected_code))
            alloc_table = dash_table.DataTable(
                data=df_alloc.to_dict("records"),
                columns=[{"name":c,"id":c} for c in df_alloc.columns],
                style_table={"overflowX":"auto"},
                style_cell={"padding":"4px","textAlign":"left"}
            )
        else:
            alloc_table = html.Div(f"No allocation file for {selected_code}", style={"color":"red"})

        return (
            ward_fig,
            lsoa_fig,
            HALF_MAP_STYLE,
            HALF_MAP_STYLE,
            alloc_table
        )
        
    if mode == "pred_vs_perceived":
        df_pred = pd.read_csv(PRED_CSV_NORM_PATH)
        df_perc = pd.read_csv(PERC_CSV_NORM_PATH)

        if "predicted_burglary_norm" not in df_pred.columns:
            raise ValueError("Missing 'predicted_burglary_norm' in predicted data.")
        if "LSOA code" not in df_perc.columns or "survey_normalized" not in df_perc.columns:
            raise ValueError("Perceived data must contain 'LSOA code' and 'survey_normalized' columns.")

        df_merged = df_pred.merge(
            df_perc.rename(columns={"LSOA code": "lsoa_code", "survey_normalized": "perceived_burglary"}),
            on="lsoa_code",
            how="left"
        )
        df_merged["gap"] = df_merged["predicted_burglary_norm"] - df_merged["perceived_burglary"]

        if level == "ward":
            df_merged["ward_code"] = df_merged["lsoa_code"].map(lsoa_to_ward)
            wc = (
                df_merged.groupby("ward_code")["gap"]
                .mean().reset_index(name="count")
            )
            
            all_w = [f["properties"]["GSS_Code"] for f in ward_geo["features"]]
            dfw = pd.DataFrame({"code": all_w}).merge(
                wc.rename(columns={"ward_code": "code"}), on="code", how="left"
            )
            dfw["count"] = dfw["count"].fillna(0)

            fig = px.choropleth_map(
                dfw, geojson=ward_geo, featureidkey="properties.GSS_Code",
                locations="code", color="count",
                color_continuous_scale="RdBu", opacity=0.7,
                map_style="open-street-map",
                center={"lat": 51.5074, "lon": -0.1278}, zoom=10,
                labels={"count": "Predicted - Perceived"},
            )
            
        else:
            df_lsoa = df_merged[["lsoa_code", "gap"]].rename(columns={"lsoa_code": "code", "gap": "count"})
            
            fig = px.choropleth_map(
                df_lsoa, geojson=lsoa_geo, featureidkey="properties.LSOA11CD",
                locations="code", color="count",
                color_continuous_scale="RdBu", opacity=0.7,
                map_style="open-street-map",
                center={"lat": 51.5074, "lon": -0.1278}, zoom=10,
                labels={"count": "Predicted - Perceived"},
            )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig, fig, FULL_MAP_STYLE, {"display": "none"}, html.Div()

    # ─────────────────────── Past mode
    if level == "ward":
        df["ward_code"] = df.lsoa_code.map(lsoa_to_ward)
        wc = (
            df.groupby("ward_code")["burglary_count"]
            .sum().reset_index(name="count")
        )
        all_w = [f["properties"]["GSS_Code"] for f in ward_geo["features"]]
        dfw = pd.DataFrame({"code": all_w}).merge(
            wc.rename(columns={"ward_code":"code"}), on="code", how="left"
        )
        dfw["count"] = dfw["count"].fillna(0).astype(int)

        ward_fig = px.choropleth_map(
            dfw, geojson=ward_geo, featureidkey="properties.GSS_Code",
            locations="code", color="count", opacity=0.7,
            color_continuous_scale="oryel",
            map_style="open-street-map",
            center={"lat":51.5074,"lon":-0.1278}, zoom=10,
            labels={"count":"Burglary Count"},
        )
        ward_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        if not selected_code:
            return (
                ward_fig,
                ward_fig,
                FULL_MAP_STYLE,
                {"display":"none"},
                html.Div()
            )

        ward_geom = next(
            f["geometry"] for f in ward_geo["features"]
            if f["properties"]["GSS_Code"] == selected_code
        )
        feats = [
            f for f in lsoa_geo["features"]
            if shape(ward_geom).contains(shape(f["geometry"]).centroid)
        ]
        geo_l = {"type":"FeatureCollection","features":feats}
        lcodes = [f["properties"]["LSOA11CD"] for f in feats]
        fl = (
            df[df.lsoa_code.isin(lcodes)]
            .groupby("lsoa_code")["burglary_count"]
            .sum().reset_index(name="count")
        ).rename(columns={"lsoa_code":"code"})

        minx, miny, maxx, maxy = shape(ward_geom).bounds
        center_l = {"lat":(miny+maxy)/2, "lon":(minx+maxx)/2}

        lsoa_fig = px.choropleth_map(
            fl, geojson=geo_l, featureidkey="properties.LSOA11CD",
            locations="code", color="count", opacity=0.7,
            color_continuous_scale="oryel",
            map_style="open-street-map",
            center=center_l, zoom=12,
            labels={"count":"Burglary Count"},
        )
        lsoa_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        return (
            ward_fig,
            lsoa_fig,
            HALF_MAP_STYLE,
            HALF_MAP_STYLE,
            html.Div()
        )

    # ─────────────── Full LSOA view
    df_ls = (
        df.groupby("lsoa_code")["burglary_count"]
          .sum().reset_index(name="count")
          .rename(columns={"lsoa_code":"code"})
    )
    lsoaf = px.choropleth_map(
        df_ls, geojson=lsoa_geo, featureidkey="properties.LSOA11CD",
        locations="code", color="count", opacity=0.7,
        color_continuous_scale="oryel",
        map_style="open-street-map",
        center={"lat":51.5074,"lon":-0.1278}, zoom=10,
        labels={"count":"Burglary Count"},
    )
    lsoaf.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return (
        lsoaf,                # map-ward (hidden)
        lsoaf,                # map-lsoa
        {"display":"none"},   # ward style
        FULL_MAP_STYLE,       # lsoa style
        html.Div()            # empty alloc container
    )

@app.callback(
    Output("Schedule Button", "children"),
    Input("selected-ward", "data")
)
def update_button_label(selected_ward):
    if not selected_ward:
        return "Download All Ward Schedules"
    else:
        ward_code = selected_ward["code"]
        return f"Download {ward_mapping.get(ward_code, ward_code)} Schedule"

@app.callback(
    Output("download-schedule", "data"),
    Input("Schedule Button", "n_clicks"),
    State("selected-ward", "data"),
    prevent_initial_call=True
)
def download_schedule(n_clicks, selected_ward):
    if n_clicks == 0:
        raise PreventUpdate
    
    if not selected_ward:
        df = pd.read_csv(os.path.join(DATA_DIR, "allocations", "All_wards_patrol_schedule.csv"))
        return dcc.send_data_frame(df.to_csv, "All_wards_patrol_schedule.csv", index=False)
    else:
        selected_ward_name = map_wards_function(selected_ward.get("code"))
        ward_df = get_ward_schedule(selected_ward_name)
        filename = f"{selected_ward_name}_patrol_schedule.csv"
        return dcc.send_data_frame(ward_df.to_csv, filename, index=False)
    
def map_wards_function(selected_ward):
    print(f"Selected ward: {selected_ward}")
    return ward_gdf[ward_gdf['GSS_Code'] == selected_ward]['Name'].values[0]

def get_ward_schedule(Ward_name): #This only works if Generate_schedules has run already
    full_schedule_df = pd.read_csv(os.path.join(DATA_DIR, "allocations", f"All_wards_patrol_schedule.csv"))
    first_col = full_schedule_df.columns[0]
    return full_schedule_df[full_schedule_df[first_col].str.startswith(f"{Ward_name}_Officer_")].copy()


if __name__ == "__main__":
    app.run(debug=True)
