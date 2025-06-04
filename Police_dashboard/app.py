import os
import json

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
from shapely.geometry import shape

# ─── Paths ────────────────────────────────────────────────────────────────────
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(APP_DIR)
DATA_DIR     = os.path.join(PROJECT_DIR, "police-dashboard", "data")
WARD_GEOJSON = "C:/Users/20232726/Desktop/me/ward_choro.json"
LSOA_GEOJSON = "C:/Users/20232726/Desktop/me/lsoa_choro.json"

# ─── Load GeoJSONs once ───────────────────────────────────────────────────────
with open(WARD_GEOJSON) as f:
    ward_geo = json.load(f)
with open(LSOA_GEOJSON) as f:
    lsoa_geo = json.load(f)

# ─── Build DataFrames ─────────────────────────────────────────────────────────
ward_df = pd.DataFrame([
    {"code": feat["properties"]["code"], "count": feat["properties"]["count"]}
    for feat in ward_geo["features"]
])

lsoa_df = pd.DataFrame([
    {"code": feat["properties"]["code"], "count": feat["properties"]["count"],
     "geometry": shape(feat["geometry"])}
    for feat in lsoa_geo["features"]
])

# ─── Dash App Setup ──────────────────────────────────────────────────────────
app = dash.Dash(__name__)

# CSS styles
SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "300px", "padding": "20px", "background-color": "#f8f9fa",
    "overflow": "auto", "transition": "transform 0.3s ease"
}
SIDEBAR_HIDDEN = {**SIDEBAR_STYLE, "transform": "translateX(-100%)"}
CONTENT_STYLE = {"margin-left": "320px", "margin-right": "20px", "padding": "20px"}
MAP_CONTAINER = {"display": "flex", "flexDirection": "row"}
HALF_MAP_STYLE = {"width": "50%", "height": "80vh"}
FULL_MAP_STYLE = {"width": "100%", "height": "80vh"}

app.layout = html.Div([
    dcc.Store(id="selected-ward", data=None),
    dcc.Store(id="sidebar-open", data=True),

    # Toggle filters
    html.Button("☰ Filters", id="btn-toggle", n_clicks=0,
                style={"position": "fixed", "top": "10px", "left": "10px", "zIndex": 1000}),

    # Sidebar
    html.Div(id="sidebar", children=[
        html.H2("Filters", style={"margin-top": "20px"}), html.Hr(),
        html.Label("Data View"),
        dcc.RadioItems(id="data-mode",
                       options=[{"label": "Past Data", "value": "past"},
                                {"label": "Predicted Data", "value": "pred"}],
                       value="past", labelStyle={"display": "block"}),
        html.Br(),
        html.Label("View Level"),
        dcc.Dropdown(id="level",
                     options=[{"label": "Ward", "value": "ward"},
                              {"label": "LSOA", "value": "lsoa"}],
                     value="ward", clearable=False),
        html.Br(),
        html.Div(id="past-controls", children=[
            html.Label("Date Range (Year-Month)"),
            dcc.RangeSlider(id="past-range", min=2018, max=2025, step=1/12,
                            marks={y: str(y) for y in range(2018, 2026)}, value=[2019, 2022])
        ]),
        html.Div(id="pred-controls", children=[
            html.Label("Prediction Horizon"),
            dcc.Dropdown(id="pred-horizon",
                         options=[{"label": "1 month", "value": 1},
                                  {"label": "2 months", "value": 2},
                                  {"label": "3 months", "value": 3}],
                         value=1, clearable=False),
            html.Br(),
            html.Label("Prediction Data Source"),
            dcc.RadioItems(id="pred-data-choice",
                           options=[{"label": "Last Prediction", "value": "last"},
                                    {"label": "Upload New", "value": "upload"}],
                           value="last", labelStyle={"display": "block"}),
            dcc.Upload(id="upload-pred-data", children=html.Div(["Drag & Drop or ", html.A("Select")]),
                       style={"width": "100%", "height": "40px", "borderWidth": "1px",
                              "borderStyle": "dashed", "borderRadius": "5px", "textAlign": "center"})
        ], style={"display": "none"}),
        html.Hr(),
        html.Button("← Back to Wards", id="back-button", n_clicks=0,
                    style={"display": "none", "width": "100%"}),
        html.Br(), html.Br(),
        html.Button("Apply", id="apply-button", n_clicks=0, style={"width": "100%"})
    ], style=SIDEBAR_STYLE),

    # Main content: maps
    html.Div(id="page-content", style=CONTENT_STYLE, children=[
        html.Div(id="map-container", style=MAP_CONTAINER, children=[
            dcc.Graph(id="map-ward", style=FULL_MAP_STYLE),
            dcc.Graph(id="map-lsoa", style={**HALF_MAP_STYLE, "display": "none"})
        ])
    ])
])

# ─── Callbacks ───────────────────────────────────────────────────────────────
@app.callback(
    Output("sidebar", "style"), Output("page-content", "style"),
    Input("btn-toggle", "n_clicks"), State("sidebar-open", "data")
)
def toggle_sidebar(n, open_):
    if n:
        return (SIDEBAR_HIDDEN, {**CONTENT_STYLE, "margin-left": "20px"}) if open_ else (SIDEBAR_STYLE, CONTENT_STYLE)
    return SIDEBAR_STYLE, CONTENT_STYLE

@app.callback(Output("sidebar-open", "data"), Input("sidebar", "style"))
def store_sidebar_state(style):
    return style.get("transform") != "translateX(-100%)"

@app.callback(Output("past-controls", "style"), Output("pred-controls", "style"),
              Input("data-mode", "value"))
def toggle_mode(mode):
    return ({}, {"display": "none"}) if mode == "past" else ({"display": "none"}, {})

@app.callback(Output("selected-ward", "data"), Output("back-button", "style"),
              Input("map-ward", "clickData"), Input("back-button", "n_clicks"))
def handle_selection(clickData, n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    if trig == "map-ward" and clickData:
        return clickData["points"][0]["location"], {"display": "block", "width": "100%"}
    if trig == "back-button":
        return None, {"display": "none"}
    raise PreventUpdate

@app.callback(
    Output("map-ward", "figure"),
    Output("map-lsoa", "figure"),
    Output("map-ward", "style"),
    Output("map-lsoa", "style"),
    Input("apply-button", "n_clicks"),
    Input("selected-ward", "data"),
    State("level", "value"),
    State("data-mode", "value"),
    State("past-range", "value"),
    State("pred-horizon", "value"),
    State("pred-data-choice", "value")
)
def update_maps(n_clicks, selected_ward, level, mode, past_range, pred_horizon, pred_source):
    # Base ward map
    df_w, geo_w = ward_df, ward_geo
    center_w, zoom_w = {"lat": 51.5074, "lon": -0.1278}, 10
    ward_fig = px.choropleth_map(
        df_w.astype({"code": str, "count": int}), geojson=geo_w,
        featureidkey="properties.code", locations="code", color="count",
        color_continuous_scale="oryel", map_style="open-street-map",
        center=center_w, zoom=zoom_w, opacity=0.7,
        labels={"count": "Burglary Count"}
    )
    ward_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # If a ward is selected, render LSOA map side-by-side
    if selected_ward:
        ward_feat = next((f for f in ward_geo["features"] if f["properties"]["code"] == selected_ward), None)
        if ward_feat:
            ward_poly = shape(ward_feat["geometry"])
            feats = [f for f in lsoa_geo["features"] if ward_poly.contains(shape(f["geometry"]).centroid)]
            geo_l = {"type": "FeatureCollection", "features": feats}
            codes = [f["properties"]["code"] for f in feats]
            df_l = lsoa_df[lsoa_df["code"].isin(codes)].drop(columns="geometry")
            center_l = {"lat": (ward_poly.bounds[1] + ward_poly.bounds[3]) / 2,
                        "lon": (ward_poly.bounds[0] + ward_poly.bounds[2]) / 2}
            zoom_l = 12
            lsoa_fig = px.choropleth_map(
                df_l.astype({"code": str, "count": int}), geojson=geo_l,
                featureidkey="properties.code", locations="code", color="count",
                color_continuous_scale="oryel", map_style="open-street-map",
                center=center_l, zoom=zoom_l, opacity=0.7,
                labels={"count": "Burglary Count"}
            )
            lsoa_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            return ward_fig, lsoa_fig, HALF_MAP_STYLE, HALF_MAP_STYLE
    # No ward selected: full-width ward map, hide LSOA
    empty_fig = ward_fig
    return ward_fig, empty_fig, FULL_MAP_STYLE, {**HALF_MAP_STYLE, "display": "none"}

if __name__ == "__main__":
    app.run(debug=True)

# by David Mandado May-June 2025
# Notes:
# - 