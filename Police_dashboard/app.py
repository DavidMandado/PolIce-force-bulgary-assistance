import os
import json

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import os, pandas as pd
import geopandas as gpd
import plotly.express as px
from shapely.geometry import shape

from dash import dash_table

# ─── Paths ────────────────────────────────────────────────────────────────────

# Centralized data folder
DATA_DIR = r"C:/Users/20232726/Desktop/me/PolIce-force-bulgary-assistance/data"

# The “master” CSV with all LSOA × month burglary counts and features
MASTER_CSV_PATH  = os.path.join(DATA_DIR, "crime_fixed_data.csv")

# GeoJSON files (make sure these exist inside DATA_DIR)
WARD_GEOJSON     = os.path.join(DATA_DIR, "wards.geojson")
LSOA_GEOJSON     = os.path.join(DATA_DIR, "LSOAs.geojson")

# (Leave these for your predicted/allocation modes later)
PRED_CSV_PATH    = os.path.join(DATA_DIR, "burglary_next_month_forecast.csv")
ALLOC_CSV_PATH   = os.path.join(DATA_DIR, "abbey_ward_patrol_schedule.csv")


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
app = dash.Dash(__name__)

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
                        value=[2021, 2022]
                    ),
                ]
            ),

            # ── Predicted Controls (only when “Predicted Data”) ─────────────────
            html.Div(
                id="pred-controls",
                children=[
                    html.Br(),
                    html.Button(
                        "Predict Next Month",
                        id="predict-button",
                        n_clicks=0,
                        style={"width": "100%"}
                    ),
                    html.Br(), html.Br(),
                    html.Label("Upload New Monthly CSV:"),
                    dcc.Upload(
                        id="upload-file",
                        children=html.Div(["Drag & Drop or ", html.A("Select CSV")]),
                        style={
                            "width": "100%", "height": "40px",
                            "borderWidth": "1px", "borderStyle": "dashed",
                            "borderRadius": "5px", "textAlign": "center"
                        }
                    ),
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
    elif mode == "alloc":
        return {"display": "none"}, {"display": "none"}, {}
    return {}, {"display": "none"}, {"display": "none"}


# 5.2) Handle upload of a new monthly CSV (just append raw rows)
@app.callback(
    Output("upload-file", "children"),
    Input("upload-file", "contents"),
    State("upload-file", "filename"),
)
def handle_upload(contents, filename):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df_new = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception:
        return html.Div("Error: could not read uploaded CSV.")

    if os.path.exists(MASTER_CSV_PATH):
        df_master = pd.read_csv(MASTER_CSV_PATH)
        df_combined = pd.concat([df_master, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(MASTER_CSV_PATH, index=False)
    return html.Div(f"Uploaded “{filename}” → master.csv updated.")


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
    Input("map-ward",        "clickData"),
    Input("back-button",     "n_clicks"),
    Input("ward-search-button", "n_clicks"),
    State("ward-search-input", "value"),
)
def handle_selection(map_click, back_click, search_click, search_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]["prop_id"].split(".")[0]

    # 1) user clicked on the map
    if trig == "map-ward" and map_click:
        code = map_click["points"][0]["location"]
        return code, {"display": "block", "width": "100%"}

    # 2) user clicked the Back button
    if trig == "back-button":
        return None, {"display": "none"}

    # 3) user clicked the Go button in the search box
    if trig == "ward-search-button" and search_value:
        q = search_value.strip()
        # try code first
        if q.upper() in ward_mapping:
            out_code = q.upper()
        else:
            # then name lookup (case‐insensitive)
            out_code = name_to_code.get(q.lower())
        if not out_code:
            raise PreventUpdate
        return out_code, {"display": "block", "width": "100%"}

    # anything else → do nothing
    raise PreventUpdate

@app.callback(
    Output("map-ward", "figure"),
    Output("map-lsoa", "figure"),
    Output("map-ward", "style"),
    Output("map-lsoa", "style"),
    Output("allocation-table-container", "children"),
    Input("apply-button",    "n_clicks"),
    Input("selected-ward",   "data"),
    State("level",           "value"),
    State("data-mode",       "value"),
    State("past-range",      "value"),
)
def update_maps(n_clicks, selected_ward, level, mode, past_range):
    # ── 1) Load & filter master CSV ──────────────────────────────────────
    df = pd.read_csv(MASTER_CSV_PATH, parse_dates=["year_month"])
    y0, y1 = int(past_range[0]), int(past_range[1])
    df = df[(df.year_month.dt.year >= y0) & (df.year_month.dt.year <= y1)]
    # restrict to London LSOAs
    df = df[df.lsoa_code.isin(lsoa_to_ward)]
    if df.empty:
        blank = px.choropleth_mapbox(
            pd.DataFrame({"code":[], "count":[]}),
            geojson=ward_geo, featureidkey="properties.GSS_Code",
            locations="code", color="count",
            mapbox_style="open-street-map",
            center={"lat":51.5074,"lon":-0.1278}, zoom=10
        )
        blank.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return blank, blank, FULL_MAP_STYLE, {"display":"none"}, html.Div()

    # ── 2) Ward‐level view ────────────────────────────────────────────────
    if level == "ward":
        # aggregate monthly counts up to wards
        df["ward_code"] = df.lsoa_code.map(lsoa_to_ward)
        wc = (
            df.groupby("ward_code")["burglary_count"]
              .sum().reset_index(name="count")
        )
        # ensure every ward shows up
        all_w = [f["properties"]["GSS_Code"] for f in ward_geo["features"]]
        dfw = pd.DataFrame({"code": all_w}).merge(
            wc.rename(columns={"ward_code":"code"}), on="code", how="left"
        )
        dfw["count"] = dfw["count"].fillna(0).astype(int)

        ward_fig = px.choropleth_mapbox(
            dfw, geojson=ward_geo, featureidkey="properties.GSS_Code",
            locations="code", color="count", opacity=0.7,
            color_continuous_scale="oryel",
            mapbox_style="open-street-map",
            center={"lat":51.5074,"lon":-0.1278}, zoom=10,
            labels={"count":"Burglary Count"},
        )
        ward_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        # ── no ward drilled down → full-width ward map, hide LSOA & alloc
        if not selected_ward:
            return ward_fig, ward_fig, FULL_MAP_STYLE, {"display":"none"}, html.Div()

        # ── 2a) Ward drilled down → LSOA map + allocation table ─────────
        # get that ward geometry
        ward_geom = next(
            f["geometry"] for f in ward_geo["features"]
            if f["properties"]["GSS_Code"] == selected_ward
        )
        # pick only those LSOAs whose centroid falls inside
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

        # centre on ward
        minx, miny, maxx, maxy = shape(ward_geom).bounds
        center_l = {"lat":(miny+maxy)/2, "lon":(minx+maxx)/2}

        lsoa_fig = px.choropleth_mapbox(
            fl, geojson=geo_l, featureidkey="properties.LSOA11CD",
            locations="code", color="count", opacity=0.7,
            color_continuous_scale="oryel",
            mapbox_style="open-street-map",
            center=center_l, zoom=12,
            labels={"count":"Burglary Count"},
        )
        lsoa_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        # load the allocation CSV for this ward
        alloc_path = os.path.join(DATA_DIR, "allocations", f"{selected_ward}.csv")
        if os.path.exists(alloc_path):
            df_alloc = pd.read_csv(alloc_path)
            alloc_table = dash_table.DataTable(
                data=df_alloc.to_dict("records"),
                columns=[{"name":c,"id":c} for c in df_alloc.columns],
                style_table={"overflowX":"auto"},
                style_cell={"padding":"4px","textAlign":"left"}
            )
        else:
            alloc_table = html.Div(f"No allocation file for {selected_ward}", style={"color":"red"})

        return (
            ward_fig,      # left top
            lsoa_fig,      # right top
            HALF_MAP_STYLE,# left graph style
            HALF_MAP_STYLE,# right graph style
            alloc_table    # bottom
        )

    # ── 3) Full-London LSOA view ─────────────────────────────────────────
    df_ls = (
        df.groupby("lsoa_code")["burglary_count"]
          .sum().reset_index(name="count")
          .rename(columns={"lsoa_code":"code"})
    )
    lsoaf = px.choropleth_mapbox(
        df_ls, geojson=lsoa_geo, featureidkey="properties.LSOA11CD",
        locations="code", color="count", opacity=0.7,
        color_continuous_scale="oryel",
        mapbox_style="open-street-map",
        center={"lat":51.5074,"lon":-0.1278}, zoom=10,
        labels={"count":"Burglary Count"},
    )
    lsoaf.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # hide ward map & allocation if you’re in LSOA‐only view
    return (
        lsoaf,                # map-ward (hidden)
        lsoaf,                # map-lsoa
        {"display":"none"},   # ward style
        FULL_MAP_STYLE,       # lsoa style
        html.Div()            # empty alloc container
    )


if __name__ == "__main__":
    app.run(debug=True)
