
from pathlib import Path
from textwrap import dedent

import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from flask import send_from_directory, Response

###############################################################################
# 1 · Paths & basic checks
###############################################################################
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets";  ASSETS_DIR.mkdir(exist_ok=True)
PUBLIC_DIR = BASE_DIR / "public";  IMG_DIR = PUBLIC_DIR / "images"

CRIME_CSV = Path("data\combined_crime_2019-2025.csv")
LOOKUP_CSV = Path("data\LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv")
FEEDBACK_HTML = BASE_DIR / "feedback.html"

for f in (CRIME_CSV, LOOKUP_CSV, FEEDBACK_HTML):
    if not f.exists():
        raise FileNotFoundError(f"Required file missing: {f}")

###############################################################################
# 2 · (Optional) create a minimal style.css if none exists
###############################################################################
DEFAULT_CSS = ASSETS_DIR / "style.css"
if not DEFAULT_CSS.exists():
    DEFAULT_CSS.write_text(
        """
body{margin:0;font-family:sans-serif;background:#f4f6f9}
.site-header{display:flex;align-items:center;padding:10px 24px;background:#0d47a1;color:#fff}
.logo{height:48px;margin-right:12px}.site-title{font-size:1.4rem;font-weight:600;margin-right:auto}
nav a{color:#bbdefb;margin-left:18px;text-decoration:none}.nav-link.active{color:#fff;border-bottom:2px solid #fff}
.dashboard-container{padding:24px}.dashboard-grid{display:grid;gap:24px;grid-template-columns:1fr}
.panel{background:#fff;border-radius:6px;padding:16px;box-shadow:0 2px 6px rgba(0,0,0,.08)}
.graph-iframe{width:100%;height:450px;border:none}
""",
        encoding="utf-8",
    )

###############################################################################
# 3 · Load & prep data
###############################################################################
crime_cols = {"Month", "LSOA code", "Crime type"}
crime_df = pd.read_csv(CRIME_CSV, low_memory=False)
if not crime_cols.issubset(crime_df.columns):
    raise KeyError("Crime CSV missing columns: " + ", ".join(crime_cols - set(crime_df.columns)))

lookup_cols = {"LSOA21CD", "WD24CD", "WD24NM"}
lookup_df = pd.read_csv(LOOKUP_CSV, usecols=list(lookup_cols))

crime_df["Month"] = pd.to_datetime(crime_df["Month"], errors="coerce").dt.to_period("M")
crime_df = crime_df[crime_df["Crime type"].str.lower() == "burglary"].copy()

crime_df = crime_df.merge(
    lookup_df.rename(columns={"LSOA21CD": "LSOA code", "WD24CD": "Ward", "WD24NM": "WardName"}),
    on="LSOA code", how="left"
)

max_period   = crime_df["Month"].max()
end_period   = max_period - 3
start_period = end_period - 11
recent_df    = crime_df[(crime_df["Month"] >= start_period) & (crime_df["Month"] <= end_period)].copy()
if recent_df.empty:
    raise ValueError("No burglary data in the calculated 12-month window.")

ward_month_counts = (
    recent_df.groupby(["Month", "Ward", "WardName"], observed=True)
             .size().rename("count").reset_index()
)
london_mean_series = ward_month_counts.groupby("Month", observed=True)["count"].mean().sort_index()

Y_RANGE_LINE = [0, 25]
Y_RANGE_BAR  = [0, 30]

wards_available = (
    ward_month_counts[["Ward", "WardName"]].drop_duplicates()
    .dropna(subset=["Ward"]).sort_values("Ward")
)
ward_name_map = dict(zip(wards_available["Ward"], wards_available["WardName"]))
DEFAULT_WARD  = wards_available["Ward"].iloc[0]

###############################################################################
# 4 · Dash app – normal two-tab layout
###############################################################################
app = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder=str(ASSETS_DIR))
app.title  = "London Burglary Dashboard"
server = app.server   # for extra Flask routes

HEADER = html.Header([
    html.Img(src="/assets/metrobadge.png", className="logo"),
    html.Span("Metropolitan Police", className="site-title"),
    html.Nav([
        html.A("Give Feedback", href="/feedback", className="nav-link", id="nav-feedback"),
        html.A("View Data",    href="#dashboard", className="nav-link active", id="nav-data"),
    ])
], className="site-header")

app.layout = html.Div([
    HEADER,
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label="Burglary Dashboard", children=[
            html.Div([
                html.Label("Select Ward:", style={"fontWeight":"bold","marginRight":"6px"}),
                dcc.Dropdown(
                    id="ward-dropdown",
                    options=[{"label":f"{w} – {ward_name_map[w]}", "value":w} for w in wards_available["Ward"]],
                    value=DEFAULT_WARD, clearable=False, style={"width":"380px"},
                ),
                html.Div([
                    html.Div([
                        html.H2("Burglary Trends"), dcc.Graph(id="line-chart")
                    ], className="panel"),
                    html.Div([
                        html.H2("Ward LSOA Distribution"), dcc.Graph(id="bar-chart")
                    ], className="panel"),
                ], className="dashboard-grid"),
            ], className="dashboard-container"),
        ]),
        dcc.Tab(label="Feedback Survey", children=[
            html.Iframe(src="/feedback", style={"width":"100%","height":"94vh","border":"none"})
        ]),
    ])
])

###############################################################################
# 5 · Callbacks
###############################################################################
@app.callback(
    Output("line-chart","figure"),
    Output("bar-chart","figure"),
    Input("ward-dropdown","value"))
def update_charts(ward):
    ward_series = (
        ward_month_counts[ward_month_counts["Ward"] == ward]
        .set_index("Month")["count"]
        .reindex(london_mean_series.index, fill_value=0)
    )
    months = [m.strftime("%Y-%m") for m in london_mean_series.index.to_timestamp()]

    line_fig = go.Figure([
        go.Scatter(x=months, y=london_mean_series,
                   name="London mean per ward", mode="lines+markers", line=dict(dash="dash")),
        go.Scatter(x=months, y=ward_series,
                   name=f"{ward} total", mode="lines+markers"),
    ]).update_layout(
        yaxis=dict(title="Burglaries", range=Y_RANGE_LINE, fixedrange=True),
        xaxis=dict(title="Month", tickangle=-45, fixedrange=True),
        margin=dict(l=40,r=20,t=40,b=80), legend=dict(orientation="h", y=1.02, x=1)
    )

    lsoa_counts = (
        recent_df[recent_df["Ward"] == ward]
        .groupby("LSOA code", observed=True).size()
        .sort_values(ascending=False)
    )
    bar_fig = go.Figure([
        go.Bar(x=lsoa_counts.index, y=lsoa_counts.values, name="Burglaries")
    ]).update_layout(
        yaxis=dict(title="Burglaries", range=Y_RANGE_BAR, fixedrange=True),
        xaxis=dict(title="LSOA", tickangle=45, fixedrange=True),
        margin=dict(l=40,r=20,t=40,b=120)
    )
    return line_fig, bar_fig

###############################################################################
# 6 · Extra Flask routes  (for static pages & chart iframes)
###############################################################################
@server.route("/feedback")
def serve_feedback():               # full feedback page
    return send_from_directory(BASE_DIR, "feedback.html")

@server.route("/line-chart")
def serve_line_chart():             # minimal HTML with Plotly figure
    fig, _ = update_charts(DEFAULT_WARD)
    return Response(fig.to_html(full_html=False, include_plotlyjs="cdn"),
                    mimetype="text/html")

@server.route("/bar-chart")
def serve_bar_chart():
    _, fig = update_charts(DEFAULT_WARD)
    return Response(fig.to_html(full_html=False, include_plotlyjs="cdn"),
                    mimetype="text/html")

###############################################################################
# 7 · Entrypoint
###############################################################################
if __name__ == "__main__":
    app.run_server(debug=True)



"""
Dashboard App – London Burglary Dashboard & Feedback Survey
==========================================================

• Tab 1 – interactive dashboard (Drop-down ➜ line + bar charts)
• Tab 2 – original feedback.html

NEW  — 2025-06-15
-----------------
*  `/line-chart` and `/bar-chart` routes now accept a **query parameter
   ?ward=<WD24CD>**, so external pages can request a specific ward.
*  `/wards`  returns JSON list [{code,name}, …] so dashboard.html can build
   its own dropdown.
"""

from pathlib import Path
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from flask import (
    send_from_directory,
    Response,
    request,
    jsonify,
)

# ───────────────────────────── Paths ────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets";  ASSETS_DIR.mkdir(exist_ok=True)
DATA_DIR   = BASE_DIR / "data"
PUBLIC_DIR = BASE_DIR / "public";  IMG_DIR = PUBLIC_DIR / "images"

CRIME_CSV = Path("data\combined_crime_2019-2025.csv")
LOOKUP_CSV = Path("data\LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv")
FEEDBACK_HTML = BASE_DIR / "feedback.html"

for f in (CRIME_CSV, LOOKUP_CSV, FEEDBACK_HTML):
    if not f.exists():
        raise FileNotFoundError(f"Required file missing: {f}")

# ─────────────────────────── CSS (auto-create) ──────────────────────────────
DEFAULT_CSS = ASSETS_DIR / "style.css"
if not DEFAULT_CSS.exists():
    DEFAULT_CSS.write_text(
        """
body{margin:0;font-family:sans-serif;background:#f4f6f9}
.site-header{display:flex;align-items:center;padding:10px 24px;background:#0d47a1;color:#fff}
.logo{height:48px;margin-right:12px}.site-title{font-size:1.4rem;font-weight:600;margin-right:auto}
nav a{color:#bbdefb;margin-left:18px;text-decoration:none}.nav-link.active{color:#fff;border-bottom:2px solid #fff}
.dashboard-container{padding:24px}.dashboard-grid{display:grid;gap:24px;grid-template-columns:1fr}
.panel{background:#fff;border-radius:6px;padding:16px;box-shadow:0 2px 6px rgba(0,0,0,.08)}
.graph-iframe{width:100%;height:450px;border:none}.long-panel{grid-column:1/-1}
""",
        encoding="utf-8",
    )

# ─────────────────────── Load & prep data ───────────────────────────────────
crime_df = pd.read_csv(CRIME_CSV, low_memory=False)
crime_df["Month"] = pd.to_datetime(crime_df["Month"], errors="coerce").dt.to_period("M")
crime_df = crime_df[crime_df["Crime type"].str.lower() == "burglary"].copy()

lookup_df = pd.read_csv(
    LOOKUP_CSV, usecols=["LSOA21CD", "WD24CD", "WD24NM"]
).rename(columns={"LSOA21CD": "LSOA code", "WD24CD": "Ward", "WD24NM": "WardName"})

crime_df = crime_df.merge(lookup_df, on="LSOA code", how="left")

max_p       = crime_df["Month"].max()
end_p       = max_p - 3
start_p     = end_p - 11
recent_df   = crime_df[(crime_df["Month"] >= start_p) & (crime_df["Month"] <= end_p)].copy()

ward_month_counts = (
    recent_df.groupby(["Month", "Ward", "WardName"], observed=True)
             .size().rename("count").reset_index()
)
london_mean = ward_month_counts.groupby("Month")["count"].mean().sort_index()

wards_df = (
    ward_month_counts[["Ward", "WardName"]]
    .drop_duplicates()
    .dropna(subset=["Ward"])
    .sort_values("Ward")
)
WARD_NAME = dict(zip(wards_df["Ward"], wards_df["WardName"]))
DEFAULT_WARD = wards_df["Ward"].iloc[0]

# ───────────────────────── Dash app layout ─────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder=str(ASSETS_DIR))
app.title = "London Burglary Dashboard"
server = app.server  # Flask

HEADER = html.Header([
    html.Img(src="/assets/metrobadge.png", className="logo"),
    html.Span("Metropolitan Police", className="site-title"),
    html.Nav([
        html.A("Give Feedback", href="/feedback", className="nav-link", id="nav-feedback"),
        html.A("View Data",    href="#dashboard", className="nav-link active", id="nav-data"),
    ])
], className="site-header")

app.layout = html.Div([
    HEADER,
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label="Burglary Dashboard", children=[
            html.Div([
                html.Label("Select Ward:", style={"fontWeight":"bold","marginRight":"6px"}),
                dcc.Dropdown(
                    id="ward-dropdown",
                    options=[{"label":f"{c} – {WARD_NAME[c]}", "value":c} for c in wards_df["Ward"]],
                    value=DEFAULT_WARD, clearable=False, style={"width":"380px"},
                ),
                html.Div([
                    html.Div([html.H2("Burglary Trends"), dcc.Graph(id="line-chart")],
                             className="panel"),
                    html.Div([html.H2("Ward LSOA Distribution"), dcc.Graph(id="bar-chart")],
                             className="panel"),
                ], className="dashboard-grid"),
            ], className="dashboard-container")
        ]),
        dcc.Tab(label="Feedback Survey", children=[
            html.Iframe(src="/feedback", style={"width":"100%","height":"94vh","border":"none"})
        ])
    ])
])

# ───────────────────────── Chart builder ───────────────────────────────────
def build_figures(ward_code: str):
    months = [m.strftime("%Y-%m") for m in london_mean.index.to_timestamp()]

    ward_series = (
        ward_month_counts[ward_month_counts["Ward"] == ward_code]
        .set_index("Month")["count"]
        .reindex(london_mean.index, fill_value=0)
    )

    line_fig = go.Figure([
        go.Scatter(x=months, y=london_mean, name="London mean / ward",
                   mode="lines+markers", line=dict(dash="dash")),
        go.Scatter(x=months, y=ward_series, name=f"{ward_code} total",
                   mode="lines+markers"),
    ]).update_layout(
        yaxis=dict(range=[0,25], title="Burglaries", fixedrange=True),
        xaxis=dict(title="Month", tickangle=-45, fixedrange=True),
        margin=dict(l=40,r=20,t=40,b=80), legend=dict(orientation="h", y=1.02, x=1)
    )

    lsoa_counts = (
        recent_df[recent_df["Ward"] == ward_code]
        .groupby("LSOA code", observed=True).size()
        .sort_values(ascending=False)
    )

    bar_fig = go.Figure([
        go.Bar(x=lsoa_counts.index, y=lsoa_counts.values, name="Burglaries")
    ]).update_layout(
        yaxis=dict(range=[0,30], title="Burglaries", fixedrange=True),
        xaxis=dict(title="LSOA", tickangle=45, fixedrange=True),
        margin=dict(l=40,r=20,t=40,b=120)
    )

    return line_fig, bar_fig

# ───────────────────────── Dash callbacks ──────────────────────────────────
@app.callback(
    Output("line-chart", "figure"),
    Output("bar-chart",  "figure"),
    Input("ward-dropdown", "value"))
def update_charts(ward_code):
    return build_figures(ward_code)

# ───────────────────────── Flask helper routes ─────────────────────────────
@server.route("/feedback")
def serve_feedback():
    return send_from_directory(BASE_DIR, "feedback.html")

@server.route("/wards")
def serve_wards():
    # JSON list for ajax in dashboard.html
    data = [{"code": c, "name": WARD_NAME[c]} for c in wards_df["Ward"]]
    return jsonify(data)

@server.route("/line-chart")
def serve_line_chart():
    ward = request.args.get("ward", DEFAULT_WARD)
    fig, _ = build_figures(ward if ward in WARD_NAME else DEFAULT_WARD)
    return Response(fig.to_html(full_html=False, include_plotlyjs="cdn"),
                    mimetype="text/html")

@server.route("/bar-chart")
def serve_bar_chart():
    ward = request.args.get("ward", DEFAULT_WARD)
    _, fig = build_figures(ward if ward in WARD_NAME else DEFAULT_WARD)
    return Response(fig.to_html(full_html=False, include_plotlyjs="cdn"),
                    mimetype="text/html")

# ───────────────────────── Entrypoint ──────────────────────────────────────
if __name__ == "__main__":
    app.run_server(debug=True)