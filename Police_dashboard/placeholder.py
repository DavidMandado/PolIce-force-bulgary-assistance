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
    df = pd.read_csv(MASTER_CSV_PATH, parse_dates=["month"])
    y0, y1 = int(past_range[0]), int(past_range[1])
    df = df[(df.month.dt.year >= y0) & (df.month.dt.year <= y1)]
    # restrict to London LSOAs
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
        # Load your predictions CSV
        df_pred = pd.read_csv(PRED_CSV_PATH)
        # Expect columns: ['lsoa_code', 'predicted_count']

        if level == "ward":
            # Map LSOA → ward then aggregate
            df_pred["ward_code"] = df_pred.lsoa_code.map(lsoa_to_ward)
            wc = (
                df_pred.groupby("ward_code")["predicted_burglary"]
                       .sum().reset_index(name="count")
            )
            # Ensure all wards displayed
            all_w = [f["properties"]["GSS_Code"] for f in ward_geo["features"]]
            dfw = pd.DataFrame({"code": all_w}).merge(
                wc.rename(columns={"ward_code":"code", "predicted_burglary":"count"}), on="code", how="left"
            )
            dfw["count"] = dfw["count"].fillna(0).astype(int)

            # Choropleth of predicted counts by ward
            fig = px.choropleth_map(
                dfw, geojson=ward_geo, featureidkey="properties.GSS_Code",
                locations="code", color="count",
                color_continuous_scale="oryel", opacity=0.7,
                map_style="open-street-map",
                center={"lat":51.5074, "lon":-0.1278}, zoom=10,
                labels={"count":"Predicted Burglaries"},
            )
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            # Hide LSOA & table in ward overview
            return fig, fig, FULL_MAP_STYLE, {"display":"none"}, html.Div()

        # LSOA view (full London)
        df_ls = df_pred.rename(columns={"lsoa_code":"code", "predicted_burglary":"count"})
        fig = px.choropleth_map(
            df_ls, geojson=lsoa_geo, featureidkey="properties.LSOA11CD",
            locations="code", color="count",
            color_continuous_scale="oryel", opacity=0.7,
            map_style="open-street-map",
            center={"lat":51.5074, "lon":-0.1278}, zoom=10,
            labels={"count":"Predicted Burglaries"},
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
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
            df_pred[df_pred.lsoa_code.isin(lcodes)]
            .groupby("lsoa_code")["burglary_count"]
            .sum().reset_index(name="count")
        ).rename(columns={"lsoa_code":"code"})

        # centre on ward
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

    if mode == "past":
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

            ward_fig = px.choropleth_map(
                dfw, geojson=ward_geo, featureidkey="properties.GSS_Code",
                locations="code", color="count", opacity=0.7,
                color_continuous_scale="oryel",
                map_style="open-street-map",
                center={"lat":51.5074,"lon":-0.1278}, zoom=10,
                labels={"count":"Burglary Count"},
            )
            ward_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

            if not selected_ward:
                return (
                    ward_fig,   # left map
                    ward_fig,   # right map (hidden)
                    FULL_MAP_STYLE,
                    {"display":"none"},
                    html.Div()  # empty allocation
                )

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
                ward_fig,      # left top
                lsoa_fig,      # right top
                HALF_MAP_STYLE,# left graph style
                HALF_MAP_STYLE,# right graph style
                html.Div()
            )

    # ── 3) Full-London LSOA view ─────────────────────────────────────────
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

    # hide ward map & allocation if you’re in LSOA‐only view
    return (
        lsoaf,                # map-ward (hidden)
        lsoaf,                # map-lsoa
        {"display":"none"},   # ward style
        FULL_MAP_STYLE,       # lsoa style
        html.Div()            # empty alloc container
    )