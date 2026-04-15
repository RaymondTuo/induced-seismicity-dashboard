from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Project Geminae Dashboard", page_icon=":earth_americas:", layout="wide")


CSV_CANDIDATES = [
    Path("dowhy_mediation_analysis.csv"),
    Path("results") / "dowhy_mediation_analysis.csv",
]

CHANGE7_BRIEF_HTML = Path("ProjectGeminae_Change7_Brief_20260405.html")

SPATIAL_LINK_TEMPLATE = "event_well_links_with_faults_{lookback}d_{radius}km.csv"


def field_zone(radius_km: float) -> str:
    if radius_km <= 5:
        return "Near-field (1-5km)"
    if radius_km <= 10:
        return "Mid-field (6-10km)"
    return "Far-field (11-20km)"


def build_fallback_data() -> pd.DataFrame:
    radius = np.arange(1, 21)
    rows = []
    for r in radius:
        zone_shift = 0.02 if r <= 5 else (0.01 if r <= 10 else 0.0)
        rows.append(
            {
                "radius_km": r,
                "model_type": "OLS",
                "r2": max(0.08, 0.30 - 0.006 * r),
                "mae": 0.68 - 0.006 * min(r, 10),
                "skill_score": 0.12 + 0.008 * min(r, 12),
                "spearman_corr": 0.22 + 0.01 * min(r, 10),
                "direct_effect": (2.0e-5 * np.exp(-r / 5)) - (5e-7 if r > 10 else 0.0),
                "indirect_effect": 8.0e-6 * np.exp(-r / 8) + 1.8e-6,
                "total_effect": (2.0e-5 * np.exp(-r / 5)) + (8.0e-6 * np.exp(-r / 8) + 1.8e-6),
                "mediation_pct": 100 * (8.0e-6 * np.exp(-r / 8) + 1.8e-6)
                / ((2.0e-5 * np.exp(-r / 5)) + (8.0e-6 * np.exp(-r / 8) + 1.8e-6)),
                "lookback_days": 90,
                "pressure_missing_pct": 4 + 0.15 * r,
                "log_transform_applied": True,
                "hurdle_occurrence_prob": min(0.95, 0.12 + 0.03 * np.exp(-r / 6) + 0.004 * r),
                "hurdle_conditional_magnitude": 1.6 + 0.2 * np.exp(-r / 10),
            }
        )
        rows.append(
            {
                "radius_km": r,
                "model_type": "NonParamDML",
                "r2": max(0.12, 0.36 - 0.004 * r + zone_shift),
                "mae": 0.63 - 0.006 * min(r, 10),
                "skill_score": 0.18 + 0.010 * min(r, 12),
                "spearman_corr": 0.30 + 0.012 * min(r, 10),
                "direct_effect": (2.3e-5 * np.exp(-r / 5.2)) - (3.5e-7 if r > 10 else 0.0),
                "indirect_effect": 9.5e-6 * np.exp(-r / 7.8) + 1.9e-6,
                "total_effect": (2.3e-5 * np.exp(-r / 5.2)) + (9.5e-6 * np.exp(-r / 7.8) + 1.9e-6),
                "mediation_pct": 100 * (9.5e-6 * np.exp(-r / 7.8) + 1.9e-6)
                / ((2.3e-5 * np.exp(-r / 5.2)) + (9.5e-6 * np.exp(-r / 7.8) + 1.9e-6)),
                "lookback_days": 90,
                "pressure_missing_pct": 4 + 0.15 * r,
                "log_transform_applied": True,
                "hurdle_occurrence_prob": min(0.96, 0.14 + 0.03 * np.exp(-r / 6) + 0.004 * r),
                "hurdle_conditional_magnitude": 1.65 + 0.22 * np.exp(-r / 10),
            }
        )
    return pd.DataFrame(rows)


def find_csv_path() -> Path | None:
    for path in CSV_CANDIDATES:
        if path.exists():
            return path
    return None


_MODEL_NAME_LABELS = {
    "ols_baseline": "OLS (baseline)",
    "gbm_nonparam": "DML-GBM",
    "rf_nonparam": "DML-RF",
    "lasso_nonparam": "DML-Lasso",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "radius": "radius_km",
        "model": "model_type",
        "R2": "r2",
        "MAE": "mae",
        "SkillScore": "skill_score",
        "Spearman": "spearman_corr",
        "direct": "direct_effect",
        "indirect": "indirect_effect",
        "total": "total_effect",
        "mediation_percent": "mediation_pct",
        "predictive_r2": "r2",
        "spearman_rho": "spearman_corr",
        "prop_mediated": "mediation_pct",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "total_effect" not in df.columns and {"direct_effect", "indirect_effect"}.issubset(df.columns):
        df["total_effect"] = df["direct_effect"] + df["indirect_effect"]
    if "mediation_pct" not in df.columns and {"indirect_effect", "total_effect"}.issubset(df.columns):
        nonzero = df["total_effect"].replace(0, np.nan)
        df["mediation_pct"] = 100 * df["indirect_effect"] / nonzero
    if "model_name" in df.columns:
        df["model_type"] = df["model_name"].map(lambda m: _MODEL_NAME_LABELS.get(str(m), str(m)))
    elif "model_type" not in df.columns:
        df["model_type"] = "NonParamDML"
    if "lookback_days" not in df.columns:
        df["lookback_days"] = 90
    if "log_transform_applied" not in df.columns:
        df["log_transform_applied"] = True
    if "pressure_missing_pct" not in df.columns:
        df["pressure_missing_pct"] = np.nan
    if "hurdle_occurrence_prob" not in df.columns:
        df["hurdle_occurrence_prob"] = np.nan
    if "hurdle_conditional_magnitude" not in df.columns:
        df["hurdle_conditional_magnitude"] = np.nan
    if "hurdle_stage1_auc" not in df.columns:
        df["hurdle_stage1_auc"] = np.nan
    if "hurdle_stage2_mae" not in df.columns:
        df["hurdle_stage2_mae"] = np.nan
    return df


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, str]:
    csv_path = find_csv_path()
    if csv_path is None:
        return normalize_columns(build_fallback_data()), "fallback"
    df = pd.read_csv(csv_path)
    return normalize_columns(df), str(csv_path)


@st.cache_data
def discover_spatial_index() -> tuple[list[int], list[int]]:
    lookbacks: set[int] = set()
    radii: set[int] = set()
    pat = re.compile(r"event_well_links_with_faults_(\d+)d_(\d+)km\.csv$")
    for p in Path(".").glob("event_well_links_with_faults_*d_*km.csv"):
        m = pat.match(p.name)
        if m:
            lookbacks.add(int(m.group(1)))
            radii.add(int(m.group(2)))
    return sorted(lookbacks), sorted(radii)


@st.cache_data
def load_spatial_links(lookback_days: int, radius_km: int, max_rows: int = 20000) -> tuple[pd.DataFrame | None, str]:
    path = Path(SPATIAL_LINK_TEMPLATE.format(lookback=lookback_days, radius=radius_km))
    used_lookback = lookback_days
    if not path.exists():
        available_lookbacks, _ = discover_spatial_index()
        if available_lookbacks:
            # Fallback to first available lookback for this radius.
            for lb in available_lookbacks:
                alt = Path(SPATIAL_LINK_TEMPLATE.format(lookback=lb, radius=radius_km))
                if alt.exists():
                    path = alt
                    used_lookback = lb
                    break
    if not path.exists():
        return None, f"Missing spatial file: `{path}`"
    links = pd.read_csv(path, low_memory=False)
    links = links.rename(columns={"API Number": "api_number"})
    if "EventID" not in links.columns or "api_number" not in links.columns:
        return None, "Spatial file missing required columns `EventID` and/or `API Number`."
    if len(links) > max_rows:
        links = links.sample(max_rows, random_state=42)
    note = ""
    if used_lookback != lookback_days:
        note = f" (requested {lookback_days}d; using {used_lookback}d)"
    return links, f"Spatial source: `{path}`{note} ({len(links):,} rows displayed)"


def load_spatial_links_by_radius(radius_km: int, max_rows: int = 20000) -> tuple[pd.DataFrame | None, str]:
    lookbacks, _ = discover_spatial_index()
    paths: list[tuple[int, Path]] = []
    for lb in lookbacks:
        path = Path(SPATIAL_LINK_TEMPLATE.format(lookback=lb, radius=radius_km))
        if path.exists():
            paths.append((lb, path))
    if not paths:
        return None, f"Missing spatial files for radius {radius_km} km."

    frames: list[pd.DataFrame] = []
    for lb, path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame = frame.rename(columns={"API Number": "api_number"})
        frame["source_lookback_days"] = lb
        frames.append(frame)
    links = pd.concat(frames, ignore_index=True)
    if "EventID" not in links.columns or "api_number" not in links.columns:
        return None, "Spatial file missing required columns `EventID` and/or `API Number`."

    # Keep one row per event-well pair when multiple lookback exports overlap.
    links = links.drop_duplicates(subset=["EventID", "api_number"], keep="first")
    if len(links) > max_rows:
        links = links.sample(max_rows, random_state=42)
    used_lookbacks = sorted({lb for lb, _ in paths})
    return links, f"Spatial sources: radius {radius_km} km across lookbacks {used_lookbacks} ({len(links):,} rows displayed)"


def _build_map_points(links: pd.DataFrame) -> pd.DataFrame:
    event_pts = links[["EventID", "api_number", "Latitude (WGS84)", "Longitude (WGS84)", "Local Magnitude"]].copy()
    event_pts = event_pts.rename(columns={"Latitude (WGS84)": "lat", "Longitude (WGS84)": "lon"})
    event_pts["point_type"] = "Event"

    well_pts = links[["EventID", "api_number", "Surface Latitude", "Surface Longitude", "Vol Prev N (BBLs)"]].copy()
    well_pts = well_pts.rename(
        columns={"Surface Latitude": "lat", "Surface Longitude": "lon", "Vol Prev N (BBLs)": "metric_value"}
    )
    well_pts["point_type"] = "Well"
    event_pts = event_pts.rename(columns={"Local Magnitude": "metric_value"})

    pts = pd.concat([event_pts, well_pts], ignore_index=True)
    pts = pts.dropna(subset=["lat", "lon"])
    pts["label"] = np.where(
        pts["point_type"].eq("Event"),
        "Event " + pts["EventID"].astype(str),
        "Well " + pts["api_number"].astype(str),
    )
    return pts


def main() -> None:
    st.title("Project Geminae - Dashboard Overview")
    st.caption("Causal and predictive monitoring across 1-20 km radii.")

    df, source = load_dashboard_data()
    if source == "fallback":
        st.warning("No `dowhy_mediation_analysis.csv` found. Showing fallback demo values until pipeline outputs are available.")
    else:
        st.success(f"Live data source: `{source}`")

    df["zone"] = df["radius_km"].astype(float).apply(field_zone)

    st.sidebar.header("Global Filters")
    radius_range = st.sidebar.slider("Radius range (km)", 1, 20, (1, 20))
    models = sorted(df["model_type"].dropna().unique().tolist())
    selected_models = st.sidebar.multiselect("Model types", models, default=models)

    view_df = df[
        (df["radius_km"] >= radius_range[0])
        & (df["radius_km"] <= radius_range[1])
        & (df["model_type"].isin(selected_models))
    ].copy()

    tabs = st.tabs(
        [
            "1) Model Performance",
            "2) Causal & Mediation",
            "3) Nonlinear & Heterogeneous",
            "4) Diagnostics & Transformations",
            "5) Temporal Sensitivity",
            "6) Hurdle Model",
            "7) Interactive Data Table",
            "8) Automated Updates",
            "9) Texas Basin Map",
        ]
    )

    with tabs[0]:
        st.subheader("Model Performance Overview")
        perf_cols = ["r2", "mae", "skill_score", "spearman_corr"]
        perf_long = view_df.melt(
            id_vars=["radius_km", "model_type"],
            value_vars=[c for c in perf_cols if c in view_df.columns],
            var_name="metric",
            value_name="value",
        )
        fig_perf = px.line(
            perf_long,
            x="radius_km",
            y="value",
            color="model_type",
            facet_row="metric",
            markers=True,
            height=900,
        )
        fig_perf.update_yaxes(matches=None)
        st.plotly_chart(fig_perf)

    with tabs[1]:
        st.subheader("Causal Effects & Mediation Analysis")
        causal_long = view_df.melt(
            id_vars=["radius_km", "model_type"],
            value_vars=["direct_effect", "indirect_effect", "total_effect"],
            var_name="effect_type",
            value_name="effect_value",
        )
        fig_causal = px.line(causal_long, x="radius_km", y="effect_value", color="effect_type", line_dash="model_type", markers=True)
        st.plotly_chart(fig_causal)
        if "mediation_pct" in view_df.columns:
            fig_med = px.bar(view_df, x="radius_km", y="mediation_pct", color="model_type", barmode="group")
            fig_med.add_hline(y=100, line_dash="dash", line_color="black")
            st.plotly_chart(fig_med)

    with tabs[2]:
        st.subheader("Nonlinear & Heterogeneous Effects")
        zone_summary = (
            view_df.groupby(["zone", "model_type"], as_index=False)
            .agg(total_effect_mean=("total_effect", "mean"), mediation_mean=("mediation_pct", "mean"))
            .sort_values(["zone", "model_type"])
        )
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.bar(zone_summary, x="zone", y="total_effect_mean", color="model_type", barmode="group"))
        with col2:
            st.plotly_chart(px.bar(zone_summary, x="zone", y="mediation_mean", color="model_type", barmode="group"))

    with tabs[3]:
        st.subheader("Model Diagnostics & Transformations")
        c1, c2 = st.columns(2)
        with c1:
            if "log_transform_applied" in view_df.columns:
                log_counts = view_df["log_transform_applied"].value_counts(dropna=False).rename_axis("log_transform").reset_index(name="count")
                st.plotly_chart(px.pie(log_counts, names="log_transform", values="count"))
        with c2:
            if view_df["pressure_missing_pct"].notna().any():
                st.plotly_chart(px.line(view_df, x="radius_km", y="pressure_missing_pct", color="model_type", markers=True))
            else:
                st.info("`pressure_missing_pct` not present in source file.")

    with tabs[4]:
        st.subheader("Temporal & Parameter Sensitivity Analysis")
        if df["lookback_days"].nunique() > 1:
            sens = (
                df[df["model_type"].isin(selected_models)]
                .groupby(["lookback_days", "model_type"], as_index=False)
                .agg(mean_total_effect=("total_effect", "mean"), mean_r2=("r2", "mean"))
            )
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.line(sens, x="lookback_days", y="mean_total_effect", color="model_type", markers=True))
            c2.plotly_chart(px.line(sens, x="lookback_days", y="mean_r2", color="model_type", markers=True))
        else:
            st.info("Only one lookback window found. Add multiple lookback outputs to enable sensitivity plots.")

    with tabs[5]:
        st.subheader("Occurrence vs Magnitude Modeling (Hurdle Model)")
        st.caption(
            "Stage 1 predicts P(earthquake occurs) via GBM binary classifier. "
            "Stage 2 predicts magnitude conditional on occurrence (event rows only). "
            "Note: formation depth (shallow vs. Ellenburger) is a known missing confounder "
            "for Stage 1 — unavailable without external well-completion records."
        )
        c1, c2 = st.columns(2)
        with c1:
            if view_df["hurdle_occurrence_prob"].notna().any():
                st.markdown("**Stage 1 — P(occurrence)**")
                st.plotly_chart(px.line(view_df, x="radius_km", y="hurdle_occurrence_prob", color="model_type", markers=True))
            else:
                st.info("`hurdle_occurrence_prob` not present in source file.")
        with c2:
            if view_df["hurdle_conditional_magnitude"].notna().any():
                st.markdown("**Stage 2 — E[magnitude | occurrence]**")
                st.plotly_chart(px.line(view_df, x="radius_km", y="hurdle_conditional_magnitude", color="model_type", markers=True))
            else:
                st.info("`hurdle_conditional_magnitude` not present in source file.")
        c3, c4 = st.columns(2)
        with c3:
            if view_df["hurdle_stage1_auc"].notna().any():
                st.markdown("**Stage 1 — ROC-AUC**")
                st.plotly_chart(px.line(view_df, x="radius_km", y="hurdle_stage1_auc", color="model_type", markers=True))
        with c4:
            if view_df["hurdle_stage2_mae"].notna().any():
                st.markdown("**Stage 2 — MAE (magnitude units)**")
                st.plotly_chart(px.line(view_df, x="radius_km", y="hurdle_stage2_mae", color="model_type", markers=True))

    with tabs[6]:
        st.subheader("Interactive Data Table")
        st.dataframe(view_df.sort_values(["radius_km", "model_type"]))
        st.download_button(
            "Download filtered results as CSV",
            data=view_df.to_csv(index=False),
            file_name="geminae_dashboard_filtered_results.csv",
            mime="text/csv",
        )

    with tabs[7]:
        st.subheader("Automated Updates & Integration")
        st.markdown(
            "- Dashboard reads from `dowhy_mediation_analysis.csv` automatically.\n"
            "- Refresh app after pipeline run to visualize latest outputs.\n"
            "- Optional: place additional event-level CSV in `results/` and extend loader mapping."
        )
        if CHANGE7_BRIEF_HTML.exists():
            with st.expander("Change 7 brief — two-stage hurdle model (Apr 2026)", expanded=False):
                st.caption(
                    "Executive summary shipped with the repo (`ProjectGeminae_Change7_Brief_20260405.html`). "
                    "Scroll inside the panel to read the full brief."
                )
                components.html(CHANGE7_BRIEF_HTML.read_text(encoding="utf-8"), height=720, scrolling=True)
        st.code(
            "python -m streamlit run dashboard_app.py --server.headless true",
            language="bash",
        )

    with tabs[8]:
        st.subheader("Texas Basin Map: Event-Well Link Explorer")
        _, spatial_radii = discover_spatial_index()
        map_radius_options = [r for r in spatial_radii if radius_range[0] <= r <= radius_range[1]] if spatial_radii else list(range(radius_range[0], radius_range[1] + 1))
        if not map_radius_options:
            map_radius_options = list(range(radius_range[0], radius_range[1] + 1))
        c1, c2 = st.columns(2)
        with c1:
            map_radius = st.selectbox("Map radius (km)", map_radius_options, index=0, key="map_radius")
        with c2:
            st.caption("Map includes all available lookback exports for this radius.")

        links_df, map_status = load_spatial_links_by_radius(int(map_radius))
        if links_df is None:
            st.info(map_status)
        else:
            st.caption(map_status)
            if "Local Magnitude" in links_df.columns and links_df["Local Magnitude"].notna().any():
                mag_values = links_df["Local Magnitude"].dropna().astype(float)
                mag_min = float(np.floor(mag_values.min() * 10) / 10)
                mag_max = float(np.ceil(mag_values.max() * 10) / 10)
                mag_range = st.slider(
                    "Event magnitude range",
                    min_value=mag_min,
                    max_value=mag_max,
                    value=(mag_min, mag_max),
                    step=0.1,
                )
                links_df = links_df[
                    links_df["Local Magnitude"].notna()
                    & (links_df["Local Magnitude"].astype(float) >= mag_range[0])
                    & (links_df["Local Magnitude"].astype(float) <= mag_range[1])
                ].copy()
            else:
                st.info("`Local Magnitude` not available in map source; magnitude filter disabled.")

            if links_df.empty:
                st.warning("No event-well links match the selected magnitude range.")
            else:
                points_df = _build_map_points(links_df)
                if "map_selected_event" not in st.session_state:
                    st.session_state["map_selected_event"] = None
                if "map_selected_well" not in st.session_state:
                    st.session_state["map_selected_well"] = None

                event_options = sorted(points_df["EventID"].dropna().astype(str).unique().tolist())
                well_options = sorted(points_df["api_number"].dropna().astype(str).unique().tolist())
                picked_event = st.selectbox(
                    "Selected event (click map or choose here)",
                    [""] + event_options,
                    index=0 if not st.session_state["map_selected_event"] else ([""] + event_options).index(str(st.session_state["map_selected_event"])) if str(st.session_state["map_selected_event"]) in event_options else 0,
                )
                picked_well = st.selectbox(
                    "Selected well (click map or choose here)",
                    [""] + well_options,
                    index=0 if not st.session_state["map_selected_well"] else ([""] + well_options).index(str(st.session_state["map_selected_well"])) if str(st.session_state["map_selected_well"]) in well_options else 0,
                )

                # Keep one active focus at a time: event -> linked wells, or well -> linked events.
                if picked_event:
                    st.session_state["map_selected_event"] = str(picked_event)
                    st.session_state["map_selected_well"] = None
                    picked_well = ""
                elif picked_well:
                    st.session_state["map_selected_well"] = str(picked_well)
                    st.session_state["map_selected_event"] = None
                    picked_event = ""
                else:
                    st.session_state["map_selected_event"] = None
                    st.session_state["map_selected_well"] = None

                if picked_event:
                    linked = links_df[links_df["EventID"].astype(str) == str(picked_event)].copy()
                    highlight_event_ids = {str(picked_event)}
                    highlight_well_ids = set(linked["api_number"].astype(str).tolist())
                elif picked_well:
                    linked = links_df[links_df["api_number"].astype(str) == str(picked_well)].copy()
                    highlight_event_ids = set(linked["EventID"].astype(str).tolist())
                    highlight_well_ids = {str(picked_well)}
                else:
                    linked = links_df.copy()
                    highlight_event_ids = set()
                    highlight_well_ids = set()

                points_df["highlight_role"] = points_df["point_type"]
                points_df.loc[
                    points_df["point_type"].eq("Event") & points_df["EventID"].astype(str).isin(highlight_event_ids),
                    "highlight_role",
                ] = "Selected Event"
                points_df.loc[
                    points_df["point_type"].eq("Well") & points_df["api_number"].astype(str).isin(highlight_well_ids),
                    "highlight_role",
                ] = "Selected Well"

                # Hide non-associated points once an event/well is selected.
                if picked_event or picked_well:
                    points_plot = points_df[
                        (
                            points_df["point_type"].eq("Event")
                            & points_df["EventID"].astype(str).isin(highlight_event_ids)
                        )
                        | (
                            points_df["point_type"].eq("Well")
                            & points_df["api_number"].astype(str).isin(highlight_well_ids)
                        )
                    ].copy()
                else:
                    points_plot = points_df.copy()

                # Scale event marker size by local magnitude; wells stay fixed size.
                points_plot["marker_size"] = 9.0
                event_mask = points_plot["point_type"].eq("Event")
                if event_mask.any():
                    event_mag = pd.to_numeric(points_plot.loc[event_mask, "metric_value"], errors="coerce")
                    if event_mag.notna().any():
                        mag_min = float(event_mag.min())
                        mag_max = float(event_mag.max())
                        if mag_max > mag_min:
                            norm = (event_mag - mag_min) / (mag_max - mag_min)
                            scaled = 10 + norm * 14  # 10..24
                        else:
                            scaled = pd.Series(16.0, index=event_mag.index)
                        points_plot.loc[event_mask, "marker_size"] = scaled.fillna(12.0)
                    else:
                        points_plot.loc[event_mask, "marker_size"] = 12.0

                fig_map = px.scatter_mapbox(
                    points_plot,
                    lat="lat",
                    lon="lon",
                    color="highlight_role",
                    size="marker_size",
                    size_max=24,
                    hover_name="label",
                    hover_data={"EventID": True, "api_number": True, "metric_value": ":.3f", "marker_size": False, "highlight_role": False},
                    custom_data=["point_type", "EventID", "api_number"],
                    color_discrete_map={
                        "Event": "#636EFA",
                        "Well": "#EF553B",
                        "Selected Event": "#00CC96",
                        "Selected Well": "#AB63FA",
                    },
                    zoom=5.2,
                    height=650,
                )
                fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select")
                selected = st.plotly_chart(
                    fig_map,
                    on_select="rerun",
                    selection_mode="points",
                    key="tx_basin_map",
                    use_container_width=True,
                )

                if selected and selected.get("selection", {}).get("points"):
                    first = selected["selection"]["points"][0].get("customdata", [])
                    if len(first) >= 3:
                        p_type, ev_id, api = first[0], first[1], first[2]
                        if p_type == "Event":
                            st.session_state["map_selected_event"] = str(ev_id)
                            st.session_state["map_selected_well"] = None
                        elif p_type == "Well":
                            st.session_state["map_selected_well"] = str(api)
                            st.session_state["map_selected_event"] = None

                st.write(f"Associated links found: **{len(linked):,}**")
                show_cols = [c for c in ["EventID", "api_number", "Local Magnitude", "Vol Prev N (BBLs)", "Distance from Well to Event", "Nearest Fault Dist (km)"] if c in linked.columns]
                st.dataframe(linked[show_cols].head(500))


if __name__ == "__main__":
    main()
