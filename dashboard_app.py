from datetime import datetime
from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Project Geminae — Induced Seismicity Dashboard",
    page_icon=":earth_americas:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Project Geminae — Induced Seismicity Dashboard**\n\n"
            "Causal and predictive monitoring of injection-induced seismicity in the Midland Basin. "
            "Data sources: TexNet catalog, SWD injection records, Horne et al. (2023) fault map."
        ),
    },
)


# ---------------------------------------------------------------------------
# Visual theme — one place to tune the global look.
# ---------------------------------------------------------------------------

THEME_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
CHART_HEIGHT = 420

# Plotly template used everywhere via `style_fig()`.
_TEMPLATE = go.layout.Template()
_TEMPLATE.layout.update(
    font=dict(family=THEME_FONT, size=13, color="#1f2933"),
    title=dict(font=dict(family=THEME_FONT, size=16, color="#102a43")),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    colorway=[
        "#0d47a1", "#e65100", "#2e7d32", "#6a1b9a",
        "#00838f", "#b71c1c", "#455a64", "#5d4037",
    ],
    margin=dict(l=60, r=20, t=48, b=48),
    hoverlabel=dict(
        bgcolor="#102a43",
        font=dict(color="#ffffff", family=THEME_FONT, size=12),
        bordercolor="#102a43",
    ),
    xaxis=dict(
        showgrid=True, gridcolor="#eef2f7", zeroline=False,
        ticks="outside", tickcolor="#cbd2d9", ticklen=5,
        linecolor="#cbd2d9", linewidth=1, title_standoff=12,
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#eef2f7", zeroline=False,
        ticks="outside", tickcolor="#cbd2d9", ticklen=5,
        linecolor="#cbd2d9", linewidth=1, title_standoff=12,
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cbd2d9", borderwidth=0,
        font=dict(size=12),
    ),
)
pio.templates["geminae"] = _TEMPLATE
pio.templates.default = "plotly_white+geminae"


def style_fig(fig: go.Figure, *, height: int | None = None, tight: bool = False) -> go.Figure:
    """Apply dashboard-wide visual polish to any Plotly figure."""
    fig.update_layout(
        template="plotly_white+geminae",
        height=height or CHART_HEIGHT,
        hovermode="x unified",
    )
    if tight:
        fig.update_layout(margin=dict(l=50, r=15, t=30, b=40))
    return fig


# Distinct, colorblind-friendly model colors (extend if new model_type values appear)
_MODEL_COLOR_MAP = {
    "OLS": "#0d47a1",
    "NonParamDML": "#e65100",
    "DML-GBM": "#2e7d32",
    "DML-RF": "#6a1b9a",
    "DML-Lasso": "#00838f",
    "OLS (baseline)": "#0d47a1",
}

# Semantic colors used in the Hurdle & Causal tabs.
COLOR_PROB = "#1e88e5"        # earthquake probability / occurrence
COLOR_MAG = "#c62828"         # earthquake magnitude / severity
COLOR_NEUTRAL = "#455a64"     # baseline / neutral reference

_METRIC_LABELS = {
    "r2": "R² (predictive)",
    "mae": "MAE",
    "skill_score": "Skill score",
    "spearman_corr": "Spearman ρ",
}

_EFFECT_LABELS = {
    "direct_effect": "Direct effect (W → S)",
    "indirect_effect": "Indirect (W → P → S)",
    "total_effect": "Total effect",
}

_EFFECT_COLOR_MAP = {
    "Direct effect (W → S)": "#0d47a1",
    "Indirect (W → P → S)": "#e65100",
    "Total effect": "#2e7d32",
}


def _model_color_map(model_types: list[str]) -> dict[str, str]:
    out = {}
    palette = px.colors.qualitative.Dark2 + px.colors.qualitative.Set2
    for i, m in enumerate(sorted(set(model_types))):
        out[m] = _MODEL_COLOR_MAP.get(m, palette[i % len(palette)])
    return out


def _apply_radius_xaxis(fig) -> None:
    fig.update_xaxes(title_text="Radius (km)")


def _strip_facet_equals(fig) -> None:
    """Turn 'metric_label=R²' style facet titles into short labels."""

    def _fix(a):
        if "=" in (a.text or ""):
            a.update(text=a.text.split("=", 1)[-1].strip())

    fig.for_each_annotation(_fix)


def _flatten_plotly_customdata(raw) -> list:
    """Plotly/Streamlit may supply customdata as a nested list."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        first = raw[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return [x for x in first]
        return [x for x in raw]
    return [raw]


def _parse_map_selection_point(point: dict) -> tuple[str | None, str | None, str | None]:
    """From a Plotly selection point dict, return (point_type, EventID, api_number)."""
    cd = point.get("customdata")
    flat = _flatten_plotly_customdata(cd)
    if len(flat) >= 3:
        return str(flat[0]), str(flat[1]), str(flat[2])
    return None, None, None


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


# Compact x labels for zone bar charts (ranges stay in the tab caption)
_ZONE_BAR_LABELS = {
    "Near-field (1-5km)": "Near-field",
    "Mid-field (6-10km)": "Mid-field",
    "Far-field (11-20km)": "Far-field",
}
_ZONE_BAR_ORDER = ["Near-field", "Mid-field", "Far-field"]

# Project standard for injection history in the Temporal tab (days before each event)
STANDARD_LOOKBACK_DAYS = 90


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
def load_dashboard_data() -> tuple[pd.DataFrame, str, str | None]:
    """Return (dataframe, source_label, modified_iso).

    source_label is the path string (or "fallback") and modified_iso is the
    CSV's last-modified timestamp in ISO format (None for fallback).
    """
    csv_path = find_csv_path()
    if csv_path is None:
        return normalize_columns(build_fallback_data()), "fallback", None
    df = pd.read_csv(csv_path)
    mtime = datetime.fromtimestamp(csv_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return normalize_columns(df), str(csv_path), mtime


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


def _inject_css() -> None:
    """Custom CSS for a denser, more professional look."""
    st.markdown(
        """
        <style>
          /* Tighten default Streamlit padding */
          .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }

          /* Hero banner */
          .geminae-hero {
            background: linear-gradient(135deg, #102a43 0%, #1e3a5f 60%, #2d5282 100%);
            color: #ffffff;
            padding: 1.4rem 1.8rem;
            border-radius: 10px;
            margin-bottom: 1.2rem;
            box-shadow: 0 2px 8px rgba(16,42,67,0.12);
          }
          .geminae-hero h1 {
            color: #ffffff !important;
            margin: 0 0 0.25rem 0;
            font-size: 1.65rem;
            font-weight: 600;
            letter-spacing: 0.2px;
          }
          .geminae-hero .subtitle {
            color: #d6e4f1;
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
          }
          .geminae-hero .meta {
            display: flex; gap: 1.5rem; flex-wrap: wrap;
            font-size: 0.82rem; color: #bcd3e6;
          }
          .geminae-hero .meta strong { color: #ffffff; font-weight: 600; }
          .geminae-hero .pill {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 500;
          }

          /* Section dividers inside tabs */
          .geminae-section {
            font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
            text-transform: uppercase; color: #627d98;
            margin: 1.3rem 0 0.4rem 0;
          }

          /* Polished metric cards (st.metric) */
          div[data-testid="stMetric"] {
            background: #f7fafc;
            border: 1px solid #e4e9f0;
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
            box-shadow: 0 1px 2px rgba(16,42,67,0.03);
          }
          div[data-testid="stMetricLabel"] p {
            font-size: 0.78rem; color: #486581; font-weight: 500;
          }
          div[data-testid="stMetricValue"] {
            color: #102a43; font-weight: 600;
          }

          /* Tabs — make them quieter and more elegant */
          button[data-baseweb="tab"] {
            font-size: 0.92rem; font-weight: 500;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(source: str, modified_iso: str | None, n_rows: int, n_models: int, radii: tuple[int, int]) -> None:
    """Top banner with project title and live data provenance."""
    if source == "fallback":
        status_pill = '<span class="pill" style="background:#b45309;">Demo data · pipeline outputs not found</span>'
        meta_source = "Fallback demo values"
    else:
        status_pill = '<span class="pill" style="background:#0f7a4d;">Live data</span>'
        meta_source = f"<code>{source}</code>"

    modified_str = modified_iso or "—"
    st.markdown(
        f"""
        <div class="geminae-hero">
          <h1>Project Geminae · Induced Seismicity Dashboard</h1>
          <div class="subtitle">
            Causal and predictive monitoring of injection-induced seismicity in the Midland Basin —
            well-level attribution across 1–20 km radii with a two-stage hurdle forecast.
          </div>
          <div class="meta">
            <div>{status_pill}</div>
            <div><strong>Source:</strong> {meta_source}</div>
            <div><strong>Last refreshed:</strong> {modified_str}</div>
            <div><strong>Rows:</strong> {n_rows:,}</div>
            <div><strong>Models:</strong> {n_models}</div>
            <div><strong>Radii:</strong> {radii[0]}–{radii[1]} km</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section(label: str) -> None:
    st.markdown(f'<div class="geminae-section">{label}</div>', unsafe_allow_html=True)


def main() -> None:
    _inject_css()

    df, source, modified_iso = load_dashboard_data()
    df["zone"] = df["radius_km"].astype(float).apply(field_zone)

    n_models = int(df["model_type"].dropna().nunique())
    n_rows_total = int(len(df))
    radii_bounds = (int(df["radius_km"].min()), int(df["radius_km"].max()))
    _render_hero(source, modified_iso, n_rows_total, n_models, radii_bounds)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### Data")
        refresh_col, stamp_col = st.columns([1, 1])
        with refresh_col:
            if st.button(
                "Refresh data",
                help=(
                    "Clear all cached CSVs and reload from disk. Use this after "
                    "the pipeline writes new outputs so every tab picks up the "
                    "latest data."
                ),
                use_container_width=True,
            ):
                st.cache_data.clear()
                st.session_state["_last_refresh"] = datetime.now().strftime("%H:%M:%S")
                st.toast("Cache cleared. Reloading data…", icon="✅")
                st.rerun()
        with stamp_col:
            last = st.session_state.get("_last_refresh")
            if last:
                st.caption(f"Refreshed at {last}")
            elif modified_iso:
                try:
                    mtime = datetime.fromisoformat(modified_iso).strftime("%H:%M:%S")
                    st.caption(f"File mtime {mtime}")
                except ValueError:
                    st.caption("File mtime n/a")
            else:
                st.caption("No cache activity yet")

        st.markdown("---")
        st.markdown("### Global filters")
        radius_range = st.slider("Radius range (km)", 1, 20, (1, 20))
        models = sorted(df["model_type"].dropna().unique().tolist())
        selected_models = st.multiselect("Model types", models, default=models)

        st.markdown("---")
        st.markdown("### About")
        with st.expander("What's in this dashboard?", expanded=False):
            st.markdown(
                "- **Performance** — predictive accuracy (R², MAE, skill, Spearman ρ) by radius.\n"
                "- **Causal effects** — DoWhy mediation: direct + indirect = total effect of injection on seismicity.\n"
                "- **Zone breakdown** — near / mid / far-field averages.\n"
                "- **90-day lookback** — project-standard injection window.\n"
                "- **Hurdle forecast** — P(earthquake) and predicted magnitude by radius.\n"
                "- **Data table** — filter, sort, and export.\n"
                "- **About** — pipeline integration notes.\n"
                "- **Map** — Texas Basin event/well link explorer."
            )
        with st.expander("Definitions", expanded=False):
            st.markdown(
                "**Radius** is the great-circle distance from well to event.  \n"
                "**Total effect** is the DoWhy-estimated causal coefficient; values are small (scientific notation).  \n"
                "**Hurdle Stage 1** predicts P(earthquake); **Stage 2** predicts magnitude given occurrence."
            )

    view_df = df[
        (df["radius_km"] >= radius_range[0])
        & (df["radius_km"] <= radius_range[1])
        & (df["model_type"].isin(selected_models))
    ].copy()

    if view_df.empty:
        st.warning("No rows match the current filters. Widen the radius range or add models in the sidebar.")
        return

    tabs = st.tabs(
        [
            "Performance",
            "Causal effects",
            "Zones",
            "90-day lookback",
            "Hurdle forecast",
            "Data",
            "About",
            "Basin map",
        ]
    )

    with tabs[0]:
        st.markdown("### Model performance overview")
        st.caption(
            "Predictive quality by **radius** (well-to-event distance). "
            "Higher R² / skill / Spearman ρ and lower MAE indicate better out-of-sample fit."
        )

        # Headline KPIs — best model by predictive R² across filtered radii.
        if "r2" in view_df.columns and view_df["r2"].notna().any():
            by_model = view_df.groupby("model_type", dropna=True).agg(
                r2=("r2", "mean"),
                mae=("mae", "mean"),
                skill=("skill_score", "mean"),
                rho=("spearman_corr", "mean"),
            )
            best = by_model["r2"].idxmax()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Best model (mean R²)", str(best), f"R² {by_model.loc[best, 'r2']:.3f}")
            k2.metric("Avg MAE", f"{by_model['mae'].mean():.3f}")
            k3.metric("Avg skill score", f"{by_model['skill'].mean():.3f}")
            k4.metric("Avg Spearman ρ", f"{by_model['rho'].mean():.3f}")

        _section("Metric curves by radius")
        perf_cols = ["r2", "mae", "skill_score", "spearman_corr"]
        perf_long = view_df.melt(
            id_vars=["radius_km", "model_type"],
            value_vars=[c for c in perf_cols if c in view_df.columns],
            var_name="metric",
            value_name="value",
        )
        perf_long["metric_label"] = perf_long["metric"].map(lambda m: _METRIC_LABELS.get(m, str(m)))
        metric_order = [_METRIC_LABELS.get(c, c) for c in perf_cols if c in view_df.columns]
        fig_perf = px.line(
            perf_long,
            x="radius_km",
            y="value",
            color="model_type",
            facet_row="metric_label",
            markers=True,
            category_orders={"metric_label": metric_order},
            color_discrete_map=_model_color_map(perf_long["model_type"].unique().tolist()),
            facet_row_spacing=0.08,
        )
        fig_perf.update_yaxes(matches=None, title_text="")
        _apply_radius_xaxis(fig_perf)
        _strip_facet_equals(fig_perf)
        fig_perf.update_traces(line=dict(width=2.2), marker=dict(size=6))
        fig_perf.update_layout(legend_title_text="Model", hovermode="x unified")
        style_fig(fig_perf, height=min(260 * max(len(metric_order), 1), 1100))
        st.plotly_chart(fig_perf, use_container_width=True)

    with tabs[1]:
        st.markdown("### Causal effects & mediation analysis")
        st.caption(
            "DoWhy decomposes the effect of injection volume on seismicity into a **direct** "
            "mechanical pathway and an **indirect** pathway mediated by pore pressure. Their sum is the **total effect**. "
            "Coefficients are small, so the y-axis uses scientific notation (e.g. 2.3×10⁻⁵)."
        )

        # Headline KPIs on total effect and mediation share.
        agg_radius = view_df.groupby("radius_km", as_index=False).agg(
            total_effect=("total_effect", "mean"),
            mediation_pct=("mediation_pct", "mean"),
        )
        if not agg_radius.empty:
            peak = agg_radius.loc[agg_radius["total_effect"].idxmax()]
            mean_med = float(agg_radius["mediation_pct"].mean())
            k1, k2, k3 = st.columns(3)
            k1.metric(
                "Peak total effect",
                f"{peak['total_effect']:.2e}",
                help=f"Largest mean total effect occurs at {peak['radius_km']:.0f} km radius.",
            )
            k2.metric("Peak radius", f"{peak['radius_km']:.0f} km")
            k3.metric("Average mediation", f"{mean_med:.1f}%", help="Share of total effect travelling through pore-pressure (P).")

        _section("Effect curves by radius")
        causal_long = view_df.melt(
            id_vars=["radius_km", "model_type"],
            value_vars=["direct_effect", "indirect_effect", "total_effect"],
            var_name="effect_type",
            value_name="effect_value",
        )
        causal_long["effect_label"] = causal_long["effect_type"].map(lambda e: _EFFECT_LABELS.get(e, str(e)))
        fig_causal = px.line(
            causal_long,
            x="radius_km",
            y="effect_value",
            color="effect_label",
            line_dash="model_type",
            markers=True,
            color_discrete_map=_EFFECT_COLOR_MAP,
        )
        fig_causal.add_hline(y=0, line_dash="dot", line_color="#9aa5b1", line_width=1)
        fig_causal.update_traces(line=dict(width=2.2), marker=dict(size=6))
        fig_causal.update_layout(
            xaxis_title="Radius (km)",
            yaxis_title="Causal effect (DoWhy coefficient)",
            legend_title_text="Effect pathway / model",
        )
        fig_causal.update_yaxes(tickformat=".2e")
        style_fig(fig_causal, height=460)
        st.plotly_chart(fig_causal, use_container_width=True)

        if "mediation_pct" in view_df.columns:
            _section("Share of total effect mediated by pore pressure")
            fig_med = px.bar(
                view_df,
                x="radius_km",
                y="mediation_pct",
                color="model_type",
                barmode="group",
                color_discrete_map=_model_color_map(view_df["model_type"].unique().tolist()),
            )
            fig_med.add_hline(
                y=100, line_dash="dash", line_color="#486581", line_width=1,
                annotation_text="100% (fully mediated)", annotation_position="top right",
                annotation_font_color="#486581",
            )
            fig_med.update_layout(
                xaxis_title="Radius (km)",
                yaxis_title="Mediation (% of total effect)",
                legend_title_text="Model",
            )
            style_fig(fig_med, height=380)
            st.plotly_chart(fig_med, use_container_width=True)

    with tabs[2]:
        st.markdown("### Nonlinear & heterogeneous effects — by field zone")
        st.caption(
            "Radii grouped into three operational zones: "
            "**near-field** (1–5 km), **mid-field** (6–10 km), **far-field** (11–20 km). "
            "Near-field effects tend to be dominated by direct mechanical coupling; "
            "far-field effects almost exclusively travel through pore pressure."
        )
        zone_summary = (
            view_df.groupby(["zone", "model_type"], as_index=False)
            .agg(total_effect_mean=("total_effect", "mean"), mediation_mean=("mediation_pct", "mean"))
            .sort_values(["zone", "model_type"])
        )
        zone_summary["zone_label"] = zone_summary["zone"].map(lambda z: _ZONE_BAR_LABELS.get(z, str(z)))
        zmap = _model_color_map(zone_summary["model_type"].unique().tolist())

        _section("Mean total effect by zone")
        col1, col2 = st.columns(2)
        with col1:
            fig_z1 = px.bar(
                zone_summary,
                x="zone_label",
                y="total_effect_mean",
                color="model_type",
                barmode="group",
                color_discrete_map=zmap,
                category_orders={"zone_label": _ZONE_BAR_ORDER},
            )
            fig_z1.update_layout(
                xaxis_title="Field zone",
                yaxis_title="Mean total effect (DoWhy coeff.)",
                legend_title_text="Model",
            )
            fig_z1.update_yaxes(tickformat=".2e")
            fig_z1.update_xaxes(tickangle=0, automargin=True)
            style_fig(fig_z1, height=380)
            st.plotly_chart(fig_z1, use_container_width=True)
        with col2:
            fig_z2 = px.bar(
                zone_summary,
                x="zone_label",
                y="mediation_mean",
                color="model_type",
                barmode="group",
                color_discrete_map=zmap,
                category_orders={"zone_label": _ZONE_BAR_ORDER},
            )
            fig_z2.add_hline(y=100, line_dash="dash", line_color="#9aa5b1", line_width=1)
            fig_z2.update_layout(
                xaxis_title="Field zone",
                yaxis_title="Mean mediation (% of total)",
                legend_title_text="Model",
            )
            fig_z2.update_xaxes(tickangle=0, automargin=True)
            style_fig(fig_z2, height=380)
            st.plotly_chart(fig_z2, use_container_width=True)

        # Quick interpretive table so sponsors don't have to read bars.
        _section("Zone summary (averaged across selected models)")
        zt = (
            view_df.groupby("zone", as_index=False)
            .agg(
                mean_total_effect=("total_effect", "mean"),
                mean_mediation=("mediation_pct", "mean"),
                n_rows=("radius_km", "count"),
            )
            .set_index("zone")
            .reindex(["Near-field (1-5km)", "Mid-field (6-10km)", "Far-field (11-20km)"])
            .reset_index()
        )
        zt.columns = ["Zone", "Mean total effect", "Mean mediation (%)", "# rows"]
        zt["Mean total effect"] = zt["Mean total effect"].map(lambda v: f"{v:.2e}" if pd.notna(v) else "—")
        zt["Mean mediation (%)"] = zt["Mean mediation (%)"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
        st.dataframe(zt, hide_index=True, use_container_width=True)

    with tabs[3]:
        st.markdown(f"### Temporal analysis — {STANDARD_LOOKBACK_DAYS}-day injection lookback")
        st.caption(
            f"Filters to rows that used the project-standard **{STANDARD_LOOKBACK_DAYS}-day** cumulative injection window "
            "(volume and pressure) before each event. The charts show how the total causal effect and predictive R² vary with radius at that lookback."
        )
        temp_df = view_df.copy()
        if "lookback_days" in temp_df.columns:
            temp_df = temp_df[temp_df["lookback_days"].fillna(STANDARD_LOOKBACK_DAYS) == STANDARD_LOOKBACK_DAYS].copy()

        if temp_df.empty:
            present = sorted({int(x) for x in df["lookback_days"].dropna().unique().tolist()}) if "lookback_days" in df.columns else []
            st.warning(
                f"No rows with `lookback_days` = {STANDARD_LOOKBACK_DAYS}. "
                + (f"Values present in CSV: {present}. " if present else "")
                + f"Run the DoWhy pipeline with `--lookback-days {STANDARD_LOOKBACK_DAYS}` and refresh."
            )
        else:
            k1, k2, k3 = st.columns(3)
            peak_eff = temp_df.loc[temp_df["total_effect"].idxmax()]
            peak_r2 = temp_df.loc[temp_df["r2"].idxmax()]
            k1.metric(
                "Peak total effect",
                f"{peak_eff['total_effect']:.2e}",
                help=f"{peak_eff['model_type']} at {peak_eff['radius_km']:.0f} km.",
            )
            k2.metric(
                "Best R² radius",
                f"{peak_r2['radius_km']:.0f} km",
                f"R² {peak_r2['r2']:.3f} · {peak_r2['model_type']}",
            )
            k3.metric("Rows (90-day)", f"{len(temp_df):,}")

            temp_df = temp_df.sort_values(["radius_km", "model_type"])
            cm = _model_color_map(temp_df["model_type"].unique().tolist())
            _section("Total effect vs. R² by radius")
            c1, c2 = st.columns(2)
            f1 = px.line(
                temp_df, x="radius_km", y="total_effect",
                color="model_type", markers=True, color_discrete_map=cm,
            )
            f1.add_hline(y=0, line_dash="dot", line_color="#9aa5b1", line_width=1)
            f1.update_traces(line=dict(width=2.2), marker=dict(size=6))
            f1.update_layout(
                xaxis_title="Radius (km)",
                yaxis_title="Total effect (DoWhy coeff.)",
                legend_title_text="Model",
            )
            f1.update_yaxes(tickformat=".2e")
            style_fig(f1, height=400)
            f2 = px.line(
                temp_df, x="radius_km", y="r2",
                color="model_type", markers=True, color_discrete_map=cm,
            )
            f2.update_traces(line=dict(width=2.2), marker=dict(size=6))
            f2.update_layout(
                xaxis_title="Radius (km)",
                yaxis_title="R² (predictive)",
                legend_title_text="Model",
            )
            style_fig(f2, height=400)
            c1.plotly_chart(f1, use_container_width=True)
            c2.plotly_chart(f2, use_container_width=True)

    with tabs[4]:
        st.markdown("### Earthquake probability & predicted magnitude (hurdle forecast)")
        st.caption(
            "Two-stage hurdle model. **Stage 1** (GBM classifier) answers *how likely is an earthquake?*; "
            "**Stage 2** (regressor on event rows) answers *if one occurs, how large?* "
            "Formation depth (shallow vs. Ellenburger) is a known missing confounder; treat absolute probabilities as lower bounds."
        )

        has_prob = view_df["hurdle_occurrence_prob"].notna().any()
        has_mag = view_df["hurdle_conditional_magnitude"].notna().any()

        if not has_prob and not has_mag:
            st.info(
                "Hurdle outputs not yet present in `dowhy_mediation_analysis.csv`. "
                "Rerun the DoWhy pipeline (Change 7) to populate `hurdle_occurrence_prob` and `hurdle_conditional_magnitude`."
            )
        else:
            hcol = _model_color_map(view_df["model_type"].unique().tolist())
            hurdle_df = view_df.dropna(
                subset=[c for c in ["hurdle_occurrence_prob", "hurdle_conditional_magnitude"] if c in view_df.columns],
                how="all",
            ).copy()

            # ---- Headline metrics ------------------------------------------------
            if has_prob:
                hi_idx = hurdle_df["hurdle_occurrence_prob"].idxmax()
                hi_row = hurdle_df.loc[hi_idx]
                hi_prob = float(hi_row["hurdle_occurrence_prob"])
                hi_mag = hi_row.get("hurdle_conditional_magnitude")
                hi_mag = float(hi_mag) if pd.notna(hi_mag) else None
                hi_radius = float(hi_row["radius_km"])
                hi_model = str(hi_row["model_type"])
                mean_prob = float(hurdle_df["hurdle_occurrence_prob"].mean())
                mean_mag = float(hurdle_df["hurdle_conditional_magnitude"].mean()) if has_mag else None

                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    "Peak P(earthquake)",
                    f"{hi_prob*100:.1f}%",
                    help=f"Highest predicted probability across the filtered radii ({hi_model}, r={hi_radius:.0f} km).",
                )
                m2.metric(
                    "...expected magnitude",
                    f"M {hi_mag:.2f}" if hi_mag is not None else "—",
                    help="Stage-2 predicted magnitude at the peak-probability radius.",
                )
                m3.metric(
                    "Avg P(earthquake)",
                    f"{mean_prob*100:.1f}%",
                    help="Mean Stage-1 probability across the filtered radii and models.",
                )
                m4.metric(
                    "Avg predicted magnitude",
                    f"M {mean_mag:.2f}" if mean_mag is not None else "—",
                    help="Mean Stage-2 conditional magnitude across the filtered radii and models.",
                )

            # ---- Combined dual-axis plot ----------------------------------------
            if has_prob and has_mag:
                _section("Probability of earthquake  |  Predicted magnitude — by radius")
                combined = (
                    hurdle_df.groupby("radius_km", as_index=False)
                    .agg(
                        prob=("hurdle_occurrence_prob", "mean"),
                        magnitude=("hurdle_conditional_magnitude", "mean"),
                    )
                    .sort_values("radius_km")
                )
                combined["prob_pct"] = combined["prob"] * 100.0

                fig_combined = go.Figure()
                fig_combined.add_bar(
                    x=combined["radius_km"],
                    y=combined["prob_pct"],
                    name="P(earthquake)",
                    marker=dict(color=COLOR_PROB, line=dict(width=0)),
                    opacity=0.85,
                    hovertemplate="Radius %{x} km<br>P(earthquake): %{y:.1f}%<extra></extra>",
                    yaxis="y1",
                )
                fig_combined.add_scatter(
                    x=combined["radius_km"],
                    y=combined["magnitude"],
                    name="Predicted magnitude",
                    mode="lines+markers",
                    line=dict(color=COLOR_MAG, width=3),
                    marker=dict(size=10, line=dict(color="#ffffff", width=1.5)),
                    hovertemplate="Radius %{x} km<br>Predicted magnitude: M %{y:.2f}<extra></extra>",
                    yaxis="y2",
                )
                fig_combined.update_layout(
                    xaxis_title="Radius (km)",
                    yaxis=dict(title="Probability of earthquake (%)", rangemode="tozero", side="left"),
                    yaxis2=dict(title="Predicted magnitude (M)", overlaying="y", side="right", showgrid=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    bargap=0.25,
                )
                style_fig(fig_combined, height=440)
                st.plotly_chart(fig_combined, use_container_width=True)

            # ---- Per-model side-by-side lines -----------------------------------
            _section("By model")
            c1, c2 = st.columns(2)
            with c1:
                if has_prob:
                    st.markdown("**Stage 1 — probability of an earthquake**")
                    plot_df = hurdle_df.copy()
                    plot_df["prob_pct"] = plot_df["hurdle_occurrence_prob"] * 100.0
                    fig_p = px.line(
                        plot_df, x="radius_km", y="prob_pct",
                        color="model_type", markers=True, color_discrete_map=hcol,
                    )
                    fig_p.update_traces(
                        line=dict(width=2.4), marker=dict(size=7),
                        hovertemplate="Radius %{x} km<br>P(earthquake): %{y:.1f}%<extra></extra>",
                    )
                    fig_p.update_layout(
                        xaxis_title="Radius (km)",
                        yaxis_title="P(earthquake) (%)",
                        legend_title_text="Model",
                        yaxis=dict(rangemode="tozero"),
                    )
                    style_fig(fig_p, height=360)
                    st.plotly_chart(fig_p, use_container_width=True)
                else:
                    st.info("`hurdle_occurrence_prob` not present in source file.")
            with c2:
                if has_mag:
                    st.markdown("**Stage 2 — predicted earthquake magnitude**")
                    fig_m = px.line(
                        hurdle_df, x="radius_km", y="hurdle_conditional_magnitude",
                        color="model_type", markers=True, color_discrete_map=hcol,
                    )
                    fig_m.update_traces(
                        line=dict(width=2.4), marker=dict(size=7),
                        hovertemplate="Radius %{x} km<br>Predicted magnitude: M %{y:.2f}<extra></extra>",
                    )
                    fig_m.update_layout(
                        xaxis_title="Radius (km)",
                        yaxis_title="Predicted magnitude (M | occurrence)",
                        legend_title_text="Model",
                    )
                    style_fig(fig_m, height=360)
                    st.plotly_chart(fig_m, use_container_width=True)
                else:
                    st.info("`hurdle_conditional_magnitude` not present in source file.")

            # ---- Forecast table --------------------------------------------------
            _section("Forecast table (averaged across selected models)")
            summary = (
                hurdle_df.groupby("radius_km", as_index=False)
                .agg(
                    prob=("hurdle_occurrence_prob", "mean"),
                    magnitude=("hurdle_conditional_magnitude", "mean"),
                )
                .sort_values("radius_km")
            )
            summary["Probability of earthquake"] = (summary["prob"] * 100).round(1).astype(str) + "%"
            summary["Predicted magnitude"] = summary["magnitude"].map(lambda v: f"M {v:.2f}" if pd.notna(v) else "—")
            summary = summary.rename(columns={"radius_km": "Radius (km)"})[
                ["Radius (km)", "Probability of earthquake", "Predicted magnitude"]
            ]
            st.dataframe(summary, hide_index=True, use_container_width=True)

            # ---- Diagnostics (collapsed) ----------------------------------------
            has_auc = view_df["hurdle_stage1_auc"].notna().any()
            has_mae = view_df["hurdle_stage2_mae"].notna().any()
            if has_auc or has_mae:
                with st.expander("Model quality diagnostics — ROC-AUC, MAE", expanded=False):
                    d1, d2 = st.columns(2)
                    if has_auc:
                        with d1:
                            st.markdown("**Stage 1 — ROC-AUC**")
                            fig_auc = px.line(
                                view_df, x="radius_km", y="hurdle_stage1_auc",
                                color="model_type", markers=True, color_discrete_map=hcol,
                            )
                            fig_auc.update_layout(
                                xaxis_title="Radius (km)", yaxis_title="ROC-AUC", legend_title_text="Model",
                            )
                            style_fig(fig_auc, height=320)
                            st.plotly_chart(fig_auc, use_container_width=True)
                    if has_mae:
                        with d2:
                            st.markdown("**Stage 2 — MAE (magnitude units)**")
                            fig_mae = px.line(
                                view_df, x="radius_km", y="hurdle_stage2_mae",
                                color="model_type", markers=True, color_discrete_map=hcol,
                            )
                            fig_mae.update_layout(
                                xaxis_title="Radius (km)", yaxis_title="MAE", legend_title_text="Model",
                            )
                            style_fig(fig_mae, height=320)
                            st.plotly_chart(fig_mae, use_container_width=True)

    with tabs[5]:
        st.markdown("### Interactive data table")
        st.caption(
            "All filtered rows from the DoWhy mediation analysis. "
            "Sort by clicking column headers; download as CSV with the button below."
        )
        display_df = view_df.sort_values(["radius_km", "model_type"]).reset_index(drop=True)

        # Pretty column configuration — format numeric columns without changing underlying data.
        col_cfg: dict = {
            "radius_km": st.column_config.NumberColumn("Radius (km)", format="%d"),
            "model_type": st.column_config.TextColumn("Model"),
            "r2": st.column_config.NumberColumn("R²", format="%.3f"),
            "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
            "skill_score": st.column_config.NumberColumn("Skill", format="%.3f"),
            "spearman_corr": st.column_config.NumberColumn("Spearman ρ", format="%.3f"),
            "direct_effect": st.column_config.NumberColumn("Direct effect", format="%.3e"),
            "indirect_effect": st.column_config.NumberColumn("Indirect effect", format="%.3e"),
            "total_effect": st.column_config.NumberColumn("Total effect", format="%.3e"),
            "mediation_pct": st.column_config.NumberColumn("Mediation (%)", format="%.1f"),
            "lookback_days": st.column_config.NumberColumn("Lookback (d)", format="%d"),
            "pressure_missing_pct": st.column_config.NumberColumn("Pressure missing (%)", format="%.1f"),
            "hurdle_occurrence_prob": st.column_config.NumberColumn("P(earthquake)", format="%.3f"),
            "hurdle_conditional_magnitude": st.column_config.NumberColumn("Predicted M", format="%.2f"),
            "hurdle_stage1_auc": st.column_config.NumberColumn("Stage-1 AUC", format="%.3f"),
            "hurdle_stage2_mae": st.column_config.NumberColumn("Stage-2 MAE", format="%.3f"),
            "zone": st.column_config.TextColumn("Zone"),
        }
        st.dataframe(
            display_df,
            column_config={k: v for k, v in col_cfg.items() if k in display_df.columns},
            use_container_width=True,
            hide_index=True,
            height=520,
        )
        st.download_button(
            "Download filtered results as CSV",
            data=view_df.to_csv(index=False),
            file_name="geminae_dashboard_filtered_results.csv",
            mime="text/csv",
            type="primary",
        )

    with tabs[6]:
        st.markdown("### About this dashboard")

        a1, a2 = st.columns([1.2, 1])
        with a1:
            st.markdown(
                "**Project Geminae** monitors induced seismicity in the Midland Basin by combining the TexNet earthquake "
                "catalog, saltwater-disposal (SWD) injection records, and the Horne et al. (2023) basement-fault map. "
                "The pipeline estimates both causal (DoWhy mediation) and predictive (hurdle) effects of injection on "
                "seismic activity across 20 radii from 1 km to 20 km."
            )
            _section("Data pipeline")
            st.markdown(
                "1. **Raw import** — `swd_data_import.py`, `seismic_data_import.py`\n"
                "2. **Spatial join** — `merge_seismic_swd.py`\n"
                "3. **Injection lookback** — `filter_active_wells_before_events.py`, `filter_merge_events_and_nonevents.py`\n"
                "4. **Fault features** — `add_geoscience_to_event_well_links_with_injection.py`\n"
                "5. **Simple DoWhy (well + event level)** — `dowhy_simple_all.py`, `dowhy_simple_all_aggregate.py`\n"
                "6. **Bootstrap CI + hurdle model** — `dowhy_ci.py`, `dowhy_ci_aggregated.py`\n\n"
                "Run the full sequence with:"
            )
            st.code("python run_all.py", language="bash")
            st.markdown("Launch the dashboard with:")
            st.code(
                "python -m streamlit run dashboard_app.py --server.headless true --server.port 8502",
                language="bash",
            )

        with a2:
            _section("Data sources")
            st.markdown(
                f"**Primary input:** `{source}`  \n"
                f"**Last refreshed:** {modified_iso or '—'}  \n"
                f"**Rows (all):** {n_rows_total:,}  \n"
                f"**Models:** {n_models}  \n"
                f"**Radii covered:** {radii_bounds[0]}–{radii_bounds[1]} km"
            )
            _section("Model definitions")
            st.markdown(
                "- **OLS** — linear baseline.\n"
                "- **DML-GBM / RF / Lasso** — DoubleML with gradient-boosting / random-forest / lasso nuisance learners.\n"
                "- **NonParamDML** — nonparametric DoubleML estimator.\n"
                "- **Hurdle Stage 1** — GBM binary classifier → P(earthquake).\n"
                "- **Hurdle Stage 2** — regressor on event rows → E[M | occurrence]."
            )
            _section("Caveats")
            st.markdown(
                "- **Formation depth** (shallow vs. Ellenburger) is a known missing confounder for Stage 1.\n"
                "- **Causal coefficients** are small (×10⁻⁵); compare *ratios* across radii, not absolute magnitudes.\n"
                "- The **90-day lookback** is the project standard; shorter windows under-count cumulative injection."
            )

        if CHANGE7_BRIEF_HTML.exists():
            with st.expander("Change 7 brief — two-stage hurdle model (Apr 2026)", expanded=False):
                st.caption(
                    "Executive summary shipped with the repo (`ProjectGeminae_Change7_Brief_20260405.html`). "
                    "Scroll inside the panel to read the full brief."
                )
                components.html(CHANGE7_BRIEF_HTML.read_text(encoding="utf-8"), height=720, scrolling=True)

    with tabs[7]:
        st.markdown("### Texas Basin map — event / well link explorer")
        st.caption(
            "Interactive map of TexNet events (blue) and SWD injection wells (red) within the Midland Basin. "
            "Select a radius to load the event–well link network; click any point to focus on its linked counterparts."
        )
        _, spatial_radii = discover_spatial_index()
        map_radius_options = [r for r in spatial_radii if radius_range[0] <= r <= radius_range[1]] if spatial_radii else list(range(radius_range[0], radius_range[1] + 1))
        if not map_radius_options:
            map_radius_options = list(range(radius_range[0], radius_range[1] + 1))
        c1, c2 = st.columns([1, 2])
        with c1:
            map_radius = st.selectbox("Map radius (km)", map_radius_options, index=0, key="map_radius")
        with c2:
            st.caption("Map aggregates all available lookback exports (30-day, 90-day, etc.) for the selected radius.")

        links_df, map_status = load_spatial_links_by_radius(int(map_radius))
        if links_df is None:
            st.info(map_status)
        else:
            st.caption(map_status)
            st.caption(
                "**Click** an **event** marker to show that event and **all linked wells**. "
                "**Click** a **well** marker to show that well and **all linked events**. "
                "Use the lists or **Clear selection** to reset."
            )
            col_map, col_ctrl = st.columns([2.4, 1], gap="medium")

            with col_ctrl:
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

                with col_ctrl:
                    picked_event = st.selectbox(
                        "Selected event (click map or choose here)",
                        [""] + event_options,
                        index=0 if not st.session_state["map_selected_event"] else ([""] + event_options).index(str(st.session_state["map_selected_event"])) if str(st.session_state["map_selected_event"]) in event_options else 0,
                        key="map_pick_event",
                    )
                    picked_well = st.selectbox(
                        "Selected well (click map or choose here)",
                        [""] + well_options,
                        index=0 if not st.session_state["map_selected_well"] else ([""] + well_options).index(str(st.session_state["map_selected_well"])) if str(st.session_state["map_selected_well"]) in well_options else 0,
                        key="map_pick_well",
                    )
                    if st.button("Clear selection", key="map_clear_sel"):
                        st.session_state["map_selected_event"] = None
                        st.session_state["map_selected_well"] = None
                        st.rerun()

                # Keep one active focus: event -> all linked wells on map; well -> all linked events.
                if picked_event:
                    st.session_state["map_selected_event"] = str(picked_event)
                    st.session_state["map_selected_well"] = None
                    picked_well = ""
                elif picked_well:
                    st.session_state["map_selected_well"] = str(picked_well)
                    st.session_state["map_selected_event"] = None
                    picked_event = ""
                elif not picked_event and not picked_well:
                    # User set both dropdowns to empty — drop any prior map selection.
                    if st.session_state.get("map_selected_event") is not None or st.session_state.get("map_selected_well") is not None:
                        st.session_state["map_selected_event"] = None
                        st.session_state["map_selected_well"] = None

                # Prefer session state so map clicks and dropdowns stay aligned after rerun.
                show_ev = picked_event or st.session_state.get("map_selected_event")
                show_wl = picked_well or st.session_state.get("map_selected_well")
                if show_ev:
                    show_ev, show_wl = str(show_ev), None
                elif show_wl:
                    show_wl, show_ev = str(show_wl), None
                else:
                    show_ev, show_wl = None, None

                if show_ev:
                    linked = links_df[links_df["EventID"].astype(str) == show_ev].copy()
                    highlight_event_ids = {show_ev}
                    highlight_well_ids = set(linked["api_number"].astype(str).tolist())
                elif show_wl:
                    linked = links_df[links_df["api_number"].astype(str) == show_wl].copy()
                    highlight_event_ids = set(linked["EventID"].astype(str).tolist())
                    highlight_well_ids = {show_wl}
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
                if show_ev or show_wl:
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

                # Smaller markers + transparency reduce overplotting at basin scale.
                points_plot["marker_size"] = 5.0
                event_mask = points_plot["point_type"].eq("Event")
                if event_mask.any():
                    event_mag = pd.to_numeric(points_plot.loc[event_mask, "metric_value"], errors="coerce")
                    if event_mag.notna().any():
                        mag_lo = float(event_mag.min())
                        mag_hi = float(event_mag.max())
                        if mag_hi > mag_lo:
                            norm = (event_mag - mag_lo) / (mag_hi - mag_lo)
                            scaled = 5 + norm * 7  # ~5..12 px equivalent
                        else:
                            scaled = pd.Series(8.0, index=event_mag.index)
                        points_plot.loc[event_mask, "marker_size"] = scaled.fillna(7.0)
                    else:
                        points_plot.loc[event_mask, "marker_size"] = 7.0

                fig_map = px.scatter_mapbox(
                    points_plot,
                    lat="lat",
                    lon="lon",
                    color="highlight_role",
                    size="marker_size",
                    size_max=12,
                    hover_name="label",
                    hover_data={"EventID": True, "api_number": True, "metric_value": ":.3f", "marker_size": False, "highlight_role": False},
                    custom_data=["point_type", "EventID", "api_number"],
                    color_discrete_map={
                        "Event": "#1e88e5",
                        "Well": "#c62828",
                        "Selected Event": "#00c853",
                        "Selected Well": "#8e24aa",
                    },
                    zoom=5.2,
                    height=620,
                )
                fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select")
                fig_map.update_traces(marker=dict(opacity=0.62, sizemin=3))

                with col_map:
                    selected = st.plotly_chart(
                        fig_map,
                        on_select="rerun",
                        selection_mode="points",
                        key="tx_basin_map",
                        use_container_width=True,
                    )

                if selected and selected.get("selection", {}).get("points"):
                    pt = selected["selection"]["points"][0]
                    p_type, ev_id, api = _parse_map_selection_point(pt)
                    _changed = False
                    if p_type == "Event" and ev_id:
                        if str(st.session_state.get("map_selected_event")) != str(ev_id) or st.session_state.get("map_selected_well") is not None:
                            st.session_state["map_selected_event"] = str(ev_id)
                            st.session_state["map_selected_well"] = None
                            _changed = True
                    elif p_type == "Well" and api:
                        if str(st.session_state.get("map_selected_well")) != str(api) or st.session_state.get("map_selected_event") is not None:
                            st.session_state["map_selected_well"] = str(api)
                            st.session_state["map_selected_event"] = None
                            _changed = True
                    if _changed:
                        st.rerun()

                if show_ev:
                    n_wells = linked["api_number"].nunique()
                    apis = sorted(linked["api_number"].dropna().astype(str).unique().tolist())
                    api_preview = ", ".join(apis[:30]) + (" …" if len(apis) > 30 else "")
                    st.success(
                        f"**Event `{show_ev}`** — **{n_wells}** linked well(s). "
                        f"API numbers: {api_preview}"
                    )
                elif show_wl:
                    n_ev = linked["EventID"].nunique()
                    evs = sorted(linked["EventID"].dropna().astype(str).unique().tolist())
                    ev_preview = ", ".join(evs[:30]) + (" …" if len(evs) > 30 else "")
                    st.success(
                        f"**Well `{show_wl}`** — **{n_ev}** linked event(s). "
                        f"Event IDs: {ev_preview}"
                    )

                st.caption(f"**{len(linked):,}** link row(s) in the table below (one row per event–well pair).")
                show_cols = [c for c in ["EventID", "api_number", "Local Magnitude", "Vol Prev N (BBLs)", "Distance from Well to Event", "Nearest Fault Dist (km)"] if c in linked.columns]
                st.dataframe(linked[show_cols].head(500))


if __name__ == "__main__":
    main()
