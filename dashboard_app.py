from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Project Geminae Dashboard", page_icon=":earth_americas:", layout="wide")


CSV_CANDIDATES = [
    Path("dowhy_mediation_analysis.csv"),
    Path("results") / "dowhy_mediation_analysis.csv",
]


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
    return df


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, str]:
    csv_path = find_csv_path()
    if csv_path is None:
        return normalize_columns(build_fallback_data()), "fallback"
    df = pd.read_csv(csv_path)
    return normalize_columns(df), str(csv_path)


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
    lookbacks = sorted(df["lookback_days"].dropna().astype(int).unique().tolist())
    selected_lookback = st.sidebar.selectbox("Lookback window (days)", lookbacks, index=0)

    view_df = df[
        (df["radius_km"] >= radius_range[0])
        & (df["radius_km"] <= radius_range[1])
        & (df["model_type"].isin(selected_models))
        & (df["lookback_days"] == selected_lookback)
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
        st.plotly_chart(fig_perf, use_container_width=True)

    with tabs[1]:
        st.subheader("Causal Effects & Mediation Analysis")
        causal_long = view_df.melt(
            id_vars=["radius_km", "model_type"],
            value_vars=["direct_effect", "indirect_effect", "total_effect"],
            var_name="effect_type",
            value_name="effect_value",
        )
        fig_causal = px.line(causal_long, x="radius_km", y="effect_value", color="effect_type", line_dash="model_type", markers=True)
        st.plotly_chart(fig_causal, use_container_width=True)
        if "mediation_pct" in view_df.columns:
            fig_med = px.bar(view_df, x="radius_km", y="mediation_pct", color="model_type", barmode="group")
            fig_med.add_hline(y=100, line_dash="dash", line_color="black")
            st.plotly_chart(fig_med, use_container_width=True)

    with tabs[2]:
        st.subheader("Nonlinear & Heterogeneous Effects")
        zone_summary = (
            view_df.groupby(["zone", "model_type"], as_index=False)
            .agg(total_effect_mean=("total_effect", "mean"), mediation_mean=("mediation_pct", "mean"))
            .sort_values(["zone", "model_type"])
        )
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.bar(zone_summary, x="zone", y="total_effect_mean", color="model_type", barmode="group"), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(zone_summary, x="zone", y="mediation_mean", color="model_type", barmode="group"), use_container_width=True)

    with tabs[3]:
        st.subheader("Model Diagnostics & Transformations")
        c1, c2 = st.columns(2)
        with c1:
            if "log_transform_applied" in view_df.columns:
                log_counts = view_df["log_transform_applied"].value_counts(dropna=False).rename_axis("log_transform").reset_index(name="count")
                st.plotly_chart(px.pie(log_counts, names="log_transform", values="count"), use_container_width=True)
        with c2:
            if view_df["pressure_missing_pct"].notna().any():
                st.plotly_chart(px.line(view_df, x="radius_km", y="pressure_missing_pct", color="model_type", markers=True), use_container_width=True)
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
            c1.plotly_chart(px.line(sens, x="lookback_days", y="mean_total_effect", color="model_type", markers=True), use_container_width=True)
            c2.plotly_chart(px.line(sens, x="lookback_days", y="mean_r2", color="model_type", markers=True), use_container_width=True)
        else:
            st.info("Only one lookback window found. Add multiple lookback outputs to enable sensitivity plots.")

    with tabs[5]:
        st.subheader("Occurrence vs Magnitude Modeling (Hurdle Model)")
        c1, c2 = st.columns(2)
        with c1:
            if view_df["hurdle_occurrence_prob"].notna().any():
                st.plotly_chart(px.line(view_df, x="radius_km", y="hurdle_occurrence_prob", color="model_type", markers=True), use_container_width=True)
            else:
                st.info("`hurdle_occurrence_prob` not present in source file.")
        with c2:
            if view_df["hurdle_conditional_magnitude"].notna().any():
                st.plotly_chart(px.line(view_df, x="radius_km", y="hurdle_conditional_magnitude", color="model_type", markers=True), use_container_width=True)
            else:
                st.info("`hurdle_conditional_magnitude` not present in source file.")

    with tabs[6]:
        st.subheader("Interactive Data Table")
        st.dataframe(view_df.sort_values(["radius_km", "model_type"]), use_container_width=True)
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
        st.code(
            "python -m streamlit run dashboard_app.py --server.headless true",
            language="bash",
        )


if __name__ == "__main__":
    main()
