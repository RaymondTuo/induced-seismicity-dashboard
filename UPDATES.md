# Project Geminae — Change Log

All notable changes to the induced seismicity causal inference pipeline and dashboard are documented here.

---

## Change 7 — 90-Day Lookback Remodel (2026-04-05)

**Contributors:** 


- Merged [RaymondTuo/90-day-lookback-remodel](https://github.com/RaymondTuo/90-day-lookback-remodel) via PR #1
- Extended injection lookback window from 30 days to 90 days across the full pipeline
- Applied 90-day lookback consistently through `run_all.py`, `dowhy_ci.py`, `dowhy_ci_aggregated.py`, and `patch_lookback_into_faults.py`
- Added `run_lookback_sweep.py` to sweep and compare multiple lookback windows
- Updated methodology notes to reflect revised temporal window

---

## Change 6 — Analysis Expansion (2026-04-02)

- Expanded causal analysis outputs
- Updated sponsor-facing brief with revised effect size estimates

---

## Change 5 — Hurdle Model & Mediation Refinements (2026-03-30)

- Added hurdle model outputs (occurrence probability vs. conditional magnitude) to the dashboard
- Refined mediation analysis (direct vs. indirect effects via injection pressure)
- Updated briefs with zone-level effect breakdown (near-field ≤5km vs. far-field >10km)

---

## Change 3 — Dashboard Hardening & Report Update (2026-03-17)

- Added one-command local dashboard launcher (`run_dashboard.bat`)
- Added `DASHBOARD.md` quick-start guide
- Improved dashboard sections: temporal sensitivity, diagnostics, interactive data table
- Refined causal DAG and added methodology notes

---

## Change 2 — Pipeline Refinements (2026-02-27)

- Iterative improvements to seismic–SWD data merging and well–event linkage
- Updated geoscience feature engineering (`add_geoscience_to_event_well_links_with_injection.py`)
- Revised filter logic for active wells before events

---

## Change 1 — Initial Pipeline & Dashboard (2026-02-27)

**Initial commit establishing the full project.**

### Pipeline
- `seismic_data_import.py` — imports and cleans seismic catalog
- `swd_data_import.py` — imports saltwater disposal (SWD) injection records
- `merge_seismic_swd.py` — spatially joins seismic events to injection wells by radius (1–20 km)
- `filter_active_wells_before_events.py` — filters to wells active in the lookback window before each event
- `filter_merge_events_and_nonevents.py` — constructs event/non-event case–control dataset
- `add_geoscience_to_event_well_links_with_injection.py` — appends fault proximity and geological features
- `patch_lookback_into_faults.py` — patches lookback-window injection totals into fault-linked records
- `run_all.py` — end-to-end pipeline orchestrator with checkpoint support

### Causal Inference
- `dowhy_ci.py` — event-level DoWhy causal inference (ATE, mediation) at each radius
- `dowhy_ci_aggregated.py` — aggregated/pooled causal estimates
- `dowhy_simple_all.py` — simplified causal sweep over all radii
- `dowhy_simple_all_aggregate.py` — aggregate version of the simplified sweep
- `causal_poe_curves.py` — probability-of-exceedance curves from causal estimates
- `induced_seismicity_scaling_plots.py` — scaling relationship visualizations
- `measure_balrog.py` — model diagnostics and benchmarking

### Dashboard
- `dashboard_app.py` — Plotly Dash app serving all analysis outputs interactively

### Data
- `data/Horne_et_al._2023_MB_BSMT_FSP_V1.shp` — fault segment shapefile (Horne et al., 2023)

---

## Repo Reorganization (2026-04-15)

- Moved generated reports, briefs, posters, and doc-building scripts to `docs/` (gitignored — not committed)
- Moved fault shapefile to `data/`
- Cleaned `.gitignore` to exclude `__pycache__/`, log files, `dag_*` runtime artifacts, `checkpoints/`, `results/`, and `pipeline_output_*/`
- Added this `UPDATES.md` as a persistent changelog
