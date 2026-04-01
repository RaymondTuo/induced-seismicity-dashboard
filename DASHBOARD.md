# Dashboard Quick Start

## Run locally (recommended)

1. Create the Python 3.12 environment once:

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. Start the app:

```bash
run_dashboard.bat
```

3. Open:

`http://localhost:8502`

## Dashboard sections

1. Model Performance Overview (R2, MAE, Skill Score, Spearman by radius)
2. Causal Effects and Mediation (direct, indirect, total, mediation %)
3. Nonlinear and Heterogeneous Effects (zone-level effect breakdown)
4. Diagnostics and Transformations (log-transform flags, pressure missingness)
5. Temporal Sensitivity (lookback window comparison, if available)
6. Hurdle Model (occurrence probability vs conditional magnitude)
7. Interactive Data Table (filter/sort/download)
8. Automated Updates and Integration (CSV-driven refresh)

## Data source expectations

- Primary file: `dowhy_mediation_analysis.csv` in project root or `results/`
- Optional columns handled when present:
  - `lookback_days`
  - `pressure_missing_pct`
  - `log_transform_applied`
  - `hurdle_occurrence_prob`
  - `hurdle_conditional_magnitude`

If the file is not found, the app uses a fallback demo dataset so layout and interactions remain testable.
