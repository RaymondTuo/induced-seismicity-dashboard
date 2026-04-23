import argparse
import copy
import numpy as np
import pandas as pd
import warnings
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from scipy.stats import spearmanr, t as student_t
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dowhy import CausalModel
import dowhy.causal_estimators.linear_regression_estimator
import networkx as nx
import os
import re
import sys
from pathlib import Path

# Windows consoles often default to cp1252; pipeline logs use UTF-8 symbols.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from datetime import datetime

# --- CHANGE 5: CLI arguments for lookback window and log-transform control ---
_parser = argparse.ArgumentParser()
_parser.add_argument("--lookback-days", type=int, default=90,
                     help="Lookback window in days (default: 90, selected as optimal in Change 5). "
                          "Reads files tagged with that window (e.g. 90d).")
_parser.add_argument("--no-log-transform", action="store_true",
                     help="Disable Change 3 log-transforms on W and P. Use for lookback "
                          "sweep to identify optimal window in raw space before re-applying "
                          "the transform.")
_args, _ = _parser.parse_known_args()
LOOKBACK_DAYS_ARG = _args.lookback_days
LOOKBACK_TAG = f"{LOOKBACK_DAYS_ARG}d" if LOOKBACK_DAYS_ARG is not None else None
APPLY_LOG_TRANSFORM = not _args.no_log_transform
# --- END CHANGE 5 ---

# Change 6: zero-fill pressure; binary P_missing as covariate (not median P).
P_MISSING_COL = "P_missing"

# --- CHANGE 1: New imports for nonlinear causal estimators ---
# GradientBoostingRegressor and RandomForestRegressor are used as the model_y,
# model_t, and model_final components inside EconML's NonParamDML estimator,
# which DoWhy invokes via the string "backdoor.econml.dml.NonParamDML".
# Lasso is included as a regularised-linear midpoint between OLS and fully nonlinear.
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso

# econml.dml is loaded dynamically by DoWhy from the method_name string.
# Importing here ensures the package is present and surfaces a clear ImportError
# if econml is not installed, rather than a cryptic DoWhy failure later.
try:
    from econml.dml import LinearDML  # noqa: F401
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
# --- END CHANGE 1: imports ---


def _econml_ate(Y, T, W_confounders, model_y, model_t):
    """Estimate ATE via LinearDML — nonlinear nuisance models, linear final stage.

    DoWhy 0.14's EconML bridge (backdoor.econml.dml.NonParamDML) returns
    near-identical values to OLS regardless of model_y/model_t because it does
    not correctly wire the init_params through to the EconML estimator.
    Calling LinearDML directly ensures GBM/RF/Lasso nuisance models are actually
    used for residualization, producing genuinely model-specific causal estimates.

    LinearDML is appropriate here because we want a single scalar ATE (comparable
    to OLS's regression coefficient) rather than a heterogeneous CATE surface.
    """
    est = LinearDML(
        model_y=copy.deepcopy(model_y),
        model_t=copy.deepcopy(model_t),
        discrete_treatment=False,
        cv=3,
        random_state=42,
    )
    est.fit(
        Y=np.asarray(Y, dtype=float),
        T=np.asarray(T, dtype=float),
        X=None,
        W=np.asarray(W_confounders, dtype=float),
    )
    return float(est.ate())


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# CHANGE 1: Multi-model configuration block
# =============================================================================
# Each entry in MODEL_CONFIGS defines a causal estimator that will be run on
# every radius file. The pipeline runs ALL models in a single pass and writes
# a combined output CSV with a 'model_name' column for direct comparison.
#
# Design rationale:
#   - 'ols_baseline'  : the original OLS estimator, kept unchanged for paper
#                       comparison. Produces exact p-values via statsmodels.
#   - 'gbm_nonparam'  : primary new estimator per the Feb-18 spec. GBM handles
#                       nonlinear W→S and W→P relationships without assuming
#                       a straight-line response.
#   - 'rf_nonparam'   : Random Forest. Less prone to overfitting than GBM at
#                       small radii (1-3 km) where n < 5,000.
#   - 'lasso_nonparam': L1-regularised linear via NonParamDML. A regularised
#                       midpoint between OLS and fully nonlinear — useful for
#                       checking whether nonlinear gains are real or spurious.
#
# 'uses_ols_pvalues': True  → sm.OLS p-value extraction is valid for this model.
#                    False → p-values are derived from bootstrap CIs instead
#                            (see _bootstrap_pvalue()).
#
# 'predictive_model': sklearn estimator used in calculate_predictive_r2() so
#                     the predictive hold-out test uses the same model family
#                     as the causal estimation.
#
# NOTE: model instances here are templates. copy.deepcopy() is called inside
# each estimation function so the same fitted state is never reused across files.

MODEL_CONFIGS = {
    "ols_baseline": {
        # Original OLS estimator — unchanged from the pre-Change-1 version.
        "method_name": "backdoor.linear_regression",
        "method_params": {},
        "uses_ols_pvalues": True,
        "predictive_model": LinearRegression(),
    },
    "gbm_nonparam": {
        # CHANGE 1 (primary): Gradient Boosted Machine via EconML NonParamDML.
        # model_y and model_t are first-stage nuisance models that residualise
        # the outcome (S) and treatment (W) against the confounders (G1, G2).
        # model_final estimates the causal effect from those residuals.
        # All three stages use GBM for fully nonparametric estimation.
        "method_name": "backdoor.econml.dml.NonParamDML",
        "method_params": {
            "init_params": {
                "model_y": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "model_t": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "model_final": GradientBoostingRegressor(n_estimators=100, random_state=42),
            },
            "fit_params": {},
        },
        "uses_ols_pvalues": False,
        "predictive_model": GradientBoostingRegressor(n_estimators=100, random_state=42),
    },
    "rf_nonparam": {
        "method_name": "backdoor.econml.dml.NonParamDML",
        "method_params": {
            "init_params": {
                "model_y": RandomForestRegressor(n_estimators=100, random_state=42),
                "model_t": RandomForestRegressor(n_estimators=100, random_state=42),
                "model_final": RandomForestRegressor(n_estimators=100, random_state=42),
            },
            "fit_params": {},
        },
        "uses_ols_pvalues": False,
        "predictive_model": RandomForestRegressor(n_estimators=100, random_state=42),
    },
    "lasso_nonparam": {
        "method_name": "backdoor.econml.dml.NonParamDML",
        "method_params": {
            "init_params": {
                "model_y": Lasso(alpha=0.01),
                "model_t": Lasso(alpha=0.01),
                "model_final": Lasso(alpha=0.01),
            },
            "fit_params": {},
        },
        "uses_ols_pvalues": False,
        "predictive_model": Lasso(alpha=0.01),
    },
}

# If econml is unavailable (e.g. Python 3.14 without wheels), fall back to OLS-only.
if not ECONML_AVAILABLE:
    MODEL_CONFIGS = {
        "ols_baseline": MODEL_CONFIGS["ols_baseline"],
    }
    print(
        "Warning: econml not installed; running OLS-only mode. "
        "Use Python 3.12 (see .python-version), recreate the venv, and run: pip install -r requirements.txt"
    )
# =============================================================================
# END CHANGE 1: Model configuration block
# =============================================================================


print("Current working directory:", os.getcwd())

# Navigate to the parent directory if we're in .git-rewrite
if os.getcwd().endswith('.git-rewrite'):
    parent_dir = Path(os.getcwd()).parent
    os.chdir(parent_dir)
    print(f"Changed to parent directory: {os.getcwd()}")

# Also try going up one more level if needed
if not any(f.endswith('.csv') for f in os.listdir('.') if os.path.isfile(f)):
    if Path('..').exists():
        os.chdir('..')
        print(f"Changed to: {os.getcwd()}")

print("\nLooking for data files...")


def find_csv_files():
    csv_files = []

    # Look in current directory
    for file in os.listdir('.'):
        if file.endswith('.csv') and os.path.isfile(file):
            csv_files.append(file)

    # Look in common subdirectories
    for subdir in ['data', 'Data', 'csv', 'CSV']:
        if os.path.exists(subdir) and os.path.isdir(subdir):
            try:
                for file in os.listdir(subdir):
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(subdir, file))
            except PermissionError:
                continue

    return csv_files


def extract_radius(filename):
    """Extract radius from filename"""
    match = re.search(r'(\d+)km', filename)
    return int(match.group(1)) if match else 0


def find_column(frags, columns):
    """Return the first column that contains all substrings in frags."""
    for c in columns:
        if all(f.lower() in c.lower() for f in frags):
            return c
    return None


def safe(col: str) -> str:
    """Convert a column name to a strict Python identifier."""
    return re.sub(r"\W", "_", col)


def calculate_vif(X):
    """Calculate VIF for each variable in design matrix (skip ill-conditioned columns)."""
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vifs = []
    for i in range(X.shape[1]):
        try:
            vifs.append(variance_inflation_factor(np.asarray(X.values, dtype=float), i))
        except Exception:
            vifs.append(np.nan)
    vif_data["VIF"] = vifs
    return vif_data


def fit_ols_stable(endog, exog):
    """OLS fit resilient to ill-conditioned designs (SVD / rank-deficient X).

    Tries statsmodels ``method='pinv'`` first; on failure, fits on standardized
    non-constant columns and maps coefficients, SEs, and p-values back to the
    original scale (equivalent reparameterisation).
    """
    exog_df = exog if isinstance(exog, pd.DataFrame) else pd.DataFrame(exog)
    endog_s = pd.Series(np.asarray(endog, dtype=float).ravel())

    for cov_type in ("nonrobust", "HC0"):
        for method in ("pinv", "qr"):
            try:
                return sm.OLS(endog_s, exog_df).fit(method=method, cov_type=cov_type)
            except Exception:
                continue

    X = np.asarray(exog_df, dtype=float)
    n, k = X.shape
    if k <= 1:
        return sm.OLS(endog_s, exog_df).fit(method="pinv")

    names = list(exog_df.columns)
    has_const = np.allclose(X[:, 0], 1.0, rtol=0, atol=1e-10)

    if not has_const:
        return sm.OLS(endog_s, exog_df).fit(method="pinv")

    rest = X[:, 1:].copy()
    mu = rest.mean(axis=0)
    sig = rest.std(axis=0, ddof=0)
    sig = np.where(sig < 1e-12, 1.0, sig)
    Z = (rest - mu) / sig
    Xs = np.column_stack([X[:, :1], Z])
    exog_sdf = pd.DataFrame(Xs, columns=names, index=exog_df.index)
    res = sm.OLS(endog_s, exog_sdf).fit(method="pinv")
    gamma = res.params.to_numpy(dtype=float)
    cov_g = np.asarray(res.cov_params(), dtype=float)
    p = k
    J = np.zeros((p, p))
    J[0, 0] = 1.0
    for m in range(p - 1):
        J[0, m + 1] = -mu[m] / sig[m]
        J[m + 1, m + 1] = 1.0 / sig[m]
    beta = J @ gamma
    cov_b = J @ cov_g @ J.T
    se = np.sqrt(np.maximum(np.diag(cov_b), 0.0))
    df_resid = float(res.df_resid) if hasattr(res, "df_resid") else max(n - k, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = np.where(se > 1e-15, beta / se, np.nan)
    pvals = 2.0 * (1.0 - student_t.cdf(np.abs(t_vals), df_resid))

    y_hat = (X @ beta).ravel()
    ssr = float(np.sum((endog_s.to_numpy() - y_hat) ** 2))
    y_dem = endog_s.to_numpy() - endog_s.to_numpy().mean()
    sst = float(np.sum(y_dem ** 2))
    r_squared = 1.0 - (ssr / sst) if sst > 1e-15 else np.nan

    class _OLSCompat:
        """Minimal surface matching ``statsmodels`` results used in this script."""

        def __init__(self, params, pvalues, rsquared):
            self.params = params
            self.pvalues = pvalues
            self.rsquared = rsquared

    params_sr = pd.Series(beta, index=names)
    pval_sr = pd.Series(pvals, index=names)
    return _OLSCompat(params_sr, pval_sr, r_squared)


# --- CHANGE 1: helper for bootstrap-based p-values ---
def _bootstrap_pvalue(main_estimate, bootstrap_samples):
    """Compute a two-sided bootstrap p-value from a bootstrap distribution.

    CHANGE 1: This replaces the OLS-derived p-values for nonlinear models
    (GBM, RF, Lasso) where sm.OLS p-values are not meaningful.

    Method: count the proportion of bootstrap samples on the opposite side of
    zero from the main estimate, then double for a two-sided test. This is an
    approximate method consistent with the bootstrap CI philosophy used elsewhere
    in the pipeline.

    Returns np.nan if bootstrap_samples is empty.
    """
    if not bootstrap_samples:
        return np.nan
    samples = np.array(bootstrap_samples)
    if main_estimate >= 0:
        p_one_sided = np.mean(samples <= 0)
    else:
        p_one_sided = np.mean(samples >= 0)
    return float(np.clip(2 * p_one_sided, 0.0, 1.0))
# --- END CHANGE 1: _bootstrap_pvalue ---


# --- CHANGE 1: fmt_pval helper for output tables ---
def _fmt_pval(p):
    """Format a p-value for display; shows 'n/a' for NaN (nonlinear models)."""
    if pd.isna(p):
        return "    n/a   "
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{p:8.2e}{sig:<3}"
# --- END CHANGE 1: _fmt_pval ---


# --- CHANGE 1: model_config parameter added to all estimation functions ---
# Default is MODEL_CONFIGS['ols_baseline'] so the function signatures are
# backward-compatible; process_file() explicitly passes each config.

def calculate_mediation_effects_dowhy(data, model_config=None):
    """Calculate mediation effects using DoWhy framework with Baron & Kenny approach.

    CHANGE 1: Added model_config parameter to select the causal estimator.
    The DAG structure and Baron & Kenny mediation logic (a*b product-of-coefficients)
    are completely unchanged — only the DoWhy method_name and method_params swap.
    """
    # CHANGE 1: default to OLS baseline when called without a config
    if model_config is None:
        model_config = MODEL_CONFIGS["ols_baseline"]

    data = data.copy().reset_index(drop=True)

    pm = P_MISSING_COL
    G = nx.DiGraph([
        ('G1', 'W'), ('G1', 'P'), ('G1', 'S'), ('G1', pm),
        ('G2', 'W'), ('G2', 'P'), ('G2', 'S'), ('G2', pm),
        ('W', 'P'), ('W', 'S'), ('W', pm),
        ('P', 'S'),
        (pm, 'S'),
    ])

    # CHANGE 1: pull estimator settings out of the config dict.
    # Previously these were hardcoded as method_name="backdoor.linear_regression".
    estimator_method = model_config["method_name"]
    # deepcopy so sklearn model instances inside method_params are fresh each call
    estimator_params = copy.deepcopy(model_config["method_params"])
    uses_ols_pvalues = model_config["uses_ols_pvalues"]

    # -------------------------------------------------------------------------
    # Step 1: Path a (W → P)
    # -------------------------------------------------------------------------
    try:
        if uses_ols_pvalues:
            # OLS baseline: use DoWhy + statsmodels for p-value
            dag_dot = nx.nx_pydot.to_pydot(G).to_string()
            model_a_dowhy = CausalModel(data, treatment='W', outcome='P', graph=dag_dot)
            estimand_a = model_a_dowhy.identify_effect()
            estimate_a = model_a_dowhy.estimate_effect(
                estimand_a,
                method_name=estimator_method,
                method_params=estimator_params,
                control_value=0, treatment_value=1,
            )
            a_coef_dowhy = estimate_a.value
            X_a = sm.add_constant(data[['W', 'G1', 'G2', P_MISSING_COL]])
            model_a_sm = fit_ols_stable(data['P'], X_a)
            a_pval = model_a_sm.pvalues['W']
        else:
            # Non-OLS: call EconML LinearDML directly — bypasses DoWhy's broken bridge
            init = estimator_params.get('init_params', {})
            a_coef_dowhy = _econml_ate(
                Y=data['P'].values,
                T=data['W'].values,
                W_confounders=data[['G1', 'G2', P_MISSING_COL]].values,
                model_y=init['model_y'],
                model_t=init['model_t'],
            )
            a_pval = np.nan  # bootstrap p-value computed in process_file()
    except Exception as e:
        print(f"Warning: Path a estimation failed: {e}")
        model_a_sm = fit_ols_stable(data['P'], sm.add_constant(data[['W', 'G1', 'G2', P_MISSING_COL]]))
        a_coef_dowhy = model_a_sm.params['W']
        a_pval = model_a_sm.pvalues['W']

    # -------------------------------------------------------------------------
    # Step 2: Path b (P → S | W)
    # -------------------------------------------------------------------------
    try:
        if uses_ols_pvalues:
            dag_dot = nx.nx_pydot.to_pydot(G).to_string()
            model_b_dowhy = CausalModel(data, treatment='P', outcome='S', graph=dag_dot)
            estimand_b = model_b_dowhy.identify_effect()
            estimate_b = model_b_dowhy.estimate_effect(
                estimand_b,
                method_name=estimator_method,
                method_params=copy.deepcopy(estimator_params),
                control_value=0, treatment_value=1,
            )
            b_coef_dowhy = estimate_b.value
            X_b = sm.add_constant(data[['P', 'W', 'G1', 'G2', P_MISSING_COL]])
            model_b_sm = fit_ols_stable(data['S'], X_b)
            b_pval = model_b_sm.pvalues['P']
        else:
            # P → S controlling for W + confounders
            init = estimator_params.get('init_params', {})
            b_coef_dowhy = _econml_ate(
                Y=data['S'].values,
                T=data['P'].values,
                W_confounders=data[['W', 'G1', 'G2', P_MISSING_COL]].values,
                model_y=init['model_y'],
                model_t=init['model_t'],
            )
            b_pval = np.nan
    except Exception as e:
        print(f"Warning: Path b estimation failed: {e}")
        model_b_sm = fit_ols_stable(data['S'], sm.add_constant(data[['P', 'W', 'G1', 'G2', P_MISSING_COL]]))
        b_coef_dowhy = model_b_sm.params['P']
        b_pval = model_b_sm.pvalues['P']

    # -------------------------------------------------------------------------
    # Step 3: Total effect (W → S)
    # -------------------------------------------------------------------------
    try:
        if uses_ols_pvalues:
            dag_dot = nx.nx_pydot.to_pydot(G).to_string()
            model_c_dowhy = CausalModel(data, treatment='W', outcome='S', graph=dag_dot)
            estimand_c = model_c_dowhy.identify_effect()
            estimate_c = model_c_dowhy.estimate_effect(
                estimand_c,
                method_name=estimator_method,
                method_params=copy.deepcopy(estimator_params),
                control_value=0, treatment_value=1,
            )
            c_coef_dowhy = estimate_c.value
            X_c = sm.add_constant(data[['W', 'G1', 'G2', P_MISSING_COL]])
            model_c_sm = fit_ols_stable(data['S'], X_c)
            c_pval = model_c_sm.pvalues['W']
            model_c_r2 = model_c_sm.rsquared
        else:
            init = estimator_params.get('init_params', {})
            c_coef_dowhy = _econml_ate(
                Y=data['S'].values,
                T=data['W'].values,
                W_confounders=data[['G1', 'G2', P_MISSING_COL]].values,
                model_y=init['model_y'],
                model_t=init['model_t'],
            )
            c_pval = np.nan
            model_c_r2 = np.nan
    except Exception as e:
        print(f"Warning: Total effect estimation failed: {e}")
        model_c_sm = fit_ols_stable(data['S'], sm.add_constant(data[['W', 'G1', 'G2', P_MISSING_COL]]))
        c_coef_dowhy = model_c_sm.params['W']
        c_pval = model_c_sm.pvalues['W']
        model_c_r2 = model_c_sm.rsquared

    # -------------------------------------------------------------------------
    # Step 4: Direct effect (W → S | P)
    #
    # NOTE (CHANGE 1): This step uses sm.OLS for ALL model configs, including
    # GBM, RF, and Lasso. The Baron & Kenny c' path is inherently defined as
    # a partial regression coefficient (W's coefficient when P is included).
    # Extending this to a nonlinear estimator requires a full counterfactual
    # mediation framework (Imai et al. 2010) which is out of scope for Change 1.
    # OLS here gives a conservative, cross-model-comparable c' estimate.
    # -------------------------------------------------------------------------
    model_direct = fit_ols_stable(data['S'], sm.add_constant(data[['P', 'W', 'G1', 'G2', P_MISSING_COL]]))
    c_prime_coef = model_direct.params['W']
    c_prime_pval = model_direct.pvalues['W']

    return {
        'total_effect': c_coef_dowhy,
        'total_pval': c_pval,
        'direct_effect': c_prime_coef,
        'direct_pval': c_prime_pval,
        'indirect_effect': a_coef_dowhy * b_coef_dowhy,
        'path_a_coef': a_coef_dowhy,
        'path_a_pval': a_pval,
        'path_b_coef': b_coef_dowhy,
        'path_b_pval': b_pval,
        'model_c_r2': model_c_r2,
        'dowhy_estimates': {
            'path_a': estimate_a if 'estimate_a' in locals() else None,
            'path_b': estimate_b if 'estimate_b' in locals() else None,
            'total': estimate_c if 'estimate_c' in locals() else None
        }
    }


def bootstrap_mediation_effects_dowhy(data, n_bootstrap=100, model_config=None):
    """Calculate bootstrap confidence intervals for DoWhy mediation effects.

    CHANGE 1: Added model_config parameter so each bootstrap iteration uses
    the same estimator as the main analysis. Also returns raw bootstrap
    distributions (total, direct, indirect, path_a, path_b) so that
    process_file() can compute bootstrap-based p-values for nonlinear models.
    """
    # CHANGE 1: default to OLS baseline when called without a config
    if model_config is None:
        model_config = MODEL_CONFIGS["ols_baseline"]

    total_effects = []
    direct_effects = []
    indirect_effects = []
    # CHANGE 1: added path_a and path_b bootstrap samples for bootstrap p-values
    path_a_effects = []
    path_b_effects = []

    for i in range(n_bootstrap):
        boot_data = resample(data, random_state=i, replace=True).reset_index(drop=True)

        try:
            # CHANGE 1: pass model_config through to the estimation function.
            # Original line was: boot_effects = calculate_mediation_effects_dowhy(boot_data)
            boot_effects = calculate_mediation_effects_dowhy(boot_data, model_config=model_config)
            total_effects.append(boot_effects['total_effect'])
            direct_effects.append(boot_effects['direct_effect'])
            indirect_effects.append(boot_effects['indirect_effect'])
            # CHANGE 1: collect path coefficients for bootstrap p-value computation
            path_a_effects.append(boot_effects['path_a_coef'])
            path_b_effects.append(boot_effects['path_b_coef'])
        except Exception as e:
            print(f"Bootstrap iteration {i} failed: {e}")
            continue

    if len(total_effects) < n_bootstrap * 0.8:
        print(f"Warning: Only {len(total_effects)}/{n_bootstrap} bootstrap iterations succeeded")

    total_ci = np.percentile(total_effects, [2.5, 97.5]) if total_effects else [np.nan, np.nan]
    direct_ci = np.percentile(direct_effects, [2.5, 97.5]) if direct_effects else [np.nan, np.nan]
    indirect_ci = np.percentile(indirect_effects, [2.5, 97.5]) if indirect_effects else [np.nan, np.nan]

    return {
        'total_ci': total_ci,
        'direct_ci': direct_ci,
        'indirect_ci': indirect_ci,
        'bootstrap_success_rate': len(total_effects) / n_bootstrap,
        # CHANGE 1: raw bootstrap distributions for bootstrap p-value computation
        # in process_file() (only needed for non-OLS models).
        'total_effects_raw': total_effects,
        'direct_effects_raw': direct_effects,
        'indirect_effects_raw': indirect_effects,
        'path_a_effects_raw': path_a_effects,
        'path_b_effects_raw': path_b_effects,
    }


def run_dowhy_refutations(data, model_config=None):
    """Run DoWhy refutation tests.

    CHANGE 1: Added model_config parameter so refutations use the same estimator
    as the main analysis. Previously the estimator was hardcoded to
    backdoor.linear_regression, meaning refutations were always testing OLS
    even when the main model was GBM.
    """
    # CHANGE 1: default to OLS baseline when called without a config
    if model_config is None:
        model_config = MODEL_CONFIGS["ols_baseline"]

    try:
        uses_ols = model_config["uses_ols_pvalues"]

        if uses_ols:
            # OLS baseline: use DoWhy's built-in refutation framework
            pm = P_MISSING_COL
            G = nx.DiGraph([
                ('G1', 'W'), ('G1', 'P'), ('G1', 'S'), ('G1', pm),
                ('G2', 'W'), ('G2', 'P'), ('G2', 'S'), ('G2', pm),
                ('W', 'P'), ('P', 'S'), ('W', 'S'), ('W', pm),
                (pm, 'S'),
            ])
            dag_dot = nx.nx_pydot.to_pydot(G).to_string()
            model = CausalModel(data, treatment='W', outcome='S', graph=dag_dot)
            estimand = model.identify_effect()
            estimate = model.estimate_effect(
                estimand,
                method_name=model_config["method_name"],
                method_params=copy.deepcopy(model_config["method_params"]),
                control_value=0, treatment_value=1,
            )
            placebo_refute = model.refute_estimate(
                estimand, estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute"
            )
            subset_refute = model.refute_estimate(
                estimand, estimate,
                method_name="data_subset_refuter",
                subset_fraction=0.8
            )
            return {
                'placebo_effect': placebo_refute.new_effect,
                'subset_effect': subset_refute.new_effect,
                'original_effect': estimate.value,
            }
        else:
            # Non-OLS: run refutations manually via _econml_ate
            init = copy.deepcopy(model_config["method_params"].get("init_params", {}))
            conf_cols = ['G1', 'G2', P_MISSING_COL]

            original = _econml_ate(
                data['S'].values, data['W'].values,
                data[conf_cols].values, init['model_y'], init['model_t'],
            )

            # Placebo: permute treatment
            placebo_T = np.random.permutation(data['W'].values)
            placebo_effect = _econml_ate(
                data['S'].values, placebo_T,
                data[conf_cols].values, init['model_y'], init['model_t'],
            )

            # Subset: random 80% of data
            idx = np.random.choice(len(data), size=int(0.8 * len(data)), replace=False)
            sub = data.iloc[idx]
            subset_effect = _econml_ate(
                sub['S'].values, sub['W'].values,
                sub[conf_cols].values, init['model_y'], init['model_t'],
            )

            return {
                'placebo_effect': placebo_effect,
                'subset_effect': subset_effect,
                'original_effect': original,
            }

    except Exception as e:
        print(f"Warning: Refutations failed: {e}")
        return {
            'placebo_effect': np.nan,
            'subset_effect': np.nan,
            'original_effect': np.nan,
        }


def calculate_predictive_metrics(data, model_config=None):
    """Calculate predictive evaluation metrics for Change 2.

    Metrics added:
    - predictive_r2   : retained for backward compatibility
    - rmse            : retained from prior version
    - mae             : mean absolute error in magnitude units
    - skill_score     : 1 - MSE_model / MSE_baseline
    - spearman_rho    : rank correlation, robust to skew
    """

    if model_config is None:
        model_config = MODEL_CONFIGS["ols_baseline"]

    try:
        X = data[['W', 'P', 'G1', 'G2', P_MISSING_COL]].copy()
        # --- CHANGE 3: removed local W-only transform that was here ---
        # Original line was:
        #   X['W'] = np.log1p(X['W'])  # keep existing behavior for now
        # W and P are now both log-transformed upstream in process_file() before
        # this function is called. The local transform is removed to avoid
        # double-transforming W.
        # --- END CHANGE 3 ---
        y = data['S']

        x_tr, x_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, shuffle=True
        )

        predictive_model = copy.deepcopy(model_config["predictive_model"])
        pred = predictive_model.fit(x_tr, y_tr).predict(x_te)

        # Existing metrics
        mse = mean_squared_error(y_te, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_te, pred)

        # New Change 2 metrics
        mae = mean_absolute_error(y_te, pred)

        # Baseline = predict training-set mean on all test points
        baseline_pred = np.full(shape=len(y_te), fill_value=y_tr.mean(), dtype=float)
        baseline_mse = mean_squared_error(y_te, baseline_pred)

        # Guard against divide-by-zero if baseline_mse is zero
        if baseline_mse == 0:
            skill_score = np.nan
        else:
            skill_score = 1 - (mse / baseline_mse)

        # Spearman rank correlation
        spearman_rho, spearman_p = spearmanr(y_te, pred)

        return {
            'predictive_r2': r2,
            'rmse': rmse,
            'mae': mae,
            'skill_score': skill_score,
            'spearman_rho': spearman_rho,
            'spearman_pval': spearman_p,
            'mse': mse,
            'baseline_mse': baseline_mse,
        }

    except Exception as e:
        print(f"Warning: Predictive metric calculation failed: {e}")
        return {
            'predictive_r2': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'skill_score': np.nan,
            'spearman_rho': np.nan,
            'spearman_pval': np.nan,
            'mse': np.nan,
            'baseline_mse': np.nan,
        }


# =============================================================================
# CHANGE 7: Hurdle model — two-stage framework separating occurrence from magnitude
# =============================================================================

def fit_hurdle_model(data):
    """CHANGE 7: Two-stage hurdle model separating earthquake occurrence from magnitude.

    Physical motivation: injection triggering is a two-stage process.
      Stage 1 — does injection trigger an event at all? (binary)
      Stage 2 — conditional on triggering, how large is the event? (continuous)
    Collapsing these into one regression blends triggering probability and magnitude
    scaling into the same parameter, reducing interpretability.

    Per Chris Reudelhueber's feedback, formation depth (shallow vs. Ellenburger-and-
    below) is a known missing confounder for Stage 1: shallow-zone wells are
    physically disconnected from seismogenic faults. Stage 1 partially captures
    this through correlation with G1/G2 but cannot separate it cleanly without
    disposal-interval data.

    Stage 1: GBM binary classifier predicts P(S > 0) using W, P, G1, G2, P_missing.
    Stage 2: GBM regressor predicts magnitude conditional on S > 0 (event rows only).

    Returns per-radius summary scalars for dashboard output:
      hurdle_occurrence_prob       mean P(occurrence) on the held-out test set
      hurdle_conditional_magnitude mean predicted magnitude on event test rows
      hurdle_stage1_auc            ROC-AUC for the Stage 1 classifier
      hurdle_stage2_mae            MAE in magnitude units for Stage 2
    """
    _nan_result = {
        'hurdle_occurrence_prob': np.nan,
        'hurdle_conditional_magnitude': np.nan,
        'hurdle_stage1_auc': np.nan,
        'hurdle_stage2_mae': np.nan,
    }

    try:
        features = ['W', 'P', 'G1', 'G2', P_MISSING_COL]
        X = data[features].copy()
        S_binary = (data['S'] > 0).astype(int)
        S_magnitude = data['S'].copy()

        if S_binary.nunique() < 2:
            print("  ⚠️  Hurdle Stage 1: only one class in S_binary — skipping.")
            return _nan_result

        x_tr, x_te, yb_tr, yb_te, ym_tr, ym_te = train_test_split(
            X, S_binary, S_magnitude, test_size=0.20, random_state=42, shuffle=True
        )

        # --- Stage 1: binary occurrence classifier ---
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf.fit(x_tr, yb_tr)
        occ_prob = clf.predict_proba(x_te)[:, 1]
        stage1_auc = float(roc_auc_score(yb_te, occ_prob)) if yb_te.nunique() > 1 else np.nan

        # --- Stage 2: conditional magnitude regression (events only) ---
        event_idx_tr = yb_tr[yb_tr == 1].index
        event_idx_te = yb_te[yb_te == 1].index

        if len(event_idx_tr) < 10:
            print(f"  ⚠️  Hurdle Stage 2: only {len(event_idx_tr)} event rows in train — skipping Stage 2.")
            return {
                'hurdle_occurrence_prob': float(occ_prob.mean()),
                'hurdle_conditional_magnitude': np.nan,
                'hurdle_stage1_auc': stage1_auc,
                'hurdle_stage2_mae': np.nan,
            }

        reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        reg.fit(x_tr.loc[event_idx_tr], ym_tr.loc[event_idx_tr])

        if len(event_idx_te) > 0:
            mag_pred = reg.predict(x_te.loc[event_idx_te])
            stage2_mae = float(mean_absolute_error(ym_te.loc[event_idx_te], mag_pred))
            cond_mag = float(mag_pred.mean())
        else:
            stage2_mae = np.nan
            cond_mag = np.nan

        cond_str = f"{cond_mag:.3f}" if cond_mag is not None and not np.isnan(cond_mag) else "n/a"
        mae_str = f"{stage2_mae:.3f}" if stage2_mae is not None and not np.isnan(stage2_mae) else "n/a"
        print(f"  ✅ Hurdle model: P(occur)={float(occ_prob.mean()):.3f}, "
              f"AUC={stage1_auc:.3f}, E[mag|occur]={cond_str}, Stage2 MAE={mae_str}")

        return {
            'hurdle_occurrence_prob': float(occ_prob.mean()),
            'hurdle_conditional_magnitude': cond_mag,
            'hurdle_stage1_auc': stage1_auc,
            'hurdle_stage2_mae': stage2_mae,
        }

    except Exception as e:
        print(f"  Warning: Hurdle model failed: {e}")
        return _nan_result

# =============================================================================
# END CHANGE 7
# =============================================================================


def process_file(csv_file):
    """Process a single radius file across all model configs.

    CHANGE 1: Previously returned a single dict (one OLS result per file).
    Now returns a list of dicts — one per MODEL_CONFIGS entry — each with a
    'model_name' key. The main loop uses results.extend() to collect them.
    """
    print(f"\n{'=' * 60}")
    print(f"PROCESSING: {csv_file}")
    print(f"{'=' * 60}")

    radius = extract_radius(csv_file)

    try:
        # Load the data (unchanged)
        df_raw = pd.read_csv(csv_file, low_memory=False)
        print(f"✅ Loaded {len(df_raw):,} rows")

        # Auto-detect columns
        # W: target "Vol Prev N (BBLs)" (lookback-window sum), NOT "Volume Injected (BBLs)"
        # (same-day). Both exist in the faults files; "prev" + "bbl" uniquely identifies
        # the lookback column. Using same-day volume would make the lookback window irrelevant.
        COL_FRAGS = {
            "W": ["prev", "bbl"],
            "P": ["pressure", "psig"],
            "S": ["local", "mag"],
            "G1": ["nearest", "fault", "dist"],
            "G2": ["fault", "segment"],
        }

        found = {}
        for k, frags in COL_FRAGS.items():
            col = find_column(frags, df_raw.columns)
            if col:
                found[k] = col

        missing = [k for k in COL_FRAGS.keys() if k not in found]
        if missing:
            print(f"❌ Missing columns: {missing}")
            return []  # CHANGE 1: return empty list instead of None

        rename_map = {found[k]: safe(k) for k in found}
        data = df_raw[list(found.values())].rename(columns=rename_map)

        p_col = safe("P")
        data[P_MISSING_COL] = data[p_col].isna().astype(np.float64)
        data[p_col] = data[p_col].fillna(0.0)

        essential = [safe("W"), safe("S")]
        rows_before = len(data)
        data = data.dropna(subset=essential)
        rows_after = len(data)

        geo_cov = [safe(c) for c in ["G1", "G2"] if safe(c) in data.columns]
        if geo_cov:
            data[geo_cov] = data[geo_cov].fillna(data[geo_cov].median(numeric_only=True))

        data = data.rename(columns={
            safe("W"): "W",
            safe("P"): "P",
            safe("S"): "S",
            safe("G1"): "G1",
            safe("G2"): "G2"
        })

        # --- CHANGE 3: Apply log-transforms to W and P upstream ---
        # Previously W was log-transformed only inside calculate_predictive_metrics(),
        # and P was never transformed. This meant the causal estimator and the
        # predictive model were working in different units.
        # Applying np.log1p() here ensures both W and P are on a log scale for
        # all downstream analysis: causal mediation (Steps 1-3) and predictive hold-out.
        # P is clipped at 0 before the transform: negative pressure values in the raw
        # data (gauge readings, idle wells) would produce -inf under log1p(x) when x < -1,
        # causing downstream inf/nan failures. Zero-clipping treats sub-zero readings
        # as effectively zero injection pressure, consistent with an idle-well interpretation.
        if APPLY_LOG_TRANSFORM:
            # log1p is undefined below -1; negative "volume" rows are data artifacts -> clip at 0
            data['W'] = np.log1p(data['W'].clip(lower=0))
            data['P'] = np.log1p(data['P'].clip(lower=0))
        # --- END CHANGE 3 ---

        finite_cov = [c for c in ("W", "P", "S", "G1", "G2", P_MISSING_COL) if c in data.columns]
        _ok = np.isfinite(data[finite_cov].to_numpy(dtype=float)).all(axis=1)
        if not _ok.all():
            n_bad = int((~_ok).sum())
            print(f"Warning: dropping {n_bad:,} rows with non-finite W/P/S/G1/G2/P_missing after transforms")
            data = data.loc[_ok]

        print(f"✅ Clean data: {len(data):,} rows (dropped {rows_before - rows_after:,})")
        print(f"   Rows with missing pressure (pre-zero-fill): {100.0 * data[P_MISSING_COL].mean():.2f}%")

        # VIF calculation depends only on data, not on the estimator — compute once
        X_a = sm.add_constant(data[['W', 'G1', 'G2', P_MISSING_COL]])
        vif_a_avg = calculate_vif(X_a)['VIF'].mean()

        X_b = sm.add_constant(data[['W', 'P', 'G1', 'G2', P_MISSING_COL]])
        vif_b_avg = calculate_vif(X_b)['VIF'].mean()

        # --- CHANGE 7: Fit hurdle model once per file (not per estimator) ---
        print(f"\n  🔄 Hurdle model (Change 7)...")
        hurdle = fit_hurdle_model(data)
        # --- END CHANGE 7 ---

        # =====================================================================
        # CHANGE 1: Loop over all model configs.
        # Previously this block ran a single OLS analysis. Now each model in
        # MODEL_CONFIGS is run in turn and its result appended to file_results.
        # =====================================================================
        file_results = []

        for model_name, model_config in MODEL_CONFIGS.items():
            print(f"\n  ── Model: {model_name} ──")

            try:
                # Causal mediation analysis
                print(f"  🔄 Mediation analysis [{model_name}]...")
                effects = calculate_mediation_effects_dowhy(data, model_config=model_config)

                # Bootstrap confidence intervals
                print(f"  🔄 Bootstrap CI [{model_name}]...")
                ci_results = bootstrap_mediation_effects_dowhy(
                    data, n_bootstrap=50, model_config=model_config
                )

                # CHANGE 1: For nonlinear models, derive p-values from the bootstrap
                # distributions because OLS p-values are not valid for GBM/RF/Lasso.
                # For OLS, use the p-values already returned by calculate_mediation_effects_dowhy().
                total_pval = effects['total_pval']
                path_a_pval = effects['path_a_pval']
                path_b_pval = effects['path_b_pval']

                if not model_config["uses_ols_pvalues"]:
                    total_pval = _bootstrap_pvalue(
                        effects['total_effect'], ci_results['total_effects_raw']
                    )
                    path_a_pval = _bootstrap_pvalue(
                        effects['path_a_coef'], ci_results['path_a_effects_raw']
                    )
                    path_b_pval = _bootstrap_pvalue(
                        effects['path_b_coef'], ci_results['path_b_effects_raw']
                    )
                # END CHANGE 1 (bootstrap p-values)

                # Refutation tests
                print(f"  🔄 Refutation tests [{model_name}]...")
                refutations = run_dowhy_refutations(data, model_config=model_config)

                # Predictive metrics
                print(f"  🔄 Predictive metrics [{model_name}]...")
                pred_metrics = calculate_predictive_metrics(data, model_config=model_config)

                # Proportion mediated (unchanged formula)
                prop_mediated = (
                    (effects['indirect_effect'] / effects['total_effect']) * 100
                    if effects['total_effect'] != 0 else 0
                )

                print(f"  ✅ {model_name} complete")
                print(f"     Total effect:   {effects['total_effect']:+.3e}  (p={total_pval:.3e})")
                print(f"     Direct effect:  {effects['direct_effect']:+.3e}  (p={effects['direct_pval']:.3e})")
                print(f"     Indirect:       {effects['indirect_effect']:+.3e}")
                print(f"     % Mediated:     {prop_mediated:.1f}%")
                print(f"     Bootstrap rate: {ci_results['bootstrap_success_rate']:.1%}")
                print(f"     Pred R²:        {pred_metrics['predictive_r2']:.3f}")
                print(f"     MAE:            {pred_metrics['mae']:.3f}")
                print(f"     Skill Score:    {pred_metrics['skill_score']:.3f}")
                print(f"     Spearman rho:   {pred_metrics['spearman_rho']:.3f}")

                file_results.append({
                    # CHANGE 1: model_name column added to distinguish estimator variants
                    'model_name': model_name,
                    'radius': radius,
                    'filename': csv_file,
                    'n_rows': len(data),
                    'total_effect': effects['total_effect'],
                    'total_pval': total_pval,
                    'total_ci_low': ci_results['total_ci'][0],
                    'total_ci_high': ci_results['total_ci'][1],
                    'direct_effect': effects['direct_effect'],
                    'direct_pval': effects['direct_pval'],
                    'direct_ci_low': ci_results['direct_ci'][0],
                    'direct_ci_high': ci_results['direct_ci'][1],
                    'indirect_effect': effects['indirect_effect'],
                    'indirect_ci_low': ci_results['indirect_ci'][0],
                    'indirect_ci_high': ci_results['indirect_ci'][1],
                    'prop_mediated': prop_mediated,
                    'path_a_coef': effects['path_a_coef'],
                    'path_a_pval': path_a_pval,
                    'path_b_coef': effects['path_b_coef'],
                    'path_b_pval': path_b_pval,
                    'causal_r2': effects['model_c_r2'],
                    'predictive_r2': pred_metrics['predictive_r2'],
                    'rmse': pred_metrics['rmse'],
                    'mae': pred_metrics['mae'],
                    'skill_score': pred_metrics['skill_score'],
                    'spearman_rho': pred_metrics['spearman_rho'],
                    'spearman_pval': pred_metrics['spearman_pval'],
                    'mse': pred_metrics['mse'],
                    'baseline_mse': pred_metrics['baseline_mse'],
                    'vif_a_avg': vif_a_avg,
                    'vif_b_avg': vif_b_avg,
                    'bootstrap_success_rate': ci_results['bootstrap_success_rate'],
                    'placebo_effect': refutations['placebo_effect'],
                    'subset_effect': refutations['subset_effect'],
                    'dowhy_original_effect': refutations['original_effect'],
                    'pressure_missing_pct': 100.0 * float(data[P_MISSING_COL].mean()),
                    # CHANGE 7: hurdle model outputs (same for all models at this radius)
                    'hurdle_occurrence_prob': hurdle['hurdle_occurrence_prob'],
                    'hurdle_conditional_magnitude': hurdle['hurdle_conditional_magnitude'],
                    'hurdle_stage1_auc': hurdle['hurdle_stage1_auc'],
                    'hurdle_stage2_mae': hurdle['hurdle_stage2_mae'],
                })

            except Exception as e:
                print(f"  ❌ Model {model_name} failed for {csv_file}: {e}")
                continue

        # =====================================================================
        # END CHANGE 1: per-model loop
        # =====================================================================

        return file_results

    except Exception as e:
        print(f"❌ Error loading {csv_file}: {str(e)}")
        return []  # CHANGE 1: return empty list instead of None




def _run_main():
    """Entry point; wrapped so ProcessPoolExecutor workers do not re-execute it."""
    # --- CHANGE 5: filter event files by lookback tag if --lookback-days was given ---
    csv_files = find_csv_files()
    if LOOKBACK_TAG is not None:
        tagged = [f for f in csv_files if 'event_well_links_with_faults' in f and LOOKBACK_TAG in f]
        if tagged:
            event_files = tagged
        else:
            # 30d is the original pipeline default — untagged files are the 30d dataset
            untagged = [f for f in csv_files if 'event_well_links_with_faults' in f
                        and not re.search(r'_\d+d_', f)]
            if untagged and LOOKBACK_DAYS_ARG == 30:
                print(f"ℹ️  No files tagged '{LOOKBACK_TAG}' found — using untagged files (original 30d dataset).")
                event_files = untagged
            else:
                event_files = []
    else:
        # Legacy behaviour: prefer un-tagged originals (historical filenames).
        event_files = [f for f in csv_files if 'event_well_links_with_faults' in f
                       and not re.search(r'_\d+d_', f)]
        if not event_files:
            # Current pipeline writes lookback-tagged files such as ..._30d_12km.csv
            event_files = [f for f in csv_files if 'event_well_links_with_faults' in f
                           and re.search(r'_\d+d_', f)]
    # --- END CHANGE 5 ---

    if not event_files:
        print("❌ No event files found!")
        exit(1)

    event_files.sort(key=lambda x: extract_radius(x))

    print(f"\n🎯 Found {len(event_files)} event files to process:")
    for i, file in enumerate(event_files):
        radius = extract_radius(file)
        size = os.path.getsize(file) / (1024 * 1024)
        print(f"  {i + 1:2d}. {radius:2d}km - {file} ({size:.1f} MB)")

    # CHANGE 1: report which models will be run before the main loop starts
    print(f"\n🧪 Models to compare: {list(MODEL_CONFIGS.keys())}")

    print(f"\n{'=' * 80}")
    print("BATCH DOWHY MEDIATION ANALYSIS WITH BOOTSTRAP CI")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    # CHANGE 1: process_file() now returns a list (one dict per model per file).
    # Use extend() instead of append() so all model results land in a flat list.

    # --- PARALLEL MIGRATION: applied by migrate_to_parallel.py ---
    # Distribute radii across worker processes. Each process_file() call is
    # fully independent (loads its own CSV, uses iteration-indexed bootstrap
    # seeds), so output is bit-identical to serial execution.
    import os as _os
    try:
        from joblib import Parallel, delayed  # type: ignore
        _have_joblib = True
    except ImportError:  # pragma: no cover - fallback if joblib missing
        _have_joblib = False

    _n_jobs_env = _os.environ.get("DOWHY_CI_JOBS", "").strip()
    try:
        _n_jobs = int(_n_jobs_env) if _n_jobs_env else 0
    except ValueError:
        _n_jobs = 0
    if _n_jobs <= 0:
        _cpu = _os.cpu_count() or 1
        _n_jobs = min(8, len(event_files), max(1, _cpu - 1))

    print(f"\n🔀  Parallel execution: {_n_jobs} worker(s) across {len(event_files)} radii "
          f"(override with DOWHY_CI_JOBS)")

    results = []
    if _have_joblib and _n_jobs > 1:
        _all = Parallel(n_jobs=_n_jobs, backend="loky", verbose=5)(
            delayed(process_file)(csv_file) for csv_file in event_files
        )
        for file_results in _all:
            if file_results:
                # dowhy_ci returns list; dowhy_ci_aggregated returns dict or None.
                if isinstance(file_results, list):
                    results.extend(file_results)
                else:
                    results.append(file_results)
    else:
        for i, csv_file in enumerate(event_files):
            print(f"\n[{i + 1}/{len(event_files)}] Processing {csv_file}...")
            file_results = process_file(csv_file)
            if file_results:
                if isinstance(file_results, list):
                    results.extend(file_results)
                else:
                    results.append(file_results)
    # =============================================================================
    # Output tables
    # =============================================================================

    if results:
        # CHANGE 1: sort by model_name first, then radius, so per-model tables are contiguous
        results_df = pd.DataFrame(results).sort_values(['model_name', 'radius'])

        print(f"\n{'=' * 140}")
        print("📊 COMPREHENSIVE DOWHY MEDIATION ANALYSIS SUMMARY")
        print(f"{'=' * 140}")

        # CHANGE 1: one causal-effects table and one significance table per model
        for model_name in MODEL_CONFIGS.keys():
            model_df = results_df[results_df['model_name'] == model_name]
            if model_df.empty:
                continue

            print(f"\n{'─' * 6} MODEL: {model_name.upper()} {'─' * 6}")

            print("\n🎯 CAUSAL EFFECTS BY RADIUS:")
            print("─" * 150)
            print(
                f"{'Radius':>6} {'N':>7} {'Total Effect':>13} {'95% CI':>20} "
                f"{'Direct Effect':>14} {'95% CI':>20} {'Indirect':>11} "
                f"{'% Med':>6} {'Causal R²':>9} {'Pred R²':>8} {'MAE':>8} {'SS':>8} {'ρ':>8}"
            )
            print("─" * 150)

            for _, row in model_df.iterrows():
                causal_r2_str = f"{row['causal_r2']:8.3f}" if pd.notna(row['causal_r2']) else "     n/a"
                pred_r2_str = f"{row['predictive_r2']:7.3f}" if pd.notna(row['predictive_r2']) else "   n/a"
                mae_str = f"{row['mae']:7.3f}" if pd.notna(row['mae']) else "   n/a"
                ss_str = f"{row['skill_score']:7.3f}" if pd.notna(row['skill_score']) else "   n/a"
                rho_str = f"{row['spearman_rho']:7.3f}" if pd.notna(row['spearman_rho']) else "   n/a"
                print(
                    f"{row['radius']:>4}km {row['n_rows']:>7,} "
                    f"{row['total_effect']:>+10.3e} "
                    f"[{row['total_ci_low']:>+7.2e},{row['total_ci_high']:>+7.2e}] "
                    f"{row['direct_effect']:>+11.3e} "
                    f"[{row['direct_ci_low']:>+7.2e},{row['direct_ci_high']:>+7.2e}] "
                    f"{row['indirect_effect']:>+8.2e} "
                    f"{row['prop_mediated']:>5.1f}% "
                    f"{causal_r2_str} "
                    f"{pred_r2_str} "
                    f"{mae_str} "
                    f"{ss_str} "
                    f"{rho_str}"
                )

            print("─" * 150)

            print(f"\n📈 STATISTICAL SIGNIFICANCE & QUALITY [{model_name}]:")
            print("─" * 120)
            print(
                f"{'Radius':>6} {'Total p-val':>12} {'Direct p-val':>13} "
                f"{'Path a p-val':>13} {'Path b p-val':>13} {'Bootstrap':>10} {'VIF':>6}"
            )
            print("─" * 120)

            for _, row in model_df.iterrows():
                print(
                    f"{row['radius']:>4}km "
                    f"{_fmt_pval(row['total_pval'])} "
                    f"{_fmt_pval(row['direct_pval'])} "
                    f"{_fmt_pval(row['path_a_pval'])} "
                    f"{_fmt_pval(row['path_b_pval'])} "
                    f"{row['bootstrap_success_rate']:>8.1%} "
                    f"{row['vif_b_avg']:>5.2f}"
                )

            print("─" * 120)
            print("Significance: *** p<0.001, ** p<0.01, * p<0.05  |  n/a = not applicable for this estimator")

        # CHANGE 1: Refutation table per model
        for model_name in MODEL_CONFIGS.keys():
            model_df = results_df[results_df['model_name'] == model_name]
            if model_df.empty:
                continue

            print(f"\n🔍 DOWHY REFUTATION TESTS [{model_name}]:")
            print("─" * 80)
            print(f"{'Radius':>6} {'Original':>12} {'Placebo':>12} {'Subset':>12} {'Status':>15}")
            print("─" * 80)

            for _, row in model_df.iterrows():
                placebo_ratio = (
                    abs(row['placebo_effect'] / row['total_effect'])
                    if row['total_effect'] != 0 else np.inf
                )
                subset_ratio = (
                    abs((row['subset_effect'] - row['total_effect']) / row['total_effect'])
                    if row['total_effect'] != 0 else np.inf
                )
                status = (
                    "✅ PASS" if placebo_ratio < 0.1 and subset_ratio < 0.2
                    else "⚠️ CAUTION" if placebo_ratio < 0.3
                    else "❌ FAIL"
                )
                print(
                    f"{row['radius']:>4}km "
                    f"{row['total_effect']:>+9.2e} "
                    f"{row['placebo_effect']:>+9.2e} "
                    f"{row['subset_effect']:>+9.2e} "
                    f"{status:>15}"
                )

            print("─" * 80)

        # CHANGE 1: Cross-model comparison table — total effect and predictive R² side by side
        # This is the primary output for deciding which estimator to adopt.
        print(f"\n🔬 CROSS-MODEL COMPARISON  (Total Effect | Predictive R²  by radius):")
        print("─" * 130)
        model_names = list(MODEL_CONFIGS.keys())
        header = f"{'Radius':>6}"
        for mn in model_names:
            short_name = mn[:12]
            header += f"  {short_name:>12} {'R²':>6} {'MAE':>6} {'SS':>6} {'ρ':>6}"
        print(header)
        print("─" * 130)

        for radius in sorted(results_df['radius'].unique()):
            line = f"{radius:>4}km"
            for mn in model_names:
                mask = (results_df['model_name'] == mn) & (results_df['radius'] == radius)
                if mask.any():
                    row = results_df[mask].iloc[0]
                    r2_str = f"{row['predictive_r2']:.3f}" if pd.notna(row['predictive_r2']) else "n/a"
                    mae_str = f"{row['mae']:.3f}" if pd.notna(row['mae']) else "n/a"
                    ss_str = f"{row['skill_score']:.3f}" if pd.notna(row['skill_score']) else "n/a"
                    rho_str = f"{row['spearman_rho']:.3f}" if pd.notna(row['spearman_rho']) else "n/a"
                    line += f"  {row['total_effect']:>+12.3e} {r2_str:>6} {mae_str:>6} {ss_str:>6} {rho_str:>6}"
                else:
                    line += f"  {'n/a':>12} {'n/a':>6} {'n/a':>6} {'n/a':>6} {'n/a':>6}"
            print(line)

        print("─" * 130)

        # Key insights using OLS baseline for backward-compatible headline numbers
        ols_df = results_df[results_df['model_name'] == 'ols_baseline']
        if not ols_df.empty:
            print("\n🔍 KEY INSIGHTS (OLS BASELINE):")
            print("─" * 60)
            strongest_total = ols_df.loc[ols_df['total_effect'].abs().idxmax()]
            highest_mediation = ols_df.loc[ols_df['prop_mediated'].idxmax()]
            best_pred_r2 = ols_df.loc[ols_df['predictive_r2'].idxmax()]
            best_mae = ols_df.loc[ols_df['mae'].idxmin()]
            best_ss = ols_df.loc[ols_df['skill_score'].idxmax()]
            best_rho = ols_df.loc[ols_df['spearman_rho'].idxmax()]

            print(f"• Strongest total effect:    {strongest_total['radius']}km ({strongest_total['total_effect']:+.3e})")
            print(f"• Highest mediation:         {highest_mediation['radius']}km ({highest_mediation['prop_mediated']:.1f}%)")
            print(f"• Best predictive fit:       {best_pred_r2['radius']}km (R²={best_pred_r2['predictive_r2']:.3f})")
            print(f"• Best predictive fit (R²):  {best_pred_r2['radius']}km (R²={best_pred_r2['predictive_r2']:.3f})")
            print(f"• Lowest MAE:                {best_mae['radius']}km (MAE={best_mae['mae']:.3f})")
            print(f"• Best Skill Score:          {best_ss['radius']}km (SS={best_ss['skill_score']:.3f})")
            print(f"• Best Spearman rho:         {best_rho['radius']}km (ρ={best_rho['spearman_rho']:.3f})")
            print(f"• Sample size range:         {ols_df['n_rows'].min():,} - {ols_df['n_rows'].max():,}")

            near_field = ols_df[ols_df['radius'] <= 5]
            far_field = ols_df[ols_df['radius'] >= 10]
            print(f"• Near-field mediation:      {near_field['prop_mediated'].mean():.1f}% (≤5km)")
            print(f"• Far-field mediation:       {far_field['prop_mediated'].mean():.1f}% (≥10km)")
            print(f"• Average bootstrap success: {ols_df['bootstrap_success_rate'].mean():.1%}")
            print(f"• Average VIF:               {ols_df['vif_b_avg'].mean():.2f}")
            print("─" * 60)

        # Save full multi-model results to CSV
        _tag_part = f"_{LOOKBACK_TAG}" if LOOKBACK_TAG else ""
        _log_part = "" if APPLY_LOG_TRANSFORM else "_nolog"
        output_file = f"dowhy_mediation_analysis{_tag_part}{_log_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        print(f"   Rows: {len(results_df)}  ({len(event_files)} radii × {len(MODEL_CONFIGS)} models)")
        print(f"   Columns: {list(results_df.columns)}")

    else:
        print("\n❌ No valid results obtained from any files.")

    print(f"\n{'=' * 80}")
    print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    _run_main()
