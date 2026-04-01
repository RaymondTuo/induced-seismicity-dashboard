"""
generate_report_change3.py
Project Geminae - Change 3 Brief
Black-and-white, Times New Roman, 1-2 pages. Matches Change 1/2 brief style.
Outputs both PDF and HTML.

Run AFTER dowhy_ci.py has produced a new dowhy_mediation_analysis_*.csv.
"""

from fpdf import FPDF
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob

# ── Load results ──────────────────────────────────────────────────────────────
candidates = sorted(
    glob.glob("dowhy_mediation_analysis_*.csv"),
    key=os.path.getmtime,
    reverse=True
)
if not candidates:
    raise FileNotFoundError(
        "No dowhy_mediation_analysis_*.csv found. "
        "Run dowhy_ci.py first, then re-run this script."
    )
LATEST_CSV = candidates[0]
print(f"Loading: {LATEST_CSV}")
df = pd.read_csv(LATEST_CSV)
ols = df[df['model_name'] == 'ols_baseline'].sort_values('radius').reset_index(drop=True)
gbm = df[df['model_name'] == 'gbm_nonparam'].sort_values('radius').reset_index(drop=True)

SUP2 = "\xb2"

def _e(v):
    return f"{v:.2e}" if pd.notna(v) else "n/a"

def _f(v, fmt=".3f"):
    return f"{v:{fmt}}" if pd.notna(v) else "n/a"

def _p(v):
    if pd.isna(v): return "n/a"
    if v < 0.001:  return f"{v:.1e}***"
    if v < 0.01:   return f"{v:.4f}**"
    if v < 0.05:   return f"{v:.4f}*"
    return f"{v:.4f}"


# =============================================================================
# PDF
# =============================================================================
class Brief(FPDF):

    def header(self):
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.6)
        self.line(10, 14, 200, 14)

    def footer(self):
        self.set_y(-12)
        self.set_font("Times", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(95, 5, "Project Geminae  |  Confidential", align="L")
        self.cell(95, 5, f"Page {self.page_no()}", align="R")
        self.set_text_color(0, 0, 0)

    def rule(self, weight=0.3):
        self.set_draw_color(0, 0, 0)
        self.set_line_width(weight)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def section(self, num, title):
        self.ln(4)
        self.set_font("Times", "B", 10)
        self.cell(0, 5, f"{num}.  {title}", ln=True)
        self.rule(0.3)

    def body(self, text, indent=0):
        self.set_font("Times", "", 9)
        self.set_x(10 + indent)
        self.multi_cell(190 - indent, 4.8, text)

    def bullet(self, text, indent=4):
        self.set_font("Times", "", 9)
        self.set_x(10 + indent)
        self.cell(4, 4.8, "-")
        self.multi_cell(186 - indent, 4.8, text)

    def th(self, cols, widths):
        self.set_font("Times", "B", 8)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.2)
        for col, w in zip(cols, widths):
            self.cell(w, 5.5, col, border="B", align="C")
        self.ln()

    def td(self, vals, widths):
        self.set_font("Times", "", 8)
        for v, w in zip(vals, widths):
            self.cell(w, 4.8, str(v), align="C")
        self.ln()

    def note(self, text):
        self.set_font("Times", "I", 8)
        self.set_x(12)
        self.multi_cell(186, 4.5, text)


pdf = Brief()
pdf.set_auto_page_break(auto=True, margin=14)
pdf.set_margins(10, 16, 10)
pdf.add_page()

# ── Heading ───────────────────────────────────────────────────────────────────
pdf.set_y(16)
pdf.set_font("Times", "B", 14)
pdf.cell(0, 7, "Project Geminae", ln=True)
pdf.set_font("Times", "", 10)
pdf.cell(130, 5, "Change 3 - Consistent Log-Transforms for W and P", ln=False)
pdf.set_font("Times", "I", 9)
pdf.cell(0, 5, datetime.now().strftime("%d %B %Y"), align="R", ln=True)
pdf.set_font("Times", "I", 9)
pdf.cell(0, 5,
    f"Source: {LATEST_CSV}  |  {len(ols)} radii completed",
    ln=True)
pdf.ln(1)
pdf.rule(0.6)

# ── 1. Change Summary ─────────────────────────────────────────────────────────
pdf.section("1", "Change Summary")
pdf.body(
    "Change 3 (ref. Feb 18 sponsor meeting) fixes a unit inconsistency in the "
    "data preparation step. Previously, injection volume W was log-transformed "
    "only inside the predictive hold-out function, and pressure P was never "
    "transformed. The causal estimator and the predictive model were therefore "
    "operating in different units. The fix applies np.log1p() to both W and P "
    "upstream in process_file() (dowhy_ci.py, after column rename), so all "
    "downstream analysis - causal mediation and predictive hold-out - uses "
    "consistently transformed inputs. Negative P values are clipped to zero "
    "before the transform (idle-well interpretation). All replaced code is "
    "retained in-file with --- CHANGE 3 --- markers."
)

# ── 2. Results - OLS Baseline ─────────────────────────────────────────────────
pdf.section("2", "Results - OLS Baseline (all " + str(len(ols)) + " radii)")
if not ols.empty:
    W = [14, 16, 28, 22, 18, 20, 20, 20, 22]
    pdf.th(["km", "N", "Total Effect", "p-value", "% Med", "Pred R"+SUP2, "MAE", "Skill", "Spear rho"], W)
    for _, r in ols.iterrows():
        pdf.td([
            f"{int(r['radius'])}",
            f"{int(r['n_rows']):,}",
            _e(r['total_effect']),
            _p(r['total_pval']),
            f"{r['prop_mediated']:.1f}%",
            _f(r.get('predictive_r2'), ".3f"),
            _f(r.get('mae'), ".3f"),
            _f(r.get('skill_score'), ".3f"),
            _f(r.get('spearman_rho'), ".3f"),
        ], W)
    pdf.note("*** p<0.001  ** p<0.01  * p<0.05")

# ── 3. Results - GBM NonParamDML ──────────────────────────────────────────────
pdf.section("3", "Results - GBM NonParamDML (all " + str(len(gbm)) + " radii)")
if not gbm.empty:
    W2 = [14, 16, 28, 22, 18, 20, 20, 20, 22]
    pdf.th(["km", "N", "Total Effect", "p-value", "% Med", "Pred R"+SUP2, "MAE", "Skill", "Spear rho"], W2)
    for _, r in gbm.iterrows():
        pdf.td([
            f"{int(r['radius'])}",
            f"{int(r['n_rows']):,}",
            _e(r['total_effect']),
            _p(r['total_pval']),
            f"{r['prop_mediated']:.1f}%",
            _f(r.get('predictive_r2'), ".3f"),
            _f(r.get('mae'), ".3f"),
            _f(r.get('skill_score'), ".3f"),
            _f(r.get('spearman_rho'), ".3f"),
        ], W2)
    pdf.note("*** p<0.001  ** p<0.01  * p<0.05")

# ── 4. Key Observations ───────────────────────────────────────────────────────
pdf.section("4", "Key Observations")
obs = []
if not ols.empty:
    sig = (ols['total_pval'] < 0.05).sum()
    te_range = f"{ols['total_effect'].min():.2e} to {ols['total_effect'].max():.2e}"
    med_mean = ols['prop_mediated'].mean()
    best_mae = ols.loc[ols['mae'].idxmin()] if ols['mae'].notna().any() else None
    best_ss  = ols.loc[ols['skill_score'].idxmax()] if ols['skill_score'].notna().any() else None
    obs.append(f"OLS: {sig}/{len(ols)} radii significant (p < 0.05).  Total effect range: {te_range}.")
    obs.append(f"OLS: Average pressure-mediated proportion: {med_mean:.1f}%.")
    if best_mae is not None:
        obs.append(f"OLS: Best MAE {best_mae['mae']:.3f} at {int(best_mae['radius'])} km; best Skill Score {best_ss['skill_score']:.3f} at {int(best_ss['radius'])} km.")
if not gbm.empty:
    gbm_r2_mean = gbm['predictive_r2'].mean()
    gbm_rho_mean = gbm['spearman_rho'].mean()
    ols_r2_str = f"{ols['predictive_r2'].mean():.3f}" if not ols.empty else "n/a"
    obs.append(f"GBM: Average predictive R{SUP2} {gbm_r2_mean:.3f} vs OLS {ols_r2_str}.  Average Spearman rho {gbm_rho_mean:.3f}.")
obs.append(
    "Causal coefficients are now in log-unit terms (effect per unit increase in "
    "log(1+W)).  Results are not directly comparable to Change 1/2 outputs on a "
    "coefficient-by-coefficient basis."
)
for o in obs:
    pdf.bullet(o)

# ── 5. Next Steps ─────────────────────────────────────────────────────────────
pdf.section("5", "Next Steps")
steps = [
    ("Sponsor Q1",   "Confirm log-linear assumption at field scale before production run."),
    ("Change 4",     "CausalForestDML in dowhy_ci_aggregated.py - ensure aggregated model also receives log-transformed inputs."),
    ("Change 6",     "Missing pressure fix (zero-fill + P_missing indicator) - implement after Change 3 to avoid conflicts on P."),
    ("Production run", "Revert test-mode settings (n_estimators=100, n_bootstrap=50) and re-run."),
]
for i, (step, desc) in enumerate(steps, 1):
    pdf.set_x(10)
    pdf.set_font("Times", "B", 9)
    pdf.cell(40, 5, f"  {i}.  {step}")
    pdf.set_font("Times", "", 9)
    pdf.multi_cell(0, 5, desc)

# ── Save PDF ──────────────────────────────────────────────────────────────────
pdf_out = f"ProjectGeminae_Change3_Brief_{datetime.now().strftime('%Y%m%d')}.pdf"
pdf.output(pdf_out)
print(f"PDF saved:  {os.path.abspath(pdf_out)}")


# =============================================================================
# HTML
# =============================================================================
def make_table(df_in):
    cols = ["km","N","Total Effect","p-value","% Med","Pred R²","MAE","Skill","Spear ρ"]
    keys = ["radius","n_rows","total_effect","total_pval","prop_mediated","predictive_r2","mae","skill_score","spearman_rho"]
    rows = ""
    for _, r in df_in.iterrows():
        def v(k, f):
            val = r.get(k)
            if f == "e":  return _e(val)
            if f == "p":  return _p(val)
            if f == "%":  return f"{val:.1f}%" if pd.notna(val) else "n/a"
            if f == ",":  return f"{int(val):,}" if pd.notna(val) else "n/a"
            if f == "i":  return str(int(val)) if pd.notna(val) else "n/a"
            return _f(val, f)
        fmts = ["i",",","e","p","%",".3f",".3f",".3f",".3f"]
        cells = "".join(f"<td>{v(k,f)}</td>" for k,f in zip(keys,fmts))
        rows += f"<tr>{cells}</tr>\n"
    header = "".join(f"<th>{c}</th>" for c in cols)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"

ols_table = make_table(ols)
gbm_table = make_table(gbm)

obs_html = "".join(f"<li>{o}</li>" for o in obs)

steps_html = "".join(
    f"<tr><td><strong>{s}</strong></td><td>{d}</td></tr>"
    for s, d in steps
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Project Geminae - Change 3 Brief</title>
<style>
  body {{ font-family: 'Times New Roman', Times, serif; font-size: 10pt;
         color: #000; background: #fff; max-width: 900px; margin: 40px auto; padding: 0 24px; }}
  h1   {{ font-size: 16pt; margin-bottom: 2px; }}
  .sub {{ font-size: 10pt; margin: 0; }}
  .meta {{ font-size: 9pt; font-style: italic; color: #444; margin-bottom: 8px; }}
  hr.thick {{ border: none; border-top: 2px solid #000; margin: 6px 0 10px; }}
  h2   {{ font-size: 11pt; margin: 16px 0 2px; border-bottom: 1px solid #000; padding-bottom: 2px; }}
  p, li {{ font-size: 9pt; line-height: 1.5; margin: 4px 0; }}
  ul   {{ margin: 4px 0 4px 16px; padding: 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 6px 0; }}
  th   {{ border-bottom: 1.5px solid #000; padding: 3px 6px; text-align: center; font-weight: bold; }}
  td   {{ padding: 2px 6px; text-align: center; }}
  .note {{ font-size: 8pt; font-style: italic; color: #444; }}
  .next td:first-child {{ font-weight: bold; width: 140px; text-align: left; }}
  .next td {{ text-align: left; padding: 2px 6px; font-size: 9pt; }}
</style>
</head>
<body>

<h1>Project Geminae</h1>
<p class="sub">Change 3 &mdash; Consistent Log-Transforms for W and P</p>
<p class="meta">Generated {datetime.now().strftime("%d %B %Y")} &nbsp;|&nbsp; Source: {LATEST_CSV} &nbsp;|&nbsp; {len(ols)} radii completed</p>
<hr class="thick">

<h2>1. Change Summary</h2>
<p>Change 3 (ref. Feb 18 sponsor meeting) fixes a unit inconsistency in the data preparation step.
Previously, injection volume W was log-transformed only inside the predictive hold-out function,
and pressure P was never transformed. The causal estimator and the predictive model were therefore
operating in different units. The fix applies <code>np.log1p()</code> to both W and P upstream in
<code>process_file()</code> (dowhy_ci.py, after column rename), so all downstream analysis uses
consistently transformed inputs. Negative P values are clipped to zero before the transform
(idle-well interpretation). All replaced code is retained in-file with <code>--- CHANGE 3 ---</code> markers.</p>

<h2>2. Results &mdash; OLS Baseline ({len(ols)} radii)</h2>
{ols_table}
<p class="note">*** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05</p>

<h2>3. Results &mdash; GBM NonParamDML ({len(gbm)} radii)</h2>
{gbm_table}
<p class="note">*** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05</p>

<h2>4. Key Observations</h2>
<ul>{obs_html}</ul>

<h2>5. Next Steps</h2>
<table class="next"><tbody>{steps_html}</tbody></table>

</body>
</html>
"""

html_out = f"ProjectGeminae_Change3_Brief_{datetime.now().strftime('%Y%m%d')}.html"
with open(html_out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"HTML saved: {os.path.abspath(html_out)}")
