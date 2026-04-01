"""
generate_sponsor_brief.py
Project Geminae - Change 1 Sponsor Brief
Clean, single-page, Times New Roman, minimal colour.
"""

from fpdf import FPDF
from datetime import datetime
import os

# Superscript-2 is Latin-1 0xB2 - supported by built-in Times font
SUP2 = "\xb2"


class Brief(FPDF):

    def header(self):
        # Single thin rule across top
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
        """Table header row."""
        self.set_font("Times", "B", 8.5)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.2)
        for col, w in zip(cols, widths):
            self.cell(w, 5.5, col, border="B", align="C")
        self.ln()

    def td(self, vals, widths, shade=False):
        """Table data row."""
        self.set_font("Times", "", 8.5)
        for v, w in zip(vals, widths):
            self.cell(w, 5, str(v), align="C")
        self.ln()

    def note(self, text):
        self.set_font("Times", "I", 8)
        self.set_x(12)
        self.multi_cell(186, 4.5, text)


# ─────────────────────────────────────────────────────────────────────────────
pdf = Brief()
pdf.set_auto_page_break(auto=False)
pdf.set_margins(10, 16, 10)
pdf.add_page()

# ── Document heading ──────────────────────────────────────────────────────────
pdf.set_y(16)
pdf.set_font("Times", "B", 14)
pdf.cell(0, 7, "Project Geminae", ln=True)

pdf.set_font("Times", "", 10)
pdf.cell(130, 5, "Change 1 - Pipeline Validation Brief", ln=False)
pdf.set_font("Times", "I", 9)
pdf.cell(0, 5, datetime.now().strftime("%d %B %Y"), align="R", ln=True)

pdf.set_font("Times", "I", 9)
pdf.cell(0, 5,
    "NonParamDML estimator implementation  |  "
    "Test-mode validation run  |  Full production run pending",
    ln=True)
pdf.ln(1)
pdf.rule(0.6)

# ── 1. Change Summary ─────────────────────────────────────────────────────────
pdf.section("1", "Change Summary")
pdf.body(
    "Change 1 (ref. Feb 18 sponsor meeting) replaces the Ordinary Least Squares (OLS) "
    "causal estimator in the Baron-Kenny mediation framework with a Non-parametric "
    "Double Machine Learning model (NonParamDML, EconML) using Gradient Boosted "
    "Regressors. The substitution applies to the three causal paths estimated in "
    "dowhy_ci.py: W -> P (injection volume to pressure), P -> S (pressure to "
    "seismicity), and W -> S (total effect). The causal DAG structure, OLS "
    "p-values, and all upstream pipeline steps (Steps 1-6) are unchanged. All "
    "replaced code is retained in-file as commented-out [Change 1 - OLD] blocks."
)

# ── 2. Validation Run Parameters ─────────────────────────────────────────────
pdf.section("2", "Validation Run Parameters")
pdf.body(
    "Parameters were reduced to confirm pipeline integrity before committing "
    "full compute. [TEST] tags mark values to be reverted for production.", indent=0
)
pdf.ln(2)

params = [
    ("GBM n_estimators (per sub-model)", "20  [TEST]", "100"),
    ("Bootstrap iterations",             "5   [TEST]", "50"),
    ("Spatial radii",                    "20 (1-20 km)", "20 (1-20 km)"),
    ("Runtime  Steps 7+8",              "~20 min",     "est. 2-3 hrs"),
]
pdf.th(["Parameter", "Test Value", "Production Value"], [90, 50, 50])
for i, row in enumerate(params):
    pdf.td(list(row), [90, 50, 50])
pdf.ln(1)

# ── 3. Pipeline Validation Status ────────────────────────────────────────────
pdf.section("3", "Pipeline Validation Status  (2026-02-27)")

steps = [
    ("Steps 1-4", "Data preparation and spatial join",         "PASS", "Output files nominal."),
    ("Step 5",    "Well-level causal analysis (OLS)",          "PASS", "~3 min."),
    ("Step 6",    "Event-level causal analysis (OLS)",         "PASS", "~38 sec."),
    ("Step 7",    "Well-level CI + NonParamDML  (dowhy_ci.py)","PASS", "20/20 radii.  100% bootstrap convergence."),
    ("Step 8",    "Event-level CI (OLS, aggregated)",          "PASS", "20/20 radii.  100% bootstrap convergence."),
]

pdf.set_font("Times", "B", 8.5)
pdf.cell(20, 5.5, "Step", border="B")
pdf.cell(75, 5.5, "Description", border="B")
pdf.cell(18, 5.5, "Status", border="B", align="C")
pdf.cell(0,  5.5, "Notes", border="B")
pdf.ln()

for step, desc, status, note in steps:
    pdf.set_font("Times", "B", 8.5)
    pdf.cell(20, 5, step)
    pdf.set_font("Times", "", 8.5)
    pdf.cell(75, 5, desc)
    pdf.set_font("Times", "B", 8.5)
    pdf.cell(18, 5, status, align="C")
    pdf.set_font("Times", "", 8.5)
    pdf.cell(0,  5, note, ln=True)

pdf.ln(1)
pdf.body(
    "Validation indicators: significant causal effects at all 20 radii "
    "(p < 0.05; majority p < 10^-30); placebo effects near zero throughout; "
    "bootstrap convergence rate 100%.  Pipeline is structurally sound under the new estimator.",
    indent=0
)

# ── 4. Preliminary Observations ──────────────────────────────────────────────
pdf.section("4", "Preliminary Observations  (test-mode only - not for scientific conclusions)")
for obs in [
    "Well-level causal R" + SUP2 + ": 0.005-0.041.  Comparable to OLS baseline "
     "(R" + SUP2 + " = 0.04).  Underfitting expected at n_estimators = 20; "
     "production-run values will differ.",
    "Total causal effect (W -> S) stable across all 20 radii: 2.5 x 10^-5 to 3.1 x 10^-5.",
    "72-104% of the total effect is pressure-mediated (indirect pathway W -> P -> S), "
     "consistent with Darcy-Law pore-pressure diffusion expectations.",
    "Event-level R" + SUP2 + " (Step 8, OLS): 0.06-0.33, substantially higher than "
     "well-level, consistent with aggregation reducing observational noise.",
]:
    pdf.bullet(obs)

# ── 5. Next Steps ─────────────────────────────────────────────────────────────
pdf.section("5", "Next Steps")

next_steps = [
    ("Production run",
     "Revert n_estimators = 100, n_bootstrap = 50 and run full pipeline."),
    ("OLS baseline run",
     "Execute pipeline with OLS retained to generate a direct per-radius comparison."),
    ("Sponsor Q&A",
     "Changes 3, 5, and 6 are gated on Q1-Q3 (log-linear assumption, missing "
     "pressure data, injection-seismicity lag timescale)."),
    ("Change 2 (ungated)",
     "Add Skill Score, MAE, and Spearman rank correlation alongside R" + SUP2 + "."),
    ("Change 7 (ungated)",
     "Implement two-stage hurdle model separating occurrence probability from "
     "conditional magnitude."),
]

for i, (step, desc) in enumerate(next_steps, 1):
    pdf.set_x(10)
    pdf.set_font("Times", "B", 9)
    pdf.cell(48, 5, f"  {i}.  {step}")
    pdf.set_font("Times", "", 9)
    pdf.multi_cell(0, 5, desc)

# ── Save ──────────────────────────────────────────────────────────────────────
outfile = f"ProjectGeminae_Change1_SponsorBrief_{datetime.now().strftime('%Y%m%d')}.pdf"
pdf.output(outfile)
print(f"Saved: {os.path.abspath(outfile)}")
