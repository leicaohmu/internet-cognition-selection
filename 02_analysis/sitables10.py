"""Supplementary Note 4 and Tables 14-15 (physical activity) for v1.0."""

import pandas as pd

m = lambda v: f"{v:.3f}".replace("-", "−")
m1 = lambda v: f"{v:.1f}".replace("-", "−")
D = pd.read_csv("results_behav_vig_all.csv")
P = pd.read_csv("results_behav_vig_pooled.csv")
E = pd.read_csv("results_es_vig_pooled.csv")
EA = pd.read_csv("results_es_vig_all.csv")
ON = {
    "recall": "Word recall (SD)",
    "gcog": "Two-domain composite (SD)",
    "srh": "Incident fair/poor self-rated health (pp)",
}
DN = {
    "tte": "Target trial",
    "placebo": "Placebo (start at t3)",
    "pretrend": "Pre-trend (t0)",
    "early_vs_late": "Early vs late",
}
out = []
out.append("""## Supplementary Note 4 | Order of analyses

The analyses were developed in this order. (1) Target-trial emulation for recall, IADL difficulty and depressive symptoms, with the conventional cross-sectional comparison, adjustment ladder, negative control outcome and mortality benchmark. (2) Placebo trials, pre-trend tests and descriptive trajectories. (3) Equivalence margins for the early-versus-late contrast (±0.10 SD for recall, ±2 percentage points for binary outcomes), fixed before that contrast was estimated; then the within-adopter analyses and the simulation. (4) Additional cognitive and health outcomes, the event study and the cross-country analysis, added to test whether the recall findings generalised; each reused the estimator, covariates, trimming and pooling of steps 1–3 without modification. (5) The physical activity analyses, added last to test whether the signature is specific to internet use; weekly vigorous activity was chosen because it is the only other self-selected, brain-health-relevant behaviour recorded in all four cohorts, and the designs were applied unchanged. No analysis plan was registered. All results of steps 4 and 5 are reported, including null and heterogeneous estimates.
""")
rows = [
    "**Supplementary Table 14 | Starting weekly vigorous physical activity: estimates by cohort and design.** Augmented inverse-probability-weighted estimates (95% CI) among adults reporting no weekly vigorous activity at t0, in the overlap population; n, person-trials. – , not estimable.",
    "",
    "| Outcome | Design | HRS | ELSA | SHARE | CHARLS |",
    "| --- | --- | --- | --- | --- | --- |",
]
for o in ON:
    sc = 100 if o == "srh" else 1
    f = m1 if sc == 100 else m
    for d in DN:
        cells = []
        for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
            x = D[(D.outcome == o) & (D.design == d) & (D.cohort == c)]
            cells.append(
                "–"
                if not len(x)
                else f"{f(x.est.iloc[0]*sc)} ({f(x.lo.iloc[0]*sc)} to {f(x.hi.iloc[0]*sc)}); n = {int(x.n.iloc[0]):,}"
            )
        rows.append(f"| {ON[o]} | {DN[d]} | " + " | ".join(cells) + " |")
out.append("\n".join(rows) + "\n")
rows = [
    "**Supplementary Table 15 | Starting weekly vigorous physical activity: pooled estimates and event study.** Random-effects means with Hartung–Knapp–Sidik–Jonkman 95% and 90% intervals. Event study as in Supplementary Table 11 (reference e = −1; HRS, ELSA and SHARE).",
    "",
    "| Outcome | Quantity | k | Estimate | 95% CI | 90% CI |",
    "| --- | --- | --- | --- | --- | --- |",
]
for o in ON:
    sc = 100 if o == "srh" else 1
    f = m1 if sc == 100 else m
    for d in DN:
        x = P[(P.outcome == o) & (P.design == d)].iloc[0]
        rows.append(
            f"| {ON[o]} | {DN[d]} | {int(x.k)} | {f(x.est*sc)} | {f(x.lo*sc)} to {f(x.hi*sc)} | {f(x.lo90*sc)} to {f(x.hi90*sc)} |"
        )
    y = {"recall": "z_recall", "gcog": "z_gcog", "srh": "srh"}[o]
    for e, nm in [
        ("slope_pre", "Event study: pre-start slope per wave"),
        ("post_avg", "Event study: mean post-start difference"),
        ("post_avg_adj", "Event study: deviation from pre-trend"),
    ]:
        x = E[(E.y == y) & (E.e == e)].iloc[0]
        rows.append(
            f"| {ON[o]} | {nm} | {int(x.k)} | {f(x.est*sc)} | {f(x.lo*sc)} to {f(x.hi*sc)} | – |"
        )
out.append("\n".join(rows) + "\n")
open("si10_extra.md", "w").write("\n".join(out))
print("\n".join(out)[-2500:])
