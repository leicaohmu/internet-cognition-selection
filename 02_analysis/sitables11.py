import numpy as np
import pandas as pd

m = lambda v: f"{v:.3f}".replace("-", "−")
m2 = lambda v: f"{v:.2f}".replace("-", "−")
m1 = lambda v: f"{v:.1f}".replace("-", "−")
out = []
out.append(open("table1_v11.md").read().strip() + "\n")
AP = pd.read_csv("results_anatomy_pooled.csv")
AR = pd.read_csv("results_anatomy.csv")
NM = {
    "sp_internet": "Spouse uses internet",
    "lvl": "Recall, mean of t−1 and t0 (per SD)",
    "drz": "Recall change t−1 to t0 (per SD)",
    "edu": "Education (per SD)",
    "wealth_q": "Wealth quintile (per SD)",
    "partnered": "Partnered",
    "work": "In paid work",
    "ncond": "Chronic conditions (per SD)",
    "dep": "Depression score (per SD)",
    "live_alone": "Lives alone",
    "female": "Women",
    "srh_poor": "Fair/poor self-rated health",
    "iadl_any": "Any IADL difficulty",
    "age": "Age (per 10 years)",
    "Q2": "Recall quintile 2 vs 1",
    "Q3": "Recall quintile 3 vs 1",
    "Q4": "Recall quintile 4 vs 1",
    "Q5": "Recall quintile 5 vs 1",
    "stable": "Stable vs declining recall (tertiles of change)",
    "improve": "Improving vs declining recall (tertiles of change)",
}
rows = [
    "**Supplementary Table 17 | Predictors of starting internet use and decomposition of the baseline recall gap.** Odds ratios from logistic regressions among target-trial eligibles with trial (and country) indicators and person-clustered standard errors; quintile models adjusted for age and sex; change models adjusted for mean recall and baseline characteristics. Decomposition (Gelbach) of the unadjusted initiator − non-initiator recall gap at t0 (SD), person bootstrap (100 replicates). Pooled by random effects with Hartung–Knapp 95% CIs; k = 4 cohorts.",
    "",
    "| Quantity | Pooled estimate (95% CI) | HRS | ELSA | SHARE | CHARLS |",
    "| --- | --- | --- | --- | --- | --- |",
]
for part, terms in [
    ("quintile", ["Q2", "Q3", "Q4", "Q5"]),
    ("change", ["stable", "improve"]),
    ("predictor", list(NM)[:14]),
]:
    for t in terms:
        p = AP[(AP.part == part) & (AP.term == t)].iloc[0]
        cells = [f"OR {m2(np.exp(p.est))} ({m2(np.exp(p.lo))} to {m2(np.exp(p.hi))})"]
        for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
            x = AR[(AR.part == part) & (AR.term == t) & (AR.cohort == c)]
            cells.append(m2(np.exp(x.est.iloc[0])) if len(x) else "–")
        rows.append(f"| {NM[t]} | " + " | ".join(cells) + " |")
GN = [
    "total",
    "Age and sex",
    "Education, wealth and work",
    "Health",
    "Household and spouse",
    "Lifestyle",
    "Earlier memory (t−1)",
    "unexplained",
]
for t in GN:
    p = AP[(AP.part == "gelbach") & (AP.term == t)].iloc[0]
    cells = [f"{m(p.est)} ({m(p.lo)} to {m(p.hi)})"] + [
        m(AR[(AR.part == "gelbach") & (AR.term == t) & (AR.cohort == c)].est.iloc[0])
        for c in ["HRS", "ELSA", "SHARE", "CHARLS"]
    ]
    lab = {
        "total": "Gap at t0: total (SD)",
        "unexplained": "Gap at t0: not explained by measured characteristics",
    }.get(t, "Gap at t0 explained by: " + t.lower())
    rows.append(f"| {lab} | " + " | ".join(cells) + " |")
out.append("\n".join(rows) + "\n")
ST = pd.read_csv("results_sustained_traj.csv")
SP = pd.read_csv("results_sustained_traj_pooled.csv")
SU = pd.read_csv("results_sustained.csv")
SUP = pd.read_csv("results_sustained_pooled.csv")
rows = [
    "**Supplementary Table 18 | Sustained internet use.** People offline at t0 who were online at both t1 and t2 compared with people offline at both. Trajectory rows: age-, sex- and education-adjusted recall differences (SD) at each wave (person bootstrap SE; pooled HKSJ). g-computation rows: iterated conditional expectation estimate for recall at t3 under the two regimes, adjusted for baseline covariates and health and recall at t1, and the difference at t0 adjusted for pre-t0 history. CHARLS has no wave before t0 and too few sustained users for stable g-computation.",
    "",
    "| Quantity | Pooled (95% CI) | HRS | ELSA | SHARE | CHARLS |",
    "| --- | --- | --- | --- | --- | --- |",
]
for t, nm in [
    ("traj:t-1", "Difference at t−1"),
    ("traj:t0", "Difference at t0"),
    ("traj:t1", "Difference at t1"),
    ("traj:t2", "Difference at t2"),
    ("traj:t3", "Difference at t3"),
    ("traj:slope_pre", "Change per wave, t−1 to t0"),
    ("traj:slope_post", "Change per wave, t0 to t3"),
    ("traj:slope_diff", "Difference in change per wave"),
]:
    p = SP[SP.term == t].iloc[0]
    cells = [f"{m(p.est)} ({m(p.lo)} to {m(p.hi)})"]
    for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
        x = ST[(ST.term == t) & (ST.cohort == c)]
        cells.append(f"{m(x.est.iloc[0])} ({m(x.se.iloc[0])})" if len(x) else "–")
    rows.append(f"| {nm} | " + " | ".join(cells) + " |")
for t, nm in [
    ("sustained-never", "g-computation: recall at t3, sustained vs never"),
    ("pre:sustained-never", "Recall at t0, future sustained vs never, adjusted for t−1"),
]:
    p = SUP[SUP.term == t].iloc[0]
    cells = [f"{m(p.est)} ({m(p.lo)} to {m(p.hi)})"]
    for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
        x = SU[(SU.term == t) & (SU.cohort == c)]
        cells.append(
            f"{m(x.est.iloc[0])} ({m(x.se.iloc[0])})"
            if len(x) and np.isfinite(x.est.iloc[0]) and c != "CHARLS"
            else "–"
        )
    rows.append(f"| {nm} | " + " | ".join(cells) + " |")
out.append("\n".join(rows) + "\n")
DM = pd.read_csv("results_dementia_HRS.csv")
rows = [
    "**Supplementary Table 19 | Dementia and cognitive impairment in HRS from the imputed 27-point cognition score (Langa–Weir classification).** Dementia, score 0–6; cognitive impairment without dementia or dementia, 0–11; self-respondents, web-mode tests excluded, proxies at the outcome wave missing. Incident outcomes among people without the condition at t0; pre-trend rows are prevalence differences and ratios at t0.",
    "",
    "| Outcome | Design | Risk difference, pp (95% CI) | Risk ratio (95% CI) | n |",
    "| --- | --- | --- | --- | --- |",
]
DN = {
    "tte": "Target trial",
    "placebo": "Placebo (start at t3)",
    "pretrend": "Pre-trend (t0)",
    "early_vs_late": "Early vs late",
}
for o, on in [("dem", "Dementia"), ("cind", "Cognitive impairment or dementia")]:
    for d in DN:
        x = DM[(DM.outcome == o) & (DM.design == d)].iloc[0]
        rows.append(
            f"| {on} | {DN[d]} | {m1(x.est * 100)} ({m1(x.lo * 100)} to {m1(x.hi * 100)}) | {m2(x.rr)} ({m2(x.rr_lo)} to {m2(x.rr_hi)}) | {int(x.n):,} |"
        )
out.append("\n".join(rows) + "\n")
open("si11_extra.md", "w").write("\n".join(out))
print(out[-1][:400])
