"""Supplementary Tables 8-13 for v0.9 (markdown)."""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
from pool_hk import hksj
from gradient import metareg

m = lambda v: f"{v:.3f}".replace("-", "−")
m1 = lambda v: f"{v:.1f}".replace("-", "−")
OUTN = {
    "recall": "Word recall (SD)",
    "ser7": "Serial 7s (SD)",
    "verbf": "Verbal fluency (SD)",
    "gcog": "Two-domain composite (SD)",
    "cimp": "Cognitive impairment, HRS (pp)",
    "srh": "Incident fair/poor self-rated health (pp)",
    "lsat": "Life satisfaction (SD)",
}
DESN = {
    "tte": "Target trial",
    "placebo": "Placebo (start at t3)",
    "pretrend": "Pre-trend (t0)",
    "early_vs_late": "Early vs late",
}
ORD = ["recall", "ser7", "verbf", "gcog", "cimp", "srh", "lsat"]
CTRY = {
    "11": "Austria",
    "12": "Germany",
    "13": "Sweden",
    "15": "Spain",
    "16": "Italy",
    "17": "France",
    "18": "Denmark",
    "19": "Greece",
    "20": "Switzerland",
    "23": "Belgium",
    "25": "Israel",
    "28": "Czech Republic",
    "29": "Poland",
    "31": "Luxembourg",
    "34": "Slovenia",
    "35": "Estonia",
    "47": "Croatia",
}
out = []

out.append(
    """**Supplementary Table 8 | Additional outcomes: definitions and availability.** Scores were standardized against the distribution among offline respondents aged ≥50 at the cohort's trial baseline waves. Tests administered by web in HRS were set to missing. Binary outcomes were analysed as incident outcomes among people without the outcome at t0; in pre-trend analyses they are prevalence differences at t0. Waves are harmonised wave numbers.

| Outcome | Domain | Definition | HRS | ELSA | SHARE | CHARLS |
| --- | --- | --- | --- | --- | --- | --- |
| Serial 7s | Working memory, attention | Correct of 5 serial subtractions of 7 from 100 (SHARE numeracy item 2 used where serial 7s not asked) | 6–15 | 7–9 | 4–9 | 1–4 |
| Verbal fluency | Executive function, semantic memory | Animals named in 60 s | 10–15 | 1–5, 7–9 | 4–9 | – |
| Two-domain composite | Episodic memory plus working memory or fluency | Mean of standardized recall and serial 7s (ELSA: recall and verbal fluency) | 6–15 | 1–9 | 4–9 | 1–4 |
| Cognitive impairment | Global cognition | Score ≤11 on the 27-point HRS cognition scale (immediate and delayed recall, serial 7s, backwards counting); scores with imputed components excluded | all | – | – | – |
| Fair/poor self-rated health | General health | Self-rated health fair or poor (5-point scale) | all | all | all | all |
| Life satisfaction | Evaluative well-being | HRS: Satisfaction with Life Scale (leave-behind questionnaire); ELSA: Satisfaction with Life Scale; SHARE: 0–10 life satisfaction; CHARLS: 5-point life satisfaction | 7–15 | 2–9 | 4–9 | 1–4 |
"""
)

NO = pd.read_csv("results_newout_all.csv")
rows = [
    "**Supplementary Table 9 | Additional outcomes by cohort and design.** Augmented inverse-probability-weighted estimates (95% CI) in the overlap population; differences in SD for continuous scores and in percentage points (pp) for binary outcomes; n, person-trials analysed. Pre-trend: difference at t0, before anyone started, adjusted for pre-t0 history. Early vs late: adopters starting at t1 versus t3, outcome at t2. – , not available.",
    "",
    "| Outcome | Design | HRS | ELSA | SHARE | CHARLS |",
    "| --- | --- | --- | --- | --- | --- |",
]
for o in ORD[1:]:
    sc = 100 if o in ("cimp", "srh") else 1
    f = m1 if sc == 100 else m
    for d in DESN:
        cells = []
        for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
            x = NO[(NO.outcome == o) & (NO.design == d) & (NO.cohort == c)]
            cells.append(
                "–"
                if not len(x)
                else f"{f(x.est.iloc[0]*sc)} ({f(x.lo.iloc[0]*sc)} to {f(x.hi.iloc[0]*sc)}); n = {int(x.n.iloc[0]):,}"
            )
        rows.append(f"| {OUTN[o]} | {DESN[d]} | " + " | ".join(cells) + " |")
out.append("\n".join(rows) + "\n")

rows = [
    "**Supplementary Table 10 | Pooled estimates for additional outcomes.** Random-effects means with DerSimonian–Laird (DL) and Hartung–Knapp–Sidik–Jonkman (HKSJ) 95% intervals; for early-versus-late contrasts, HKSJ 90% intervals for equivalence within ±0.10 SD. Binary outcomes as risk differences (pp). Single-cohort rows report the cohort estimate.",
    "",
    "| Outcome | Design | k | Estimate | 95% CI (DL) | 95% CI (HKSJ) | 90% CI (HKSJ) | I² |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
]
for o in ORD[1:]:
    sc = 100 if o in ("cimp", "srh") else 1
    f = m1 if sc == 100 else m
    for d in DESN:
        g = NO[(NO.outcome == o) & (NO.design == d)]
        if len(g) == 1:
            x = g.iloc[0]
            rows.append(
                f"| {OUTN[o]} | {DESN[d]} | 1 | {f(x.est*sc)} | {f(x.lo*sc)} to {f(x.hi*sc)} | – | – | – |"
            )
            continue
        r = hksj(g.est, g.se)
        rows.append(
            f"| {OUTN[o]} | {DESN[d]} | {r['k']} | {f(r['est']*sc)} | {f(r['dl_lo']*sc)} to {f(r['dl_hi']*sc)} | "
            f"{f(r['hk_lo']*sc)} to {f(r['hk_hi']*sc)} | {f(r['hk90_lo']*sc)} to {f(r['hk90_hi']*sc)} | {r['I2']*100:.0f}% |"
        )
out.append("\n".join(rows) + "\n")

ESP = pd.read_csv("results_es_pooled.csv")
ES1 = pd.read_csv("results_es1_pooled.csv")
YN = {
    "z_recall": "Word recall (SD)",
    "z_gcog": "Two-domain composite (SD)",
    "srh": "Fair/poor self-rated health (pp)",
}
rows = [
    "**Supplementary Table 11 | Event study around first internet use, pooled across cohorts.** Callaway–Sant'Anna-type group-time estimates with not-yet-treated and never-treated comparators, regression adjustment for age, sex, education and wealth, aggregated by waves since first use (e) and pooled by HKSJ (95% CI); person-level bootstrap (200 replicates) within cohort. Reference e = −1 (base period: wave before first use) or, in the anticipation specification, e = −2. Pre-trend slope: weighted linear fit through e < 0; deviation: post-adoption estimate (e = 0 to 2, treated-weighted average) minus the extrapolated pre-adoption trend.",
    "",
    "| Quantity | Recall, ref. e = −1 | Composite, ref. e = −1 | Health (pp), ref. e = −1 | Recall, ref. e = −2 | Composite, ref. e = −2 |",
    "| --- | --- | --- | --- | --- | --- |",
]


def cell(P, y, e, sc=1):
    x = P[(P.y == y) & (P.e == e)]
    if not len(x) or x.k.iloc[0] < 2:
        return "–"
    x = x.iloc[0]
    f = m1 if sc == 100 else m
    return f"{f(x.est*sc)} ({f(x.lo*sc)} to {f(x.hi*sc)}); k = {int(x.k)}"


for e, nm in [
    ("-4", "e = −4"),
    ("-3", "e = −3"),
    ("-2", "e = −2"),
    ("-1", "e = −1"),
    ("0", "e = 0"),
    ("1", "e = 1"),
    ("2", "e = 2"),
    ("3", "e = 3"),
    ("slope_pre", "Pre-adoption slope per wave"),
    ("pre_avg", "Mean pre-adoption difference"),
    ("post_avg", "Mean post-adoption difference (e = 0 to 2)"),
    ("post_avg_adj", "Post-adoption deviation from pre-trend"),
]:
    rows.append(
        f'| {nm} | {cell(ESP,"z_recall",e)} | {cell(ESP,"z_gcog",e)} | {cell(ESP,"srh",e,100)} | {cell(ES1,"z_recall",e)} | {cell(ES1,"z_gcog",e)} |'
    )
out.append(
    "\n".join(rows)
    .replace("| e = −1 | – | – | – |", "| e = −1 | 0 (reference) | 0 (reference) | 0 (reference) |")
    .replace("k = 4 | – | – |\n| e = −1", "k = 4 | 0 (reference) | 0 (reference) |\n| e = −1")
    + "\n"
)

ESA = pd.read_csv("results_es_all.csv")
rows = [
    "**Supplementary Table 12 | Event study by cohort.** Estimates (bootstrap SE) with reference e = −1; adopters contributing at e = 0. Waves are internet-observed waves: about 2 years apart in HRS, ELSA and CHARLS and 2–4 years in SHARE (wave 7 has no internet item).",
    "",
    "| Cohort | Outcome | Adopters at e = 0 | Pre-adoption slope per wave | Mean post-adoption difference | Deviation from pre-trend |",
    "| --- | --- | --- | --- | --- | --- |",
]
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    for y in YN:
        x = ESA[(ESA.cohort == c) & (ESA.y == y)].set_index("e")
        sc = 100 if y == "srh" else 1
        f = m1 if sc == 100 else m
        g = lambda k: f'{f(x.loc[k, "est"]*sc)} ({f(x.loc[k, "se"]*sc)})' if k in x.index else "–"
        n0 = int(x.loc["0", "n_treated"]) if "0" in x.index else 0
        rows.append(
            f'| {c} | {YN[y]} | {n0:,} | {g("slope_pre")} | {g("post_avg")} | {g("post_avg_adj")} |'
        )
out.append("\n".join(rows) + "\n")

GR = pd.read_csv("results_gradient.csv", dtype={"unit": str})
rows = [
    "**Supplementary Table 13 | Cross-country selection gradient.** Prevalence, internet use among non-proxy respondents aged ≥50 at the trial baseline waves. Cross-sectional gap: user − non-user recall difference adjusted for age, age², sex and education. Pre-adoption gap: initiator − non-initiator recall difference at t0, before anyone started, adjusted for pre-t0 history. Target trial: recall at t2. Country-level estimates use logistic and linear nuisance models; SE in parentheses. Meta-regression of each estimate on logit prevalence with random effects and Knapp–Hartung SE.",
    "",
    "| Unit | Prevalence | Cross-sectional gap | Pre-adoption gap | Target trial |",
    "| --- | --- | --- | --- | --- |",
]
for u, g in GR.groupby("unit", sort=False):
    name = (
        CTRY.get(u.split("-")[1], u)
        if u.startswith("SHARE-")
        else {
            "HRS": "USA (HRS)",
            "ELSA": "England (ELSA)",
            "CHARLS": "China (CHARLS)",
            "LASI": "India (LASI)",
        }[u]
    )
    v = lambda d: (lambda x: f"{m(x.est.iloc[0])} ({m(x.se.iloc[0])})" if len(x) else "–")(
        g[g.design == d]
    )
    rows.append(f'| {name} | {g.prev.iloc[0]*100:.0f}% | {v("xs")} | {v("pre")} | {v("tte")} |')
rows.append(
    "| Meta-regression slope per logit prevalence | | "
    + " | ".join(
        f'{m(r["slope"])} ({m(r["lo"])} to {m(r["hi"])}); P = {r["p"]:.2f}; k = {r["k"]}'
        for r in [metareg(GR[GR.design == d]) for d in ["xs", "pre", "tte"]]
    )
    + " |"
)
for d, nm in []:
    r = None
    cells = ["", "", ""]
    cells[["xs", "pre", "tte"].index(d)] = (
        f'{m(r["slope"])} ({m(r["lo"])} to {m(r["hi"])}); P = {r["p"]:.2f}; k = {r["k"]}'
    )
    rows.append(f"| Meta-regression slope per logit prevalence | | " + " | ".join(cells) + " |")
out.append("\n".join(rows) + "\n")
open("si09_tables.md", "w").write("\n".join(out))
print("\n".join(out)[:6000])
