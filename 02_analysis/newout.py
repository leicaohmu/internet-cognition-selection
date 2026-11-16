"""v0.9: apply target trial, placebo, pre-trend and early-vs-late designs to additional outcomes.
Outcomes: serial 7s, verbal fluency, two-domain cognition (memory + serial 7s or fluency), HRS cognitive impairment
(27-point score <=11, Langa-Weir), fair/poor self-rated health (incident), life satisfaction. Recall rerun as a check.
Each analysis adds the outcome measured at t0 and t-1 as covariates."""

import os
import sys

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import build_trials
from build_trials import load, SCHED, COV
from aipw import aipw
from ladder import aipw_cols
from within import SCHED4

COV.extend(["base_y", "base_y_lag"])  # same list object used by aipw.design
main = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
OUTS = os.environ.get("OUTS", "recall,ser7,verbf,gcog,cimp,srh,lsat").split(",")
BIN = {"cimp", "srh"}


def panel(c):
    p = load(c)
    n = pd.read_csv(f"newout_{c}.csv.gz", dtype={"id": str}, low_memory=False)
    if c == "HRS":
        n["id"] = n.id.str.lstrip("0")
    n = n.drop_duplicates(["id", "wave"])
    p = p.merge(
        n[[x for x in n if x in ("id", "wave", "ser7", "verbf", "cog27", "lsat")]],
        on=["id", "wave"],
        how="left",
    )
    for x in ["ser7", "verbf", "cog27", "lsat"]:
        if x not in p:
            p[x] = np.nan
    p.loc[p.recall_web.fillna(0) == 1, ["ser7", "cog27", "verbf"]] = np.nan  # HRS web-mode tests
    # z-scores against the cohort's trial-baseline distribution (offline, aged >=50, baseline waves)
    base = p[(p.internet == 0) & (p.age >= 50) & p.wave.isin({s[0] for s in SCHED[c]})]
    for x in ["recall", "ser7", "verbf", "lsat"]:
        mu, sd = base[x].mean(), base[x].std()
        p["z_" + x] = (p[x] - mu) / sd if sd and sd > 0 else np.nan
    second = "z_verbf" if c == "ELSA" else "z_ser7"
    p["z_gcog"] = p[["z_recall", second]].mean(axis=1).where(p.z_recall.notna() & p[second].notna())
    p["cimp"] = (p.cog27 <= 11).astype(float).where(p.cog27.notna()) if c == "HRS" else np.nan
    p["srh"] = p.srh_poor
    return p


def outcome_cols(x):
    return {
        "recall": "z_recall",
        "ser7": "z_ser7",
        "verbf": "z_verbf",
        "gcog": "z_gcog",
        "cimp": "cimp",
        "srh": "srh",
        "lsat": "z_lsat",
    }[x]


def attach(b, p, x, w0, wy, alive_col=None):
    """b: rows with id at t0; attach outcome x at wave wy (Y), at t0 (base_y) and t0-1 (base_y_lag)."""
    col = outcome_cols(x)
    ix = p.set_index(["id", "wave"])[col]
    get = lambda w: (
        b.id.map(ix.xs(w, level="wave")) if w in set(p.wave) else pd.Series(np.nan, index=b.index)
    )
    b = b.copy()
    b["Y"] = get(wy).values
    b["base_y"] = get(w0).values
    b["base_y_lag"] = get(w0 - 1).values
    if x in BIN:  # incident outcome among those free of it at t0
        b["Y"] = b.Y.where(b.base_y == 0)
    return b


def run(d, cohort, design, x, fn):
    d = d[d.base_y == 0] if x in BIN else d
    if d.Y.notna().sum() < 200 or d.A.sum() < 40 or (1 - d.A).sum() < 40:
        return None
    d = d.assign(cohort=cohort)
    r = fn(d)
    r.update(design=design, outcome=x, cohort=cohort)
    return r


def tte(c, p):
    rows = []
    t = main[main.cohort == c].copy()
    for x in OUTS:
        parts = []
        for (t0, t1, t2), g in t.groupby(["t0", "t1", "t2"]):
            g = attach(g, p, x, t0, t2)
            parts.append(g)
        d = pd.concat(parts)
        d["id"] = c + "_" + d.id
        r = run(d, c, "tte", x, lambda d: aipw(d, "Y", trim=0.025))
        if r:
            rows.append(r)
            print(
                c,
                "tte",
                x,
                f"{r['est']:+.3f} ({r['lo']:+.3f},{r['hi']:+.3f}) n={r['n']}",
                flush=True,
            )
    return rows


def staggered(c, p):
    """people offline at t0 with known status at t1..t3; groups by first use. Returns frame with waves."""
    ix = p.set_index(["id", "wave"])
    out = []
    for k, (t0, t1, t2, t3) in enumerate(SCHED4[c]):
        b = p[
            (p.wave == t0)
            & (p.iwstat == 1)
            & (p.internet == 0)
            & (p.age >= 50)
            & (p.proxy.fillna(0) != 1)
        ].copy()
        for w, s in [(t1, "_1"), (t2, "_2"), (t3, "_3")]:
            b = b.join(ix.xs(w, level="wave")[["iwstat", "internet"]].add_suffix(s), on="id")
        b = b[(b.iwstat_1 == 1) & (b.iwstat_2 == 1) & b.internet_1.notna() & b.internet_2.notna()]
        b["grp"] = np.select(
            [
                b.internet_1 == 1,
                (b.internet_1 == 0) & (b.internet_2 == 1),
                (b.internet_1 == 0) & (b.internet_2 == 0) & (b.internet_3 == 1) & (b.iwstat_3 == 1),
                (b.internet_1 == 0) & (b.internet_2 == 0) & (b.internet_3 == 0) & (b.iwstat_3 == 1),
            ],
            ["early", "middle", "late", "never"],
            "unknown",
        )
        b["trial"] = k
        b["t0"], b["t2"] = t0, t2
        out.append(b)
    return pd.concat(out, ignore_index=True)


def falsif(c, p):
    rows = []
    s = staggered(c, p)
    for x in OUTS:
        parts = [attach(g, p, x, t0, t2) for (t0, t2), g in s.groupby(["t0", "t2"])]
        d = pd.concat(parts)
        d["id_raw"] = d.id
        d["id"] = c + "_" + d.id_raw
        # placebo: late (start t3) vs never, among offline through t2
        pl = d[d.grp.isin(["late", "never"])].assign(A=lambda z: (z.grp == "late").astype(int))
        r = run(pl, c, "placebo", x, lambda d: aipw(d, "Y", trim=0.025))
        if r:
            rows.append(r)
            print(c, "placebo", x, f"{r['est']:+.3f} ({r['lo']:+.3f},{r['hi']:+.3f})", flush=True)
        el = d[d.grp.isin(["early", "late"])].assign(A=lambda z: (z.grp == "early").astype(int))
        r = run(el, c, "early_vs_late", x, lambda d: aipw(d, "Y", trim=0.025))
        if r:
            rows.append(r)
            print(c, "EvL", x, f"{r['est']:+.3f} ({r['lo']:+.3f},{r['hi']:+.3f})", flush=True)
    return rows


PRECOLS = [
    "age",
    "female",
    "edu",
    "partnered",
    "live_alone",
    "rural",
    "year",
    "dep_lag",
    "srh_poor_lag",
    "iadl_any_lag",
    "adl_any_lag",
    "recall_lag",
    "ncond_lag",
    "work_lag",
    "wealth_q_lag",
    "base_y_lag",
]


def pretrend(c, p):
    rows = []
    t = main[main.cohort == c].copy()
    for x in OUTS:
        parts = []
        for (t0,), g in t.groupby(["t0"]):
            g = attach(g, p, x, t0, t0)  # outcome = value at t0
            parts.append(g)
        d = pd.concat(parts)
        d["id"] = c + "_" + d.id
        d = d[d.base_y_lag.notna()]
        if x in BIN:
            d = d[d.base_y_lag == 0]
            d["Y"] = d.base_y
        else:
            d["Y"] = d.base_y
        if d.Y.notna().sum() < 300 or d.A.sum() < 40:
            continue
        d = d.assign(cohort=c)
        r = aipw_cols(d, "Y", PRECOLS)
        r.update(design="pretrend", outcome=x, cohort=c)
        rows.append(r)
        print(c, "pre", x, f"{r['est']:+.3f} ({r['lo']:+.3f},{r['hi']:+.3f})", flush=True)
    return rows


if __name__ == "__main__":
    cohorts = sys.argv[1:] or ["HRS", "ELSA", "SHARE", "CHARLS"]
    for c in cohorts:
        p = panel(c)
        rows = tte(c, p) + falsif(c, p) + pretrend(c, p)
        pd.DataFrame(rows).to_csv(f"results_newout_{c}.csv", index=False)
        print("saved", c, flush=True)
