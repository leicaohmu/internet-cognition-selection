"""Evidence ladder: (1) cross-sectional association, (2) target trial emulation (primary), (3) placebo initiation 1 wave
after outcome, (4) placebo initiation 2 waves after outcome. Same outcome scales across rungs."""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings
import sys

warnings.filterwarnings("ignore")
from build_trials import load, COV, LAG, EXTRA
import aipw as A
from aipw import nuisances, summarise, meta_dl

BASIC = ["age", "female", "edu", "partnered", "live_alone", "wealth_q", "work", "rural", "year"]


def aipw_cols(t, y, cols, trim=0.025):
    t = t.reset_index(drop=True)
    X = t[[c for c in cols if c in t and t[c].notna().any()]].astype(float)
    if t.cohort.iloc[0] == "SHARE":
        X = X.join(pd.get_dummies(t.country, prefix="c", dtype=float))
    Aa = t.A.values.astype(int)
    Y = t[y].values.astype(float)
    binary = set(np.unique(Y[~np.isnan(Y)])) <= {0.0, 1.0}
    e, pi1, pi0, m1, m0 = nuisances(X, Aa, Y, t.id.values, binary)
    keep = (e >= trim) & (e <= 1 - trim)
    e = np.clip(e, 0.01, 0.99)
    pi1, pi0 = np.clip(pi1, 0.05, 1), np.clip(pi0, 0.05, 1)
    Aa, Y, e, pi1, pi0, m1, m0, ids = (v[keep] for v in (Aa, Y, e, pi1, pi0, m1, m0, t.id.values))
    R = ~np.isnan(Y)
    Yf = np.nan_to_num(Y)
    psi1 = m1 + Aa * R * (Yf - m1) / (e * pi1)
    psi0 = m0 + (1 - Aa) * R * (Yf - m0) / ((1 - e) * pi0)
    r = summarise(psi1, psi0, ids, binary)
    r.update(n=int(keep.sum()), n_A1=int(Aa.sum()), y=y, cohort=t.cohort.iloc[0])
    return r


def cross_sectional(c):
    """All respondents aged >=50 at the trial baseline waves (LASI: wave 1); exposure = same-wave internet use."""
    if c == "LASI":
        pd.DataFrame(columns=["id", "wave"]).to_csv("extra_LASI.csv.gz", index=False)
        waves = [1]
    else:
        from build_trials import SCHED

        waves = sorted({s[0] for s in SCHED[c]})
    p = load(c)
    d = p[p.wave.isin(waves) & (p.iwstat == 1) & (p.age >= 50) & p.internet.notna()].copy()
    if c != "LASI":
        m = pd.read_csv("trials.csv.gz", low_memory=False)
        m = m[m.cohort == c]
        mu, sd = m.recall.mean(), m.recall.std()
    else:
        mu, sd = d.recall.mean(), d.recall.std()
    d["A"] = d.internet
    d["Y_recall"] = (d.recall - mu) / sd
    d["Y_dep"] = d.dep_bin
    d["Y_iadl"] = d.iadl_any
    d["cohort"] = c
    d["id"] = c + "_" + d.id.astype(str)
    return d


if __name__ == "__main__":
    rows = []
    for c in ["HRS", "ELSA", "SHARE", "CHARLS", "LASI"]:
        d = cross_sectional(c)
        for y in ["Y_recall", "Y_iadl", "Y_dep"]:
            r = aipw_cols(d, y, BASIC)
            r["rung"] = "1 Cross-sectional association"
            rows.append(r)
            print(
                c, y, f"{r['est']:+.3f}", f"RR={r.get('rr', np.nan):.2f}", "n", r["n"], flush=True
            )
    pd.DataFrame(rows).to_csv("results_ladder_cross.csv", index=False)
