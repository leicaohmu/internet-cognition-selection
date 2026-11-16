"""Joint uncertainty of the four design estimates (Fig. 5b): per-person influence functions from cross-fitted AIPW for the target trial,
placebo, before-anyone-starts and early-vs-late designs (recall and composite), giving the within-cohort covariance between design estimates.
"""

import os
import sys

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
os.environ["OUTS"] = "recall,gcog"
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import newout as N
from aipw import aipw
from ladder import nuisances


def pre_phi(t, cols, trim=0.025):
    t = t.reset_index(drop=True)
    X = t[[c for c in cols if c in t and t[c].notna().any()]].astype(float)
    if t.cohort.iloc[0] == "SHARE":
        X = X.join(pd.get_dummies(t.country, prefix="c", dtype=float))
    A = t.A.values.astype(int)
    Y = t.Y.values.astype(float)
    e, pi1, pi0, m1, m0 = nuisances(X, A, Y, t.id.values, False)
    k = (e >= trim) & (e <= 1 - trim)
    e = np.clip(e, 0.01, 0.99)
    pi1, pi0 = np.clip(pi1, 0.05, 1), np.clip(pi0, 0.05, 1)
    A, Y, e, pi1, pi0, m1, m0, ids = (v[k] for v in (A, Y, e, pi1, pi0, m1, m0, t.id.values))
    R = ~np.isnan(Y)
    Yf = np.nan_to_num(Y)
    return (m1 + A * R * (Yf - m1) / (e * pi1)) - (
        m0 + (1 - A) * R * (Yf - m0) / ((1 - e) * pi0)
    ), ids


def phi_of(d):
    res, P = aipw(d, "Y", trim=0.025, return_pseudo=True)
    return np.asarray(P["psi1"] - P["psi0"]), P["t"].id.values


out = []
for c in sys.argv[1:]:
    p = N.panel(c)
    t = N.main[N.main.cohort == c].copy()
    st = N.staggered(c, p)
    for x in ["recall", "gcog"]:
        ph = {}
        d = pd.concat(
            [N.attach(g, p, x, t0, t2) for (t0, t1, t2), g in t.groupby(["t0", "t1", "t2"])]
        )
        d["id"] = c + "_" + d.id
        d = d.assign(cohort=c)
        ph["tte"] = phi_of(d)
        q = pd.concat([N.attach(g, p, x, t0, t2) for (t0, t2), g in st.groupby(["t0", "t2"])])
        q["id"] = c + "_" + q.id
        q = q.assign(cohort=c)
        ph["pla"] = phi_of(
            q[q.grp.isin(["late", "never"])].assign(A=lambda z: (z.grp == "late").astype(int))
        )
        ph["evl"] = phi_of(
            q[q.grp.isin(["early", "late"])].assign(A=lambda z: (z.grp == "early").astype(int))
        )
        r = pd.concat([N.attach(g, p, x, t0, t0) for (t0,), g in t.groupby(["t0"])])
        r["id"] = c + "_" + r.id
        r = r[r.base_y_lag.notna()].assign(cohort=c)
        r["Y"] = r.base_y
        if r.Y.notna().sum() > 300:
            ph["pre"] = pre_phi(r, N.PRECOLS)
        IF = {}
        for k, (phi, ids) in ph.items():
            est = phi.mean()
            s = pd.Series((phi - est) / len(phi)).groupby(ids).sum()
            IF[k] = (est, s)
        ks = list(IF)
        for a in ks:
            for b in ks:
                sa, sb = IF[a][1], IF[b][1]
                j = sa.index.intersection(sb.index)
                out.append(
                    dict(
                        cohort=c,
                        y=x,
                        a=a,
                        b=b,
                        est_a=IF[a][0],
                        cov=float((sa[j] * sb[j]).sum()),
                        var_a=float((sa**2).sum()),
                        var_b=float((sb**2).sum()),
                        shared=len(j),
                    )
                )
        print(c, x, {k: round(v[0], 3) for k, v in IF.items()}, flush=True)
    pd.DataFrame(out).to_csv(f"results_corr_{c}.csv", index=False)
    print("saved", c, flush=True)
