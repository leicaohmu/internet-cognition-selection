"""Dose-response by number of waves online (t1-t3) among people offline at t0 (Fig. 4i,j).
Level: recall at t0 (before any use). Change: recall t3 - t0; pre-change: t0 - t-1. All divided by cohort SD of recall at t0.
OLS on dose dummies (vs 0 waves) + age, age^2, female, trial dummies; person-clustered SEs. Pooled with HKSJ.
"""

import os

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from build_trials import load
from within import SCHED4
from pool_hk import hksj
from build_trials import COV, LAG, EXTRA

ADJ = [v for v in COV + LAG + EXTRA if v not in ("age", "female", "recall", "recall_lag", "trial")]


def ols_cl(y, X, cl):
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    u = y - X @ b
    S = np.zeros((X.shape[1],) * 2)
    for g in pd.unique(cl):
        m = cl == g
        s = X[m].T @ u[m]
        S += np.outer(s, s)
    G = len(pd.unique(cl))
    n, k = X.shape
    c = G / (G - 1) * (n - 1) / (n - k)
    return b, np.sqrt(np.diag(c * XtX @ S @ XtX))


rows, ns = [], []
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    p = load(c)
    ix = p.set_index(["id", "wave"])
    parts = []
    for k, (t0, t1, t2, t3) in enumerate(SCHED4[c]):
        b = p[
            (p.wave == t0)
            & (p.iwstat == 1)
            & (p.internet == 0)
            & (p.age >= 50)
            & (p.proxy.fillna(0) != 1)
        ][["id", "age", "female", "recall"] + [v for v in ADJ if v in p.columns]].copy()
        for w, s in [(t0 - 1, "_m"), (t1, "_1"), (t2, "_2"), (t3, "_3")]:
            if w in ix.index.get_level_values("wave"):
                b = b.join(
                    ix.xs(w, level="wave")[["iwstat", "internet", "recall"]].add_suffix(s), on="id"
                )
            else:
                for v in ["iwstat", "internet", "recall"]:
                    b[v + s] = np.nan
        b = b[
            (b.iwstat_1 == 1)
            & (b.iwstat_2 == 1)
            & (b.iwstat_3 == 1)
            & b.internet_1.notna()
            & b.internet_2.notna()
            & b.internet_3.notna()
        ]
        b["dose"] = (b.internet_1 + b.internet_2 + b.internet_3).astype(int)
        b["trial"] = k
        parts.append(b)
    d = pd.concat(parts).dropna(subset=["age", "female", "recall"]).reset_index(drop=True)
    sd = d.recall.std()
    Z = []
    for v_ in ADJ:
        if v_ not in d or d[v_].notna().sum() == 0:
            continue
        x = pd.to_numeric(d[v_], errors="coerce")
        if v_ in ("edu", "wealth_q") or (v_ == "year"):
            Z.append(
                pd.get_dummies(x.fillna(-9).astype(int), prefix=v_, drop_first=True).astype(float)
            )
        else:
            Z.append(pd.DataFrame({v_: x.fillna(x.mean()), v_ + "_m": x.isna().astype(float)}))
    Z = pd.concat(Z, axis=1)
    Z = Z.loc[:, Z.std() > 0]
    for v, yv in [
        ("level", d.recall),
        ("change", d.recall_3 - d.recall),
        ("pre", d.recall - d.recall_m),
    ]:
        for adj in ("basic", "full"):
            g = d.assign(y=yv / sd).dropna(subset=["y"])
            zz = Z.loc[g.index].values if adj == "full" else np.empty((len(g), 0))
            base = [g.age - 65, (g.age - 65) ** 2, g.female] + [
                (g.trial == t).astype(float) for t in sorted(g.trial.unique())[1:]
            ]
            X = np.column_stack(
                [np.ones(len(g))] + [(g.dose == j).astype(float) for j in (1, 2, 3)] + base + [zz]
            )
            be, se = ols_cl(g.y.values, X, g.id.values)
            for j in (1, 2, 3):
                rows.append(
                    dict(
                        cohort=c,
                        outcome=v,
                        adj=adj,
                        dose=j,
                        est=be[j],
                        se=se[j],
                        n=int((g.dose == j).sum()),
                        n0=int((g.dose == 0).sum()),
                    )
                )
            X2 = np.column_stack([np.ones(len(g)), g.dose] + base + [zz])
            be, se = ols_cl(g.y.values, X2, g.id.values)
            rows.append(
                dict(
                    cohort=c,
                    outcome=v,
                    adj=adj,
                    dose="slope",
                    est=be[1],
                    se=se[1],
                    n=len(g),
                    n0=np.nan,
                )
            )
r = pd.DataFrame(rows)
r.to_csv("results_dose.csv", index=False)
P = []
for (v, a, j), g in r.groupby(["outcome", "adj", "dose"], sort=False):
    g = g[np.isfinite(g.se) & (g.se > 0)]
    if len(g) >= 2:
        P.append(
            dict(
                outcome=v,
                adj=a,
                dose=j,
                k=len(g),
                **{
                    x: hksj(g.est, g.se)[x]
                    for x in ["est", "hk_lo", "hk_hi", "dl_lo", "dl_hi", "I2"]
                }
            )
        )
P = pd.DataFrame(P)
P.to_csv("results_dose_pooled.csv", index=False)
pd.set_option("display.width", 200)
print(r.round(3).to_string())
print(P.round(3).to_string())
