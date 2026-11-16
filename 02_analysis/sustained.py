"""Sustained internet use (v1.1): effect of the regime 'online at t1 and t2' versus 'offline at t1 and t2' on recall at t3
among people offline at t0, by inverse probability weighting for time-varying confounding (g-method).
A1 ~ L0 ; A2 ~ L0, A1, L1 (health and recall at t1). Stabilized weights truncated at the 1st/99th percentiles.
Falsification: the same weights applied to recall at t0 (before anyone started) and, for future sustained users,
contrast at t0 adjusted for t-1 (pre-trend). Person bootstrap (B=100)."""

import os
import sys

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from build_trials import load, COV, LAG, EXTRA
from within import SCHED4
from pool_hk import hksj

B = int(os.environ.get("B", 100))
rng = np.random.default_rng(11)
L1 = ["srh_poor", "adl_any", "iadl_any", "dep", "ncond", "recall", "work"]


def build(c):
    p = load(c)
    ix = p.set_index(["id", "wave"])
    out = []
    main = pd.read_csv("trials.csv.gz", usecols=["cohort", "recall"], low_memory=False)
    m = main[main.cohort == c]
    mu, sd = m.recall.mean(), m.recall.std()
    for k, (t0, t1, t2, t3) in enumerate(SCHED4[c]):
        b = p[
            (p.wave == t0)
            & (p.iwstat == 1)
            & (p.internet == 0)
            & (p.age >= 50)
            & (p.proxy.fillna(0) != 1)
        ].copy()
        b = b.join(ix.xs(t1, level="wave")[["iwstat", "internet"] + L1].add_suffix("_t1"), on="id")
        b = b.join(ix.xs(t2, level="wave")[["iwstat", "internet"]].add_suffix("_t2"), on="id")
        b = b.join(
            ix.xs(t3, level="wave")[["iwstat", "recall", "recall_web"]].add_suffix("_t3"), on="id"
        )
        b = b[
            (b.iwstat_t1 == 1) & (b.iwstat_t2 == 1) & b.internet_t1.notna() & b.internet_t2.notna()
        ]
        b["Y"] = ((b.recall_t3 - mu) / sd).where(
            (b.iwstat_t3 == 1) & (b.recall_web_t3.fillna(0) == 0)
        )
        b["Y0"] = (b.recall - mu) / sd
        b["trial"] = k
        out.append(b)
    d = pd.concat(out, ignore_index=True)
    d["cohort"] = c
    return d


from sklearn.linear_model import LinearRegression

PRE = [
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
]


def reg():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LinearRegression()
    )


def Xmat(d, cols):
    X = d[[v for v in cols if v in d and d[v].notna().any()]].astype(float)
    if d.cohort.iloc[0] == "SHARE":
        X = X.join(pd.get_dummies(d.country, prefix="c", dtype=float))
    X["trial"] = d.trial.values
    return X


def ice(d, a):
    """Iterated conditional expectation g-formula for regime (A1=a, A2=a) on Y."""
    L0 = [v for v in COV + LAG + EXTRA]
    X0 = Xmat(d, L0)
    X1 = X0.assign(**{v + "_t1": d[v + "_t1"].astype(float) for v in L1})
    ok = d.Y.notna().values & (d.internet_t1.values == a) & (d.internet_t2.values == a)
    m2 = reg().fit(X1[ok], d.Y.values[ok])
    s1 = d.internet_t1.values == a
    q2 = m2.predict(X1[s1])
    m1 = reg().fit(X0[s1], q2)
    return m1.predict(X0).mean()


def pre(d):
    """future sustained users vs never: recall at t0 adjusted for pre-t0 history (regression)."""
    k = d[
        ((d.internet_t1 == 1) & (d.internet_t2 == 1))
        | ((d.internet_t1 == 0) & (d.internet_t2 == 0))
    ]
    k = k[k.Y0.notna() & k.recall_lag.notna()]
    if len(k) < 200 or k.internet_t1.sum() < 30:
        return np.nan
    X = Xmat(k, PRE)
    X["A"] = k.internet_t1.values
    m = reg().fit(X, k.Y0.values)
    X1, X0_ = X.copy(), X.copy()
    X1["A"] = 1
    X0_["A"] = 0
    return (m.predict(X1) - m.predict(X0_)).mean()


def contrasts(d, w=None):
    return {"sustained-never": ice(d, 1) - ice(d, 0), "pre:sustained-never": pre(d)}


def run(c):
    d = build(c).reset_index(drop=True)
    pt = contrasts(d)
    u, inv = np.unique(d.id.values, return_inverse=True)
    bs = []
    for _ in range(B):
        idx = np.repeat(np.arange(len(d)), rng.poisson(1, len(u))[inv])
        db = d.loc[idx].reset_index(drop=True)
        bs.append(contrasts(db))
    bs = pd.DataFrame(bs)
    n = dict(
        always=int(((d.internet_t1 == 1) & (d.internet_t2 == 1)).sum()),
        never=int(((d.internet_t1 == 0) & (d.internet_t2 == 0)).sum()),
    )
    rows = [
        dict(
            cohort=c,
            term=k,
            est=v,
            se=bs[k].std() if np.isfinite(v) else np.nan,
            n_always=n["always"],
            n_never=n["never"],
            n=len(d),
        )
        for k, v in pt.items()
    ]
    for r in rows:
        print(c, r["term"], f"{r['est']:+.3f} ({r['se']:.3f})", flush=True)
    return rows


if __name__ == "__main__" and os.environ.get("TRAJ") != "1":
    rows = []
    for c in sys.argv[1:] or ["HRS", "ELSA", "SHARE", "CHARLS"]:
        rows += run(c)
    R = pd.DataFrame(rows)
    R.to_csv("results_sustained.csv", index=False)
    P = []
    for t, g in R.groupby("term"):
        g = g[g.se.notna() & (g.se > 0)]
        if len(g) >= 2:
            r = hksj(g.est, g.se)
            P.append(
                dict(
                    term=t,
                    k=r["k"],
                    est=r["est"],
                    lo=r["hk_lo"],
                    hi=r["hk_hi"],
                    lo90=r["hk90_lo"],
                    hi90=r["hk90_hi"],
                    I2=r["I2"],
                )
            )
    P = pd.DataFrame(P)
    P.to_csv("results_sustained_pooled.csv", index=False)
    print(P.round(3).to_string())


# ---------------------------------------------------------------- trajectories of sustained users vs never users
def traj(c):
    p = load(c)
    ix = p.set_index(["id", "wave"])
    d = build(c)
    m = pd.read_csv("trials.csv.gz", usecols=["cohort", "recall"], low_memory=False)
    m = m[m.cohort == c]
    mu, sd = m.recall.mean(), m.recall.std()
    d = d[
        ((d.internet_t1 == 1) & (d.internet_t2 == 1))
        | ((d.internet_t1 == 0) & (d.internet_t2 == 0))
    ].copy()
    d["S"] = d.internet_t1.astype(float)
    sched = {k: w for k, w in enumerate(SCHED4[c])}
    for pos, off in [("t-1", -1), ("t0", 0), ("t1", 1), ("t2", 2), ("t3", 3)]:
        vals = []
        for k, g in d.groupby("trial"):
            w = sched[k][0] + off if off >= 0 else sched[k][0] - 1
            if off > 0:
                w = sched[k][off]
            if w not in set(p.wave):
                vals.append(pd.Series(np.nan, index=g.index))
                continue
            r = ix.xs(w, level="wave")[["recall", "recall_web"]]
            r = r.recall.where(r.recall_web.fillna(0) == 0)
            vals.append(g.id.map(r))
        d["r_" + pos] = (pd.concat(vals).reindex(d.index) - mu) / sd

    def diffs(dd):
        out = {}
        for pos in ["t-1", "t0", "t1", "t2", "t3"]:
            k = dd[dd["r_" + pos].notna()]
            if len(k) < 200:
                out[pos] = np.nan
                continue
            a = (k.age - 65) / 10
            X = np.column_stack(
                [np.ones(len(k)), k.S, a, a**2, k.female.fillna(0.5), k.edu.fillna(k.edu.mean())]
                + [(k.trial == t).astype(float) for t in sorted(k.trial.unique())[1:]]
            )
            out[pos] = np.linalg.lstsq(X, k["r_" + pos].values, rcond=None)[0][1]
        out["slope_pre"] = out["t0"] - out["t-1"]
        out["slope_post"] = (out["t3"] - out["t0"]) / 3
        out["slope_diff"] = out["slope_post"] - out["slope_pre"]
        return out

    pt = diffs(d)
    u, inv = np.unique(d.id.values, return_inverse=True)
    bs = []
    for _ in range(B):
        idx = np.repeat(np.arange(len(d)), rng.poisson(1, len(u))[inv])
        bs.append(diffs(d.iloc[idx]))
    bs = pd.DataFrame(bs)
    rows = [dict(cohort=c, term="traj:" + k, est=v, se=bs[k].std()) for k, v in pt.items()]
    for r in rows:
        print(c, r["term"], f"{r['est']:+.3f} ({r['se']:.3f})", flush=True)
    return rows


if __name__ == "__main__" and os.environ.get("TRAJ") == "1":
    rows = []
    for c in ["HRS", "ELSA", "SHARE"]:
        rows += traj(c)
    R = pd.DataFrame(rows)
    R.to_csv("results_sustained_traj.csv", index=False)
    P = []
    for t, g in R.groupby("term"):
        g = g[g.se.notna() & (g.se > 0)]
        if len(g) >= 2:
            r = hksj(g.est, g.se)
            P.append(dict(term=t, k=r["k"], est=r["est"], lo=r["hk_lo"], hi=r["hk_hi"]))
    print(pd.DataFrame(P).round(3).to_string())
    pd.DataFrame(P).to_csv("results_sustained_traj_pooled.csv", index=False)
