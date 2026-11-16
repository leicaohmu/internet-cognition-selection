"""Robustness for Fig. 6: (1) memory level and slope over three waves (more reliable than a one-wave change) as predictors of starting and stopping;
(2) stopping defined as non-use at two consecutive waves; (3) unexplained gap vs target-trial and placebo estimates.
"""

import os

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
from build_trials import load
import anatomy as A
from pool_hk import hksj

COV = [
    "age",
    "female",
    "edu",
    "wealth_q",
    "work",
    "srh_poor",
    "ncond",
    "iadl_any",
    "dep",
    "partnered",
    "live_alone",
    "sp_internet",
]
rows = []


def fit(e, yv, fe, c, model):
    for v in ["edu", "wealth_q", "ncond", "dep", "age"]:
        e[v] = (e[v] - e[v].mean()) / e[v].std()
    preds = ["lvl3", "slope3"] + COV
    X = pd.concat([A.design(e, [v for v in preds if v in e]), fe], axis=1)
    m = A._glm(e[yv], X, pd.factorize(e.id)[0])
    for v in ["lvl3", "slope3", "age", "edu"]:
        rows.append(
            dict(
                cohort=c,
                model=model,
                term=v,
                est=m.params[v],
                se=m.bse[v],
                n=len(e),
                events=int(e[yv].sum()),
            )
        )
    print(
        c,
        model,
        len(e),
        {v: round(float(np.exp(m.params[v])), 2) for v in ["lvl3", "slope3"]},
        flush=True,
    )


for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    p = load(c)
    p = p[p.iwstat == 1]
    rw = sorted(p[p.recall.notna()].wave.unique())
    ix = p.set_index(["id", "wave"])
    rec = lambda w: ix.xs(w, level="wave").recall if w in rw else pd.Series(dtype=float)
    # ---- starting
    d = A.prep(c)
    d = d[d.rz.notna()].copy()
    mu, sd = d.recall.mean(), d.recall.std()
    parts = []
    for t0, g in d.groupby("t0"):
        k = rw.index(t0) if t0 in rw else None
        if k is None or k < 2:
            continue
        g = g.copy()
        g["r1"] = g.id.map(rec(rw[k - 1]))
        g["r2"] = g.id.map(rec(rw[k - 2]))
        parts.append(g)
    if parts:
        e = pd.concat(parts)
        e = e[e.r1.notna() & e.r2.notna()].copy()
        z0, z1, z2 = e.rz, (e.r1 - mu) / sd, (e.r2 - mu) / sd
        e["lvl3"] = (z0 + z1 + z2) / 3
        e["slope3"] = (z0 - z2) / 2
        if len(e) > 1000:
            fit(e, "A", A.fe(e), c, "start")
    # ---- stopping (one wave and two consecutive waves)
    obs = p.groupby("wave").internet.apply(lambda s: s.notna().mean())
    iw = sorted(w for w in obs.index if obs[w] > 0.5)
    parts = []
    for j in range(len(iw) - 1):
        w, wn = iw[j], iw[j + 1]
        wn2 = iw[j + 2] if j + 2 < len(iw) else None
        if w not in rw or rw.index(w) < 2:
            continue
        k = rw.index(w)
        b = p[(p.wave == w) & (p.internet == 1) & (p.age >= 50) & (p.proxy.fillna(0) != 1)].copy()
        b["inet_n"] = b.id.map(ix.xs(wn, level="wave").internet)
        b["inet_n2"] = b.id.map(ix.xs(wn2, level="wave").internet) if wn2 else np.nan
        b["r1"] = b.id.map(rec(rw[k - 1]))
        b["r2"] = b.id.map(rec(rw[k - 2]))
        b["wv"] = w
        parts.append(b)
    if not parts:
        continue
    d = pd.concat(parts)
    d = d[d.inet_n.notna() & d.recall.notna() & d.r1.notna() & d.r2.notna()].copy()
    mu, sd = d.recall.mean(), d.recall.std()
    z0, z1, z2 = (d.recall - mu) / sd, (d.r1 - mu) / sd, (d.r2 - mu) / sd
    d["lvl3"] = (z0 + z1 + z2) / 3
    d["slope3"] = (z0 - z2) / 2
    d["age"] = (d.age - 65) / 10
    d["stop"] = (d.inet_n == 0).astype(float)
    fe = lambda e: pd.get_dummies(
        e.wv.astype(str) + (e.country.astype(str) if c == "SHARE" else ""),
        prefix="w",
        drop_first=True,
        dtype=float,
    )
    if len(d) > 1000:
        fit(d.copy(), "stop", fe(d), c, "stop")
    d2 = d[d.inet_n2.notna()].copy()
    d2["stop2"] = ((d2.inet_n == 0) & (d2.inet_n2 == 0)).astype(float)
    d2 = d2[(d2.stop2 == 1) | (d2.inet_n == 1)]  # exclude stop-then-return
    if len(d2) > 1000:
        fit(d2.copy(), "stop2", fe(d2), c, "stop2")
R = pd.DataFrame(rows)
R.to_csv("results_robust_sel.csv", index=False)
P = []
for (mo, t), g in R.groupby(["model", "term"]):
    g = g[g.se > 0]
    h = hksj(g.est, g.se)
    P.append(
        dict(
            model=mo,
            term=t,
            k=h["k"],
            est=h["est"],
            lo=h["hk_lo"] if h["k"] >= 3 else h["dl_lo"],
            hi=h["hk_hi"] if h["k"] >= 3 else h["dl_hi"],
        )
    )
P = pd.DataFrame(P)
P.to_csv("results_robust_sel_pooled.csv", index=False)
x = P.copy()
x[["est", "lo", "hi"]] = np.exp(x[["est", "lo", "hi"]])
print(x.round(2).to_string())
