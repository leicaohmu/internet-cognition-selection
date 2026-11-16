"""Predictors of stopping internet use (mirror of Fig. 6b): among users aged >=50 (non-proxy) at wave w with known status at the next
internet-observed wave, stop = reported non-use at that wave. Mutually adjusted logistic model with the same standardized predictors as for starting;
memory = mean of standardized recall at w and the previous wave, recent change = their difference; wave (and country in SHARE) fixed effects;
person-clustered SEs; pooled with HKSJ."""

import os

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
from build_trials import load
from anatomy import _glm, design
from pool_hk import hksj

PRED = [
    "lvl",
    "drz",
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
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    p = load(c)
    p = p[p.iwstat == 1]
    obs = p.groupby("wave").internet.apply(lambda s: s.notna().mean())
    waves = sorted(w for w in obs.index if obs[w] > 0.5)
    ix = p.set_index(["id", "wave"])
    allw = sorted(p.wave.unique())
    parts = []
    for k in range(len(waves) - 1):
        w, wn = waves[k], waves[k + 1]
        prev = [x for x in allw if x < w]
        b = p[(p.wave == w) & (p.internet == 1) & (p.age >= 50) & (p.proxy.fillna(0) != 1)].copy()
        b = b.join(ix.xs(wn, level="wave")[["internet"]].add_suffix("_n"), on="id")
        if prev:
            b = b.join(ix.xs(prev[-1], level="wave")[["recall"]].add_suffix("_p"), on="id")
        else:
            b["recall_p"] = np.nan
        b = b[b.internet_n.notna()]
        b["stop"] = (b.internet_n == 0).astype(float)
        b["wv"] = w
        parts.append(b)
    d = pd.concat(parts).reset_index(drop=True)
    mu, sd = d.recall.mean(), d.recall.std()
    d["rz"] = (d.recall - mu) / sd
    d["rzp"] = (d.recall_p - mu) / sd
    d["lvl"] = (d.rz + d.rzp) / 2
    d["drz"] = d.rz - d.rzp
    d["age"] = (d.age - 65) / 10
    e = d[d.drz.notna()].copy()
    if len(e) < 500:  # no previous recall (e.g. CHARLS first waves): skip
        print(c, "too few with previous recall", len(e))
        continue
    for v in ["edu", "wealth_q", "ncond", "dep", "age"]:
        e[v] = (e[v] - e[v].mean()) / e[v].std()
    fe = pd.get_dummies(
        e.wv.astype(str) + (e.country.astype(str) if c == "SHARE" and "country" in e else ""),
        prefix="w",
        drop_first=True,
        dtype=float,
    )
    X = pd.concat([design(e, [v for v in PRED if v in e]), fe], axis=1)
    m = _glm(e.stop, X, pd.factorize(e.id)[0])
    for v in PRED:
        if v in m.params:
            rows.append(
                dict(
                    cohort=c,
                    term=v,
                    est=m.params[v],
                    se=m.bse[v],
                    n=len(e),
                    stops=int(e.stop.sum()),
                )
            )
    print(c, len(e), "stop rate", round(e.stop.mean(), 3), flush=True)
R = pd.DataFrame(rows)
R.to_csv("results_stop_pred.csv", index=False)
P = []
for t, g in R.groupby("term"):
    g = g[g.se > 0]
    h = hksj(g.est, g.se)
    P.append(dict(term=t, k=h["k"], est=h["est"], lo=h["hk_lo"], hi=h["hk_hi"], I2=h["I2"]))
P = pd.DataFrame(P)
P.to_csv("results_stop_pred_pooled.csv", index=False)
x = P.copy()
x[["est", "lo", "hi"]] = np.exp(x[["est", "lo", "hi"]])
print(x.round(2).to_string())
