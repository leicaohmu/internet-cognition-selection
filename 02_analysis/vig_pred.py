"""Predictors of starting weekly vigorous activity (mirror of Fig. 6b) among inactive adults aged >=50 (non-proxy) at wave w with known status at the
next wave; same predictors as for internet use (internet use itself added), wave (and country) fixed effects, person-clustered SEs, HKSJ pooling.
"""

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
]
rows = []
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    p = load(c)
    p = p[p.iwstat == 1]
    obs = p.groupby("wave").vig_weekly.apply(lambda s: s.notna().mean())
    waves = sorted(w for w in obs.index if obs[w] > 0.5)
    ix = p.set_index(["id", "wave"])
    allw = sorted(p.wave.unique())
    parts = []
    for k in range(len(waves) - 1):
        w, wn = waves[k], waves[k + 1]
        prev = [x for x in allw if x < w]
        b = p[(p.wave == w) & (p.vig_weekly == 0) & (p.age >= 50) & (p.proxy.fillna(0) != 1)].copy()
        b["v_n"] = b.id.map(ix.xs(wn, level="wave").vig_weekly)
        b["recall_p"] = b.id.map(ix.xs(prev[-1], level="wave").recall) if prev else np.nan
        b = b[b.v_n.notna()]
        b["start"] = (b.v_n == 1).astype(float)
        b["wv"] = w
        parts.append(b)
    if not parts:
        print(c, "no waves")
        continue
    d = pd.concat(parts).reset_index(drop=True)
    mu, sd = d.recall.mean(), d.recall.std()
    d["rz"] = (d.recall - mu) / sd
    d["rzp"] = (d.recall_p - mu) / sd
    d["lvl"] = (d.rz + d.rzp) / 2
    d["drz"] = d.rz - d.rzp
    d["age"] = (d.age - 65) / 10
    e = d[d.drz.notna()].copy()
    if len(e) < 500:
        print(c, "too few", len(e))
        continue
    for v in ["edu", "wealth_q", "ncond", "dep", "age"]:
        e[v] = (e[v] - e[v].mean()) / e[v].std()
    fe = pd.get_dummies(
        e.wv.astype(str) + (e.country.astype(str) if c == "SHARE" else ""),
        prefix="w",
        drop_first=True,
        dtype=float,
    )
    X = pd.concat([design(e, [v for v in PRED if v in e]), fe], axis=1)
    m = _glm(e.start, X, pd.factorize(e.id)[0])
    for v in PRED:
        if v in m.params:
            rows.append(
                dict(
                    cohort=c,
                    term=v,
                    est=m.params[v],
                    se=m.bse[v],
                    n=len(e),
                    events=int(e.start.sum()),
                )
            )
    print(c, len(e), "start rate", round(e.start.mean(), 3), flush=True)
R = pd.DataFrame(rows)
R.to_csv("results_vig_pred.csv", index=False)
P = []
for t, g in R.groupby("term"):
    g = g[g.se > 0]
    h = hksj(g.est, g.se)
    P.append(
        dict(
            term=t,
            k=h["k"],
            est=h["est"],
            lo=h["hk_lo"] if h["k"] >= 3 else h["dl_lo"],
            hi=h["hk_hi"] if h["k"] >= 3 else h["dl_hi"],
        )
    )
P = pd.DataFrame(P)
P.to_csv("results_vig_pred_pooled.csv", index=False)
x = P.copy()
x[["est", "lo", "hi"]] = np.exp(x[["est", "lo", "hi"]])
print(x.round(2).to_string())
