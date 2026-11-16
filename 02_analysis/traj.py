"""Descriptive recall trajectories (Fig. 3a, Fig. 4a).
Age- and sex-standardised mean recall (SD units, cohort baseline scale) by wave position, by
(i) target-trial exposure (initiators vs non-initiators, t-1..t2) and
(ii) timing of first use among people offline at t0 (early t1, middle t2, late t3, never), t-1..t3.
Cohort means pooled by inverse-variance weighting.
"""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from build_trials import load
import within
from within import SCHED4

main = within.main


def adj_means(d, y, grp, levels):
    d = d[d[y].notna()].copy()
    if len(d) < 50:
        return {}
    X = [np.ones(len(d))]
    for g in levels[1:]:
        X.append((d[grp] == g).astype(float).values)
    a = (d.age.values - 65) / 10
    X += [a, a**2, d.female.fillna(0.5).values]
    for k in sorted(d.trial.unique())[1:]:
        X.append((d.trial == k).astype(float).values)
    X = np.column_stack(X)
    yy = d[y].values
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    res = yy - X @ beta
    n, p = X.shape
    XtX = np.linalg.pinv(X.T @ X)
    V = XtX @ (X.T * res**2) @ X @ XtX  # HC0
    out = {}
    for j, g in enumerate(levels):
        Xg = X.copy()
        Xg[:, 1 : len(levels)] = 0
        if j > 0:
            Xg[:, j] = 1
        xb = Xg.mean(0)
        out[g] = (float(xb @ beta), float(np.sqrt(xb @ V @ xb)), int((d[grp] == g).sum()))
    return out


rows = []
for c in SCHED4:
    p = load(c)
    ix = p.set_index(["id", "wave"])
    m = main[main.cohort == c]
    mu, sd = m.recall.mean(), m.recall.std()
    # (ii) within-adopter timing groups
    t = within.build(c)
    t["rawid"] = t.id.str.replace(c + "_", "", n=1)
    waves = set(p.wave.unique())
    parts = []
    for k, (t0, t1, t2, t3) in enumerate(SCHED4[c]):
        s = t[t.trial == k].copy()
        for pos, w in [("t−1", t0 - 1), ("t0", t0), ("t1", t1), ("t2", t2), ("t3", t3)]:
            if w not in waves:
                continue
            r = ix.xs(w, level="wave")[["recall", "recall_web"]]
            r = r.recall.where(r.recall_web.fillna(0) == 0).rename("r_" + pos)
            s = s.join(r, on="rawid")
        parts.append(s)
    t = pd.concat(parts, ignore_index=True)
    for pos in ["t−1", "t0", "t1", "t2", "t3"]:
        col = "r_" + pos
        if col not in t:
            continue
        t[col] = (t[col] - mu) / sd
        for g, (e, se, n) in adj_means(t, col, "grp", ["never", "early", "middle", "late"]).items():
            rows.append(dict(cohort=c, analysis="timing", pos=pos, grp=g, est=e, se=se, n=n))
    # (i) target-trial exposure
    d = main[main.cohort == c].copy()
    for pos, col in [("t−1", "recall_lag"), ("t0", "recall"), ("t1", "recall_1")]:
        d["z"] = (d[col] - mu) / sd
        for g, (e, se, n) in adj_means(
            d.assign(gA=d.A.map({0: "stay", 1: "start"})), "z", "gA", ["stay", "start"]
        ).items():
            rows.append(dict(cohort=c, analysis="tte", pos=pos, grp=g, est=e, se=se, n=n))
    d["z"] = d.Y_recall  # already standardised at t2
    for g, (e, se, n) in adj_means(
        d.assign(gA=d.A.map({0: "stay", 1: "start"})), "z", "gA", ["stay", "start"]
    ).items():
        rows.append(dict(cohort=c, analysis="tte", pos="t2", grp=g, est=e, se=se, n=n))
    print(c, "done", flush=True)

R = pd.DataFrame(rows)
pool = []
for (a, pos, g), x in R.groupby(["analysis", "pos", "grp"]):
    w = 1 / x.se**2
    e = (w * x.est).sum() / w.sum()
    se = np.sqrt(1 / w.sum())
    pool.append(
        dict(cohort="Pooled", analysis=a, pos=pos, grp=g, est=e, se=se, n=int(x.n.sum()), k=len(x))
    )
R = pd.concat([R, pd.DataFrame(pool)], ignore_index=True)
R.to_csv("results_trajectories.csv", index=False)
print(R[R.cohort == "Pooled"].round(3).to_string())
