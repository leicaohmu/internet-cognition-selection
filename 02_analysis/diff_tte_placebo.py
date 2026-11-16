"""Formal difference between target-trial and lead-1 placebo estimates, with person-clustered covariance."""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from aipw import design, nuisances
import placebo
from pool_hk import hksj

TRIM = 0.025


def pseudo(t, y):
    t = t.reset_index(drop=True)
    X = design(t)
    A = t.A.values.astype(int)
    Y = t[y].values.astype(float)
    b = set(np.unique(Y[~np.isnan(Y)])) <= {0.0, 1.0}
    e, pi1, pi0, m1, m0 = nuisances(X, A, Y, t.id.values, b)
    k = (e >= TRIM) & (e <= 1 - TRIM)
    e = np.clip(e, 0.01, 0.99)
    pi1, pi0 = np.clip(pi1, 0.05, 1), np.clip(pi0, 0.05, 1)
    R = ~np.isnan(Y)
    Yf = np.nan_to_num(Y)
    d = (m1 + A * R * (Yf - m1) / (e * pi1)) - (m0 + (1 - A) * R * (Yf - m0) / ((1 - e) * pi0))
    return d[k], t.id.values[k]


main = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
main["id"] = main.cohort + "_" + main.id
rows = []
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    pl = placebo.build(c)
    for y, f in [
        ("Y_recall", lambda d: d),
        ("Y_iadl", lambda d: d[d.iadl_any == 0]),
        ("Y_dep", lambda d: d),
    ]:
        if f(pl)[y].notna().sum() < 200 or f(pl).A.sum() < 50:
            continue
        dm, im = pseudo(f(main[main.cohort == c]), y)
        dp, ip = pseudo(f(pl), y)
        em, ep = dm.mean(), dp.mean()
        a = pd.Series((dm - em) / len(dm)).groupby(im).sum()
        b = pd.Series((dp - ep) / len(dp)).groupby(ip).sum()
        s = a.sub(b, fill_value=0)
        se = np.sqrt((s**2).sum())
        shared = len(set(im) & set(ip))
        rows.append(
            dict(cohort=c, y=y, tte=em, placebo=ep, est=em - ep, se=se, shared_persons=shared)
        )
        print(
            c,
            y,
            f"TTE {em:+.4f} placebo {ep:+.4f} diff {em-ep:+.4f} ({se:.4f}) shared={shared}",
            flush=True,
        )
d = pd.DataFrame(rows)
d.to_csv("results_tte_minus_placebo.csv", index=False)
for y, g in d.groupby("y"):
    r = hksj(g.est, g.se)
    print(
        "POOLED",
        y,
        f"{r['est']:+.4f} DL({r['dl_lo']:+.4f},{r['dl_hi']:+.4f}) HK({r['hk_lo']:+.4f},{r['hk_hi']:+.4f})",
    )
