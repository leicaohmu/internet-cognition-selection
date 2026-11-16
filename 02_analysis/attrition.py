"""Round-2 review: (1) inverse-probability-of-censoring weights for person-trials lost before t1 (death, non-response,
unknown internet status); (2) death-inclusive memory outcome (death or recall in the cohort's lowest baseline quintile).
"""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from build_trials import load, SCHED, COV, LAG, EXTRA
from aipw import design, nuisances, cluster_var, _clf, aipw
from pool_hk import hksj

TRIM = 0.025


def eligible(c):
    """All eligible person-trials at t0 with an indicator for being analysable at t1."""
    p = load(c)
    ix = p.set_index(["id", "wave"])
    out = []
    for k, (t0, t1, t2) in enumerate(SCHED[c]):
        b = p[
            (p.wave == t0)
            & (p.iwstat == 1)
            & (p.internet == 0)
            & (p.age >= 50)
            & (p.proxy.fillna(0) != 1)
        ].copy()
        s1 = ix.xs(t1, level="wave")[["iwstat", "internet"]].add_suffix("_1")
        b = b.join(s1, on="id")
        b["R1"] = ((b.iwstat_1 == 1) & b.internet_1.notna()).astype(int)
        b["trial"] = k
        b["t0"] = t0
        out.append(b)
    e = pd.concat(out, ignore_index=True)
    e["cohort"] = c
    e["key"] = c + "_" + e.id.astype(str) + "_" + e.t0.astype(str)
    return e


def censor_weights(e):
    X = design(e.assign(A=0))
    R = e.R1.values
    pr = np.zeros(len(e))
    for tr, te in GroupKFold(5).split(X, R, e.id.values):
        pr[te] = _clf("hgb").fit(X.iloc[tr], R[tr]).predict_proba(X.iloc[te])[:, 1]
    return pd.Series(1 / np.clip(pr, 0.05, 1), index=e.key)


def weighted_aipw(t, y, w):
    t = t.reset_index(drop=True)
    X = design(t)
    A = t.A.values.astype(int)
    Y = t[y].values.astype(float)
    binary = set(np.unique(Y[~np.isnan(Y)])) <= {0.0, 1.0}
    e, pi1, pi0, m1, m0 = nuisances(X, A, Y, t.id.values, binary)
    keep = (e >= TRIM) & (e <= 1 - TRIM)
    e = np.clip(e, 0.01, 0.99)
    pi1, pi0 = np.clip(pi1, 0.05, 1), np.clip(pi0, 0.05, 1)
    R = ~np.isnan(Y)
    Yf = np.nan_to_num(Y)
    d = (m1 + A * R * (Yf - m1) / (e * pi1)) - (m0 + (1 - A) * R * (Yf - m0) / ((1 - e) * pi0))
    d, ww, ids = d[keep], w[keep], t.id.values[keep]
    est = np.sum(ww * d) / ww.sum()
    phi = ww * (d - est) / ww.mean()
    return dict(est=est, se=np.sqrt(cluster_var(phi, ids)), n=int(keep.sum()))


if __name__ == "__main__":
    main = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
    main["key"] = main.cohort + "_" + main.id.astype(str) + "_" + main.t0.astype(str)
    main["id"] = main.cohort + "_" + main.id
    rows = []
    for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
        e = eligible(c)
        w = censor_weights(e)
        t = main[main.cohort == c].copy()
        t["w"] = t.key.map(w).fillna(1.0).values
        print(
            c,
            "eligible",
            len(e),
            "analysable",
            int(e.R1.sum()),
            "weight mean/max among analysable",
            round(t.w.mean(), 2),
            round(t.w.max(), 1),
            flush=True,
        )
        for y, f in [
            ("Y_recall", lambda d: d),
            ("Y_iadl", lambda d: d[d.iadl_any == 0]),
            ("Y_dep", lambda d: d),
        ]:
            d = f(t)
            r = weighted_aipw(d, y, d.w.values)
            r.update(cohort=c, y=y, analysis="ipcw_t1")
            rows.append(r)
            print("  ipcw", y, f"{r['est']:+.4f} ({r['se']:.4f})", flush=True)
        # death-inclusive memory: death or recall in lowest baseline quintile
        if c != "SHARE":
            q20 = t.recall_z.quantile(0.2) if "recall_z" in t else None
            q20 = (
                (t.recall.quantile(0.2) - t.recall.mean()) / t.recall.std() if q20 is None else q20
            )
            low = (t.Y_recall <= q20).astype(float).where(t.Y_recall.notna())
            t["Y_lowrec"] = low
            t["Y_lowrec_death"] = np.where(t.Y_death == 1, 1.0, low)
            t.loc[t.Y_death.isna(), "Y_lowrec_death"] = np.nan
            for y in ["Y_lowrec", "Y_lowrec_death"]:
                d = t[t.Y_death.notna()]
                r = aipw(d, y, trim=TRIM)
                r.update(analysis="death_composite")
                rows.append(r)
                print(
                    "  ",
                    y,
                    f"RD {r['est']:+.4f} ({r['lo']:+.4f},{r['hi']:+.4f}) RR {r.get('rr', np.nan):.3f} ({r.get('rr_lo', np.nan):.3f}-{r.get('rr_hi', np.nan):.3f})",
                    flush=True,
                )
    res = pd.DataFrame(rows)
    res.to_csv("results_attrition.csv", index=False)
    for (a, y), g in res.groupby(["analysis", "y"]):
        r = hksj(g.est, g.se)
        line = f"POOLED {a} {y} k={len(g)} {r['est']:+.4f} DL({r['dl_lo']:+.4f},{r['dl_hi']:+.4f}) HK({r['hk_lo']:+.4f},{r['hk_hi']:+.4f})"
        if "se_lrr" in g and g.se_lrr.notna().all():
            rl = hksj(np.log(g.rr), g.se_lrr)
            line += (
                f" RR {np.exp(rl['est']):.3f} ({np.exp(rl['dl_lo']):.3f}-{np.exp(rl['dl_hi']):.3f})"
            )
        print(line)
