"""Staggered-adoption event study (Callaway & Sant'Anna 2021 style).
Unit: person aged >=50, non-proxy, offline at first observed internet wave.
Group g = first wave (position) online. Universal base period g-1.
ATT(g,e) = E[Y_{g+e}-Y_{g-1} | G=g] - E[Y_{g+e}-Y_{g-1} | not yet treated by max(g+e, g-1), X]
X: age, age^2, female, education, wealth quintile at g-1 (outcome regression adjustment on controls).
Aggregated to event time e by treated-count weights; person-level Poisson bootstrap (B=200).
Event time counted in internet-observed waves (~2 years in HRS/ELSA/CHARLS, 2-4 years in SHARE).
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
from newout import panel

EV = range(-4, 4)
B = int(os.environ.get("B", 200))
OUTS = os.environ.get("ES_OUTS", "z_recall,z_gcog,srh").split(",")
rng = np.random.default_rng(1)
ANT = int(os.environ.get("ANT", 0))  # anticipation: base period g-1-ANT


def wide(p, col, waves):
    return p.pivot_table(index="id", columns="wave", values=col, aggfunc="first").reindex(
        columns=waves
    )


def run(c):
    p = panel(c)
    if os.environ.get("BEH"):
        p["internet"] = p[os.environ["BEH"]]
    p = p[p.iwstat == 1] if "iwstat" in p else p
    obs = p.groupby("wave").internet.apply(lambda s: s.notna().mean())
    waves = sorted(w for w in obs.index if obs[w] > 0.5)
    pos = {w: i for i, w in enumerate(waves)}
    I = wide(p, "internet", waves)
    # first observed internet wave must be offline, age >=50 then
    first_obs = I.notna().values.argmax(1)
    I0 = I.values[np.arange(len(I)), first_obs]
    keep = I0 == 0
    I = I[keep]
    Iv = I.values.copy()
    # adoption position: first 1 (absorbing); NaN gaps tolerated only if the preceding observed wave is 0
    G = np.full(len(I), np.inf)
    for i, row in enumerate(Iv):
        ones = np.where(row == 1)[0]
        if len(ones):
            k = ones[0]
            prev = np.where(~np.isnan(row[:k]))[0]
            G[i] = (
                k if (len(prev) and prev[-1] == k - 1) else -1
            )  # -1 = adoption with unobserved gap: drop
    # status "offline through position t": cumulative max of internet==1
    on = np.nan_to_num(Iv, nan=0)
    everon = np.maximum.accumulate(on, axis=1)
    known = np.maximum.accumulate(~np.isnan(Iv), axis=1)
    ids = I.index
    cov = {}
    for x in ["age", "female", "edu", "wealth_q", "proxy"]:
        cov[x] = wide(p, x, waves).reindex(ids).values
    out = []
    for ycol in OUTS:
        Y = wide(p, ycol, waves).reindex(ids).values
        cells = []
        for g in range(1 + ANT, len(waves)):
            for e in EV:
                t, b = g + e, g - 1 - ANT
                if t < 0 or t >= len(waves) or t == b:
                    continue
                last = max(t, b)
                age_b = cov["age"][:, b]
                base = (
                    (~np.isnan(Y[:, t]))
                    & (~np.isnan(Y[:, b]))
                    & (age_b >= 50)
                    & (np.nan_to_num(cov["proxy"][:, b]) == 0)
                )
                tr = base & (G == g)
                ct = (
                    base
                    & (G > last)
                    & (everon[:, last] == 0)
                    & known[:, last]
                    & (~np.isnan(Iv[:, last]) | (G == np.inf))
                )
                ct &= ~np.isnan(Iv[:, last])
                if tr.sum() < 30 or ct.sum() < 100:
                    continue
                X = np.column_stack(
                    [
                        np.ones(len(ids)),
                        (age_b - 65) / 10,
                        ((age_b - 65) / 10) ** 2,
                        np.nan_to_num(cov["female"][:, b], nan=0.5),
                        np.nan_to_num(cov["edu"][:, b], nan=np.nanmean(cov["edu"][:, b])),
                        np.nan_to_num(cov["wealth_q"][:, b], nan=2),
                    ]
                )
                dY = Y[:, t] - Y[:, b]
                cells.append(dict(g=g, e=e, tr=np.where(tr)[0], ct=np.where(ct)[0], X=X, dY=dY))
        if not cells:
            continue

        def est(w):
            res = {}
            for cl in cells:
                tr, ct = cl["tr"], cl["ct"]
                wt, wc = w[tr], w[ct]
                if wt.sum() == 0:
                    continue
                Xc = cl["X"][ct] * np.sqrt(wc)[:, None]
                yc = cl["dY"][ct] * np.sqrt(wc)
                beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
                att = np.sum(wt * (cl["dY"][tr] - cl["X"][tr] @ beta)) / wt.sum()
                res.setdefault(cl["e"], []).append((att, wt.sum()))
            return {e: sum(a * n for a, n in v) / sum(n for a, n in v) for e, v in res.items()}

        nT = {e: sum(len(cl["tr"]) for cl in cells if cl["e"] == e) for e in EV}

        def derived(r):
            # linear pre-trend through e<0 (reference e=-1-ANT fixed at 0), extrapolated to e>=0;
            # post_avg = treated-weighted mean of e=0..2; *_adj = deviation from the extrapolated pre-trend
            r = dict(r)
            r[-1 - ANT] = 0.0
            pre = [e for e in r if e < 0]
            post = [e for e in r if 0 <= e <= 2]
            if len(pre) >= 3:
                w = np.array([nT.get(e, 0) or max(nT.values()) for e in pre], float)
                b = np.polyfit(pre, [r[e] for e in pre], 1, w=np.sqrt(w))
                for e in post:
                    r[f"adj{e}"] = r[e] - np.polyval(b, e)
                r["slope_pre"] = b[0]
            wp = np.array([nT[e] for e in post], float)
            r["post_avg"] = float(np.dot(wp, [r[e] for e in post]) / wp.sum())
            if all(f"adj{e}" in r for e in post):
                r["post_avg_adj"] = float(np.dot(wp, [r[f"adj{e}"] for e in post]) / wp.sum())
            r["pre_avg"] = float(np.mean([r[e] for e in pre if e != -1 - ANT]))
            return r

        pt = derived(est(np.ones(len(ids))))
        bs = [derived(est(rng.poisson(1, len(ids)).astype(float))) for _ in range(B)]
        for e in sorted(pt, key=str):
            se = np.nanstd([b_[e] for b_ in bs if e in b_], ddof=1)
            ntr = nT.get(e, 0) if isinstance(e, (int, np.integer)) else 0
            out.append(dict(cohort=c, y=ycol, e=e, est=pt[e], se=se, n_treated=ntr))
            print(c, ycol, e, f"{pt[e]:+.3f} ({se:.3f}) nT={ntr}", flush=True)
    pd.DataFrame(out).to_csv(
        f'results_es{ANT or ""}{"_" + os.environ["BEH"] if os.environ.get("BEH") else ""}_{c}.csv',
        index=False,
    )


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run(c)
