"""Sensitivity to the target population and calendar era.
One nuisance fit per cohort x design x outcome, then:
  trim   : AIPW in the 0.025-0.975 overlap population (primary, reproduces earlier results)
  untrim : AIPW in everyone (e clipped at 0.01)
  ato    : augmented overlap-weighted estimator (Mao, Li & Greene 2019), weights e(1-e), no trimming
  era    : trimmed AIPW pseudo-outcomes averaged within baseline year < 2010 vs >= 2010
Designs: main target trial, lead-1 placebo, within-adopter early vs late.
"""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import sys
import time
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from aipw import design, nuisances, cluster_var
import placebo
import within

TRIM, TRUNC, ERA = 0.025, 0.01, 2010
OUTS = [("Y_recall", lambda d: d), ("Y_iadl", lambda d: d[d.iadl_any == 0]), ("Y_dep", lambda d: d)]


def est_block(A, Y, e, pi1, pi0, m1, m0, ids, keep):
    A, Y, e, pi1, pi0, m1, m0, ids = (v[keep] for v in (A, Y, e, pi1, pi0, m1, m0, ids))
    R = ~np.isnan(Y)
    Yf = np.nan_to_num(Y)
    psi1 = m1 + A * R * (Yf - m1) / (e * pi1)
    psi0 = m0 + (1 - A) * R * (Yf - m0) / ((1 - e) * pi0)
    d = psi1 - psi0
    est = d.mean()
    se = np.sqrt(cluster_var(d - est, ids))
    return est, se, len(A), int(A.sum()), d, ids


def run_cell(t, y, design_name):
    t = t.reset_index(drop=True)
    X = design(t)
    A = t.A.values.astype(int)
    Y = t[y].values.astype(float)
    binary = set(np.unique(Y[~np.isnan(Y)])) <= {0.0, 1.0}
    e_raw, pi1, pi0, m1, m0 = nuisances(X, A, Y, t.id.values, binary)
    e = np.clip(e_raw, TRUNC, 1 - TRUNC)
    pi1, pi0 = np.clip(pi1, 0.05, 1), np.clip(pi0, 0.05, 1)
    ids = t.id.values
    out = []
    base = dict(cohort=t.cohort.iloc[0], design=design_name, y=y)
    keep_t = (e_raw >= TRIM) & (e_raw <= 1 - TRIM)
    est, se, n, n1, d, idk = est_block(A, Y, e, pi1, pi0, m1, m0, ids, keep_t)
    out.append(dict(base, method="trim", est=est, se=se, n=n, n_A1=n1))
    yr = t.year.values[keep_t]
    for lab, msk in [("era<2010", yr < ERA), ("era>=2010", yr >= ERA)]:
        if msk.sum() > 100 and A[keep_t][msk].sum() >= 30:
            dd = d[msk]
            m = dd.mean()
            out.append(
                dict(
                    base,
                    method=lab,
                    est=m,
                    se=np.sqrt(cluster_var(dd - m, idk[msk])),
                    n=int(msk.sum()),
                    n_A1=int(A[keep_t][msk].sum()),
                )
            )
    est, se, n, n1, _, _ = est_block(A, Y, e, pi1, pi0, m1, m0, ids, np.ones(len(A), bool))
    out.append(dict(base, method="untrim", est=est, se=se, n=n, n_A1=n1))
    # augmented overlap weights
    R = ~np.isnan(Y)
    Yf = np.nan_to_num(Y)
    h = e * (1 - e)
    num = h * (m1 - m0) + A * (1 - e) * R * (Yf - m1) / pi1 - (1 - A) * e * R * (Yf - m0) / pi0
    tau = num.sum() / h.sum()
    phi = (num - h * tau) / h.mean()
    out.append(
        dict(
            base,
            method="ato",
            est=tau,
            se=np.sqrt(cluster_var(phi, ids)),
            n=len(A),
            n_A1=int(A.sum()),
        )
    )
    return out


if __name__ == "__main__":
    main = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
    main["id"] = main.cohort + "_" + main.id
    rows = []
    for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
        pl = placebo.build(c)
        pl["year"] = pl["year"] if "year" in pl else np.nan
        wi = within.build(c)
        wi = wi[wi.grp.isin(["early", "late"])].copy()
        wi["A"] = (wi.grp == "early").astype(int)
        for name, tt in [("main", main[main.cohort == c]), ("placebo", pl), ("early_vs_late", wi)]:
            for y, f in OUTS:
                d = f(tt)
                if d[y].notna().sum() < 200 or d.A.sum() < 40 or (1 - d.A).sum() < 40:
                    continue
                t0 = time.time()
                r = run_cell(d, y, name)
                rows += r
                print(
                    c,
                    name,
                    y,
                    " | ".join(
                        f"{x['method']} {x['est']:+.4f}({x['se']:.4f}) n={x['n']}" for x in r
                    ),
                    f"[{time.time()-t0:.0f}s]",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv("results_sens_overlap.csv", index=False)
    print("done")
