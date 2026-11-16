"""Cross-country selection gradient.
Units: HRS (US), ELSA (England), CHARLS (China), each SHARE country; LASI (India) for the cross-sectional gap only.
Per unit:
  prev   share online among offline-or-online respondents aged >=50 at the trial baseline waves
  xs     cross-sectional user - non-user recall gap (SD), adjusted for age, age^2, sex, education
  pre    initiator - non-initiator recall gap at t0 (before exposure), AIPW adjusting for pre-t0 history
  tte    target-trial estimate for recall at t2 (AIPW, full t0 covariates)
Nuisance models are parametric (logistic / linear with imputation) because many SHARE country samples are small.
Meta-regression: estimate on logit(prev); random-effects weights (DL tau^2), Knapp-Hartung SE.
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import numpy as np
import pandas as pd
from scipy import stats
from build_trials import load, SCHED
from aipw import aipw, nuisances, summarise

main = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
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


def pre_gap(t, mu, sd):
    t = t[t.recall.notna()].reset_index(drop=True)
    X = t[[c for c in PRE if t[c].notna().any()]].astype(float)
    A = t.A.values.astype(int)
    Y = ((t.recall - mu) / sd).values
    e, pi1, pi0, m1, m0 = nuisances(X, A, Y, t.id.values, False, kind="lr")
    keep = (e >= 0.025) & (e <= 0.975)
    e = np.clip(e, 0.01, 0.99)
    A, Y, e, m1, m0, ids = (v[keep] for v in (A, Y, e, m1, m0, t.id.values))
    psi1 = m1 + A * (Y - m1) / e
    psi0 = m0 + (1 - A) * (Y - m0) / (1 - e)
    return summarise(psi1, psi0, ids, False)


def xs_gap(p, waves, mu, sd):
    d = p[
        p.wave.isin(waves)
        & (p.age >= 50)
        & p.internet.notna()
        & p.recall.notna()
        & (p.recall_web.fillna(0) == 0)
    ]
    d = d[d.proxy.fillna(0) == 0]
    a = (d.age - 65) / 10
    X = np.column_stack(
        [np.ones(len(d)), d.internet, a, a**2, d.female.fillna(0.5), d.edu.fillna(d.edu.mean())]
        + [(d.wave == w).astype(float) for w in waves[1:]]
    )
    y = ((d.recall - mu) / sd).values
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    # person-clustered sandwich
    XtXi = np.linalg.pinv(X.T @ X)
    S = pd.DataFrame(X * r[:, None]).groupby(d.id.values).sum().values
    V = XtXi @ S.T @ S @ XtXi
    return dict(est=b[1], se=np.sqrt(V[1, 1]), n=len(d))


def unit_rows(c, label, t, p):
    rows = []
    waves = sorted({s[0] for s in SCHED[c]}) if c != "LASI" else [1]
    base = p[p.wave.isin(waves) & (p.age >= 50) & p.internet.notna() & (p.proxy.fillna(0) == 0)]
    prev = base.internet.mean()
    m = main[main.cohort == c] if c != "LASI" else None
    mu, sd = (
        (m.recall.mean(), m.recall.std())
        if m is not None
        else (base.recall.mean(), base.recall.std())
    )
    r = xs_gap(p, waves, mu, sd)
    rows.append(dict(unit=label, cohort=c, design="xs", prev=prev, n_base=len(base), **r))
    if t is not None and t.A.sum() >= 40 and (1 - t.A).sum() >= 40:
        r = aipw(t, "Y_recall", kind="lr", trim=0.025)
        rows.append(
            dict(
                unit=label,
                cohort=c,
                design="tte",
                prev=prev,
                n_base=len(base),
                est=r["est"],
                se=r["se"],
                n=r["n"],
            )
        )
        r = pre_gap(t, mu, sd)
        rows.append(
            dict(
                unit=label,
                cohort=c,
                design="pre",
                prev=prev,
                n_base=len(base),
                est=r["est"],
                se=r["se"],
                n=len(t),
            )
        )
    for x in rows[-3:]:
        print(label, x["design"], f"prev={prev:.2f} {x['est']:+.3f} ({x['se']:.3f})", flush=True)
    return rows


def metareg(d):
    y, v, x = d.est.values, d.se.values**2, np.log(d.prev / (1 - d.prev)).values
    X = np.column_stack([np.ones(len(y)), x])
    k, q = X.shape
    W = np.diag(1 / v)
    b = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    Q = float((y - X @ b) @ W @ (y - X @ b))
    P = W - W @ X @ np.linalg.solve(X.T @ W @ X, X.T @ W)
    tau2 = max(0, (Q - (k - q)) / np.trace(P))
    W = np.diag(1 / (v + tau2))
    XtWXi = np.linalg.inv(X.T @ W @ X)
    b = XtWXi @ X.T @ W @ y
    res = y - X @ b
    s2 = max(1, float(res @ W @ res) / (k - q))  # Knapp-Hartung
    se = np.sqrt(np.diag(XtWXi) * s2)
    tq = stats.t.ppf(0.975, k - q)
    return dict(
        k=k,
        intercept=b[0],
        slope=b[1],
        slope_se=se[1],
        lo=b[1] - tq * se[1],
        hi=b[1] + tq * se[1],
        p=2 * stats.t.sf(abs(b[1] / se[1]), k - q),
        tau2=tau2,
    )


if __name__ == "__main__":
    rows = []
    for c in ["HRS", "ELSA", "CHARLS", "SHARE"]:
        p = load(c)
        t = main[main.cohort == c]
        if c == "SHARE":
            for ctry, tc in t.groupby("country"):
                rows += unit_rows(c, f"SHARE-{ctry}", tc.copy(), p[p.country == ctry])
        else:
            rows += unit_rows(c, c, t.copy(), p)
    if os.path.exists("panel_LASI.csv.gz"):
        try:
            rows += unit_rows("LASI", "LASI", None, load("LASI"))
        except Exception as ex:
            print("LASI skipped", ex)
    R = pd.DataFrame(rows)
    R.to_csv("results_gradient.csv", index=False)
    M = [dict(design=g, **metareg(d)) for g, d in R.groupby("design") if len(d) >= 5]
    M = pd.DataFrame(M)
    M.to_csv("results_gradient_metareg.csv", index=False)
    print(M.round(3).to_string())
