"""Anatomy of selection into internet use (v1.1).
Among target-trial eligibles (offline, aged >=50 at t0):
 (a) probability of starting by t1 across quintiles of recall at t0 (age-sex standardized; OR vs bottom quintile)
 (b) probability of starting by prior recall change (t-1 -> t0), holding recall at t0 and covariates fixed
 (c) Gelbach decomposition of the initiator - non-initiator recall gap at t0 into covariate groups
 (d) mutually adjusted predictors of starting (standardized odds ratios)
Cohort estimates pooled by random effects (HKSJ)."""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression


class _Fit:
    def __init__(self, X, y, groups):
        X = X.astype(float)
        self.cols = list(X.columns)
        lr = LogisticRegression(penalty=None, max_iter=5000, solver="lbfgs").fit(X.values, y)
        b = np.r_[lr.intercept_, lr.coef_[0]]
        Xc = np.column_stack([np.ones(len(X)), X.values])
        p = 1 / (1 + np.exp(-Xc @ b))
        W = p * (1 - p)
        H = np.linalg.pinv((Xc * W[:, None]).T @ Xc)
        S = pd.DataFrame(Xc * (y - p)[:, None]).groupby(groups).sum().values
        V = H @ (S.T @ S) @ H
        self.params = pd.Series(b, index=["const"] + self.cols)
        self.bse = pd.Series(np.sqrt(np.diag(V)), index=self.params.index)
        self._b = b

    def predict(self, X):
        Xc = np.column_stack([np.ones(len(X)), X[self.cols].values.astype(float)])
        return 1 / (1 + np.exp(-Xc @ self._b))


def _glm(y, X, groups):
    return _Fit(X, np.asarray(y, float), groups)


from pool_hk import hksj

T = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
COH = ["HRS", "ELSA", "SHARE", "CHARLS"]
rng = np.random.default_rng(7)
GROUPS = {
    "Age and sex": ["age", "age2", "female"],
    "Education, wealth and work": ["edu", "wealth_q", "work"],
    "Health": ["srh_poor", "ncond", "adl_any", "iadl_any", "dep"],
    "Household and spouse": ["partnered", "live_alone", "sp_internet"],
    "Lifestyle": ["smoke", "drink", "vig_weekly"],
    "Earlier memory (t−1)": ["recall_lag"],
}


def prep(c):
    d = T[T.cohort == c].copy()
    mu, sd = d.recall.mean(), d.recall.std()
    d["rz"] = (d.recall - mu) / sd
    d["rz_lag"] = (d.recall_lag - mu) / sd
    d["drz"] = d.rz - d.rz_lag
    d["age2"] = ((d.age - 65) / 10) ** 2
    d["age"] = (d.age - 65) / 10
    d["recall_lag"] = d.rz_lag
    d["lvl"] = (d.rz + d.rz_lag) / 2
    return d


def design(d, cols):
    X = []
    for v in cols:
        x = d[v].astype(float)
        if x.isna().any():
            X.append(x.fillna(x.median()).rename(v))
            X.append(x.isna().astype(float).rename(v + "_m"))
        else:
            X.append(x.rename(v))
    X = pd.concat(X, axis=1)
    X = X.loc[:, X.std() > 0]
    return X


def fe(d):
    return pd.get_dummies(
        d.trial.astype(str) + (d.country.astype(str) if d.cohort.iloc[0] == "SHARE" else ""),
        prefix="t",
        drop_first=True,
        dtype=float,
    )


# ---------------------------------------------------------------- (a) quintiles, (b) change, (d) predictors
def adoption_models(c):
    d = prep(c)
    d = d[d.rz.notna()]
    d["q"] = pd.qcut(d.rz.rank(method="first"), 5, labels=False)
    out = []
    X = pd.concat(
        [
            pd.get_dummies(d.q, prefix="q", drop_first=True, dtype=float),
            design(d, ["age", "age2", "female"]),
            fe(d),
        ],
        axis=1,
    )
    m = _glm(d.A, X, pd.factorize(d.id)[0])
    for k in range(1, 5):
        out.append(
            dict(
                cohort=c,
                part="quintile",
                term=f"Q{k + 1}",
                est=m.params[f"q_{k}"],
                se=m.bse[f"q_{k}"],
            )
        )
    # standardized probabilities by quintile (for plotting)
    for k in range(5):
        Xk = X.copy()
        Xk[[f"q_{j}" for j in range(1, 5)]] = 0
        if k:
            Xk[f"q_{k}"] = 1
        out.append(
            dict(
                cohort=c,
                part="prob_quintile",
                term=f"Q{k + 1}",
                est=float(m.predict(Xk).mean()),
                se=np.nan,
            )
        )
    # mutually adjusted predictors, standardized
    e = d[d.drz.notna()].copy()
    preds = [
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
    for v in ["edu", "wealth_q", "ncond", "dep", "age"]:
        e[v] = (e[v] - e[v].mean()) / e[v].std()
    X = pd.concat([design(e, preds), fe(e)], axis=1)
    m = _glm(e.A, X, pd.factorize(e.id)[0])
    for v in preds:
        if v in m.params:
            out.append(
                dict(cohort=c, part="predictor", term=v, est=m.params[v], se=m.bse[v], n=len(e))
            )
    # change tertiles holding level fixed
    e["ct"] = pd.qcut(e.drz.rank(method="first"), 3, labels=["decline", "stable", "improve"])
    X = pd.concat(
        [
            pd.get_dummies(e.ct, prefix="ct", drop_first=True, dtype=float),
            design(
                e,
                [
                    "lvl",
                    "age",
                    "age2",
                    "female",
                    "edu",
                    "wealth_q",
                    "srh_poor",
                    "ncond",
                    "iadl_any",
                    "dep",
                ],
            ),
            fe(e),
        ],
        axis=1,
    )
    m = _glm(e.A, X, pd.factorize(e.id)[0])
    for t in ["stable", "improve"]:
        out.append(
            dict(cohort=c, part="change", term=t, est=m.params[f"ct_{t}"], se=m.bse[f"ct_{t}"])
        )
    return out


# ---------------------------------------------------------------- (c) Gelbach decomposition
def gelbach(d, cols_by_group):
    allc = [v for g in cols_by_group.values() for v in g]
    X = pd.concat([design(d, allc), fe(d)], axis=1)
    y = d.rz.values
    A = d.A.values.astype(float)
    base = np.column_stack([np.ones(len(d)), A, fe(d).values])
    b_base = np.linalg.lstsq(base, y, rcond=None)[0][1]
    full = np.column_stack([np.ones(len(d)), A, X.values])
    bf = np.linalg.lstsq(full, y, rcond=None)[0]
    beta = pd.Series(bf[2:], index=X.columns)
    b_full = bf[1]
    res = {"total": b_base, "unexplained": b_full}
    Z = np.column_stack([np.ones(len(d)), A, fe(d).values])
    for g, cols in cols_by_group.items():
        gc = [k for k in X.columns if k in cols or k[:-2] in cols]
        contrib = 0.0
        for k in gc:
            gam = np.linalg.lstsq(Z, X[k].values, rcond=None)[0][1]
            contrib += gam * beta[k]
        res[g] = contrib
    return res


def decomposition(c, B=100):
    d = prep(c)
    d = d[d.rz.notna()].reset_index(drop=True)
    groups = dict(GROUPS)
    if d.rz_lag.notna().mean() < 0.5:
        d = d.copy()
    pt = gelbach(d, groups)
    ids = d.id.values
    u, inv = np.unique(ids, return_inverse=True)
    bs = []
    for _ in range(B):
        w = rng.poisson(1, len(u))[inv]
        dd = d.loc[np.repeat(np.arange(len(d)), w)].reset_index(drop=True)
        bs.append(gelbach(dd, groups))
    bs = pd.DataFrame(bs)
    return [
        dict(cohort=c, part="gelbach", term=k, est=v, se=bs[k].std(), n=len(d))
        for k, v in pt.items()
    ]


if __name__ == "__main__":
    import os

    SKIPDEC = os.environ.get("SKIPDEC") == "1"
    if SKIPDEC:
        OLD = pd.read_csv("results_anatomy.csv")
        OLD = OLD[OLD.part == "gelbach"]
    rows = [] if not SKIPDEC else OLD.to_dict("records")
    for c in COH:
        rows += adoption_models(c)
        rows += decomposition(c) if SKIPDEC is False else []
        print("done", c, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv("results_anatomy.csv", index=False)
    P = []
    for (part, term), g in R[R.part.isin(["quintile", "change", "predictor", "gelbach"])].groupby(
        ["part", "term"]
    ):
        g = g[g.se.notna() & (g.se > 0)]
        if len(g) < 2:
            continue
        r = hksj(g.est, g.se)
        P.append(
            dict(
                part=part,
                term=term,
                k=r["k"],
                est=r["est"],
                lo=r["hk_lo"],
                hi=r["hk_hi"],
                I2=r["I2"],
            )
        )
    P = pd.DataFrame(P)
    P.to_csv("results_anatomy_pooled.csv", index=False)
    pd.set_option("display.width", 200)
    x = P.copy()
    m = x.part.isin(["quintile", "change", "predictor"])
    for k in ["est", "lo", "hi"]:
        x.loc[m, k] = np.exp(x.loc[m, k])
    print(x.round(3).to_string())
    print(R[R.part == "prob_quintile"].pivot(index="term", columns="cohort", values="est").round(3))
