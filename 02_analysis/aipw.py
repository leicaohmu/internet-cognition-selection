"""Cross-fitted AIPW with outcome-missingness (IPCW) and person-clustered influence-function SEs."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import (
    HistGradientBoostingClassifier as HGBC,
    HistGradientBoostingRegressor as HGBR,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from build_trials import COV, LAG, T1, EXTRA
import os

USE_LAG = os.environ.get("USE_LAG", "0") == "1"
USE_T1 = os.environ.get("USE_T1", "0") == "1"
USE_EXTRA = os.environ.get("USE_EXTRA", "0") == "1"

HGB = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, random_state=1)


def design(t):
    X = t[
        COV + (LAG if USE_LAG else []) + (T1 if USE_T1 else []) + (EXTRA if USE_EXTRA else [])
    ].copy()
    if t.cohort.iloc[0] == "SHARE":
        X = X.join(pd.get_dummies(t.country, prefix="c", dtype=float))
    X = X.drop(columns=[c for c in X if X[c].isna().all()])
    return X.astype(float)


def _clf(kind):
    if kind == "hgb":
        return HGBC(**HGB)
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )


def _reg(kind):
    if kind == "hgb":
        return HGBR(**HGB)
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LinearRegression()
    )


def nuisances(X, A, Y, groups, binary, kind="hgb", k=5, seed=1):
    """Return out-of-fold e(X), pi_a(X)=P(R=1|A=a,X), mu_a(X)."""
    n = len(A)
    R = (~np.isnan(Y)).astype(int)
    e, pi1, pi0, m1, m0 = (np.zeros(n) for _ in range(5))
    folds = GroupKFold(n_splits=k).split(X, A, groups)
    for tr, te in folds:
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ok = lambda idx: X.columns[X.iloc[idx].notna().any().values]
        c = ok(tr)
        e[te] = _clf(kind).fit(Xtr[c], A[tr]).predict_proba(Xte[c])[:, 1]
        for a, pi, m in [(1, pi1, m1), (0, pi0, m0)]:
            sa = tr[A[tr] == a]
            if (R[sa] == 0).sum() < 10:
                pi[te] = 1.0
            else:
                c = ok(sa)
                pi[te] = _clf(kind).fit(X.iloc[sa][c], R[sa]).predict_proba(Xte[c])[:, 1]
            so = sa[R[sa] == 1]
            c = ok(so)
            if binary:
                yo = Y[so].astype(int)
                m[te] = (
                    _clf(kind).fit(X.iloc[so][c], yo).predict_proba(Xte[c])[:, 1]
                    if yo.min() != yo.max()
                    else yo.mean()
                )
            else:
                m[te] = _reg(kind).fit(X.iloc[so][c], Y[so]).predict(Xte[c])
    return e, pi1, pi0, m1, m0


def aipw(t, y, kind="hgb", trunc=0.01, binary=None, return_pseudo=False, trim=None):
    t = t.reset_index(drop=True)
    X = design(t)
    A = t.A.values.astype(int)
    Y = t[y].values.astype(float)
    if binary is None:
        binary = set(np.unique(Y[~np.isnan(Y)])) <= {0.0, 1.0}
    e, pi1, pi0, m1, m0 = nuisances(X, A, Y, t.id.values, binary, kind)
    keep = np.ones(len(A), bool) if trim is None else (e >= trim) & (e <= 1 - trim)
    e = np.clip(e, trunc, 1 - trunc)
    pi1, pi0 = np.clip(pi1, 0.05, 1), np.clip(pi0, 0.05, 1)
    A, Y, e, pi1, pi0, m1, m0 = (v[keep] for v in (A, Y, e, pi1, pi0, m1, m0))
    t = t[keep].reset_index(drop=True)
    X = X[keep].reset_index(drop=True)
    R = ~np.isnan(Y)
    Yf = np.nan_to_num(Y)
    psi1 = m1 + A * R * (Yf - m1) / (e * pi1)
    psi0 = m0 + (1 - A) * R * (Yf - m0) / ((1 - e) * pi0)
    res = summarise(psi1, psi0, t.id.values, binary)
    res.update(
        n=len(t), n_obs=int(R.sum()), n_A1=int(A.sum()), y=y, cohort=t.cohort.iloc[0], learner=kind
    )
    if return_pseudo:
        return res, dict(psi1=psi1, psi0=psi0, e=e, X=X, t=t)
    return res


def cluster_var(phi, ids):
    s = pd.Series(phi).groupby(ids).sum()
    n = len(phi)
    return float((s**2).sum() / n**2)


def summarise(psi1, psi0, ids, binary):
    mu1, mu0 = psi1.mean(), psi0.mean()
    rd = mu1 - mu0
    se_rd = np.sqrt(cluster_var(psi1 - psi0 - rd, ids))
    out = dict(mu1=mu1, mu0=mu0, est=rd, se=se_rd, lo=rd - 1.96 * se_rd, hi=rd + 1.96 * se_rd)
    if binary and mu0 > 0 and mu1 > 0:
        phi = (psi1 - mu1) / mu1 - (psi0 - mu0) / mu0
        se_lrr = np.sqrt(cluster_var(phi, ids))
        lrr = np.log(mu1 / mu0)
        out.update(
            rr=np.exp(lrr),
            rr_lo=np.exp(lrr - 1.96 * se_lrr),
            rr_hi=np.exp(lrr + 1.96 * se_lrr),
            se_lrr=se_lrr,
        )
    return out


def balance(t, kind="hgb", trunc=0.01, trim=None):
    """Standardised mean differences before/after ATE IP weighting."""
    X = design(t)
    A = t.A.values.astype(int)
    e = np.zeros(len(A))
    for tr, te in GroupKFold(5).split(X, A, t.id.values):
        e[te] = _clf(kind).fit(X.iloc[tr], A[tr]).predict_proba(X.iloc[te])[:, 1]
    if trim is not None:
        k = (e >= trim) & (e <= 1 - trim)
        X, A, e, t = X[k], A[k], e[k], t[k]
    e = np.clip(e, trunc, 1 - trunc)
    w = np.where(A == 1, 1 / e, 1 / (1 - e))
    rows = []
    for c in COV:
        if c not in X:
            continue
        x = X[c].values
        ok = ~np.isnan(x)

        def smd(wt):
            a1, a0 = ok & (A == 1), ok & (A == 0)
            m1, m0 = np.average(x[a1], weights=wt[a1]), np.average(x[a0], weights=wt[a0])
            v1 = np.average((x[a1] - m1) ** 2, weights=wt[a1])
            v0 = np.average((x[a0] - m0) ** 2, weights=wt[a0])
            return (m1 - m0) / np.sqrt((v1 + v0) / 2) if v1 + v0 > 0 else 0.0

        rows.append(
            dict(cohort=t.cohort.iloc[0], var=c, smd_raw=smd(np.ones_like(w)), smd_w=smd(w))
        )
    return pd.DataFrame(rows), e


def meta_dl(est, se):
    est, se = np.asarray(est), np.asarray(se)
    w = 1 / se**2
    fe = (w * est).sum() / w.sum()
    Q = (w * (est - fe) ** 2).sum()
    k = len(est)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum())) if k > 1 else 0.0
    ws = 1 / (se**2 + tau2)
    re = (ws * est).sum() / ws.sum()
    sre = np.sqrt(1 / ws.sum())
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(est=re, se=sre, lo=re - 1.96 * sre, hi=re + 1.96 * sre, tau2=tau2, I2=I2, Q=Q, k=k)
