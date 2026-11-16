"""Heterogeneous treatment effects via DR-learner on cross-fitted AIPW pseudo-outcomes.
(a) Prespecified subgroup effects per cohort + random-effects pooling + interaction tests
(b) GBM DR-learner CATE, GATES by predicted-benefit quintile (within-cohort cross-fitting)
(c) Transportability: CATE model trained on 3 cohorts, GATES evaluated in held-out 4th cohort
"""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings
import sys

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.model_selection import GroupKFold
from aipw import aipw, cluster_var, meta_dl

COH = ["HRS", "ELSA", "SHARE", "CHARLS"]
OUT = {"Y_recall": lambda d: d, "Y_iadl": lambda d: d[d.iadl_any == 0], "Y_dep": lambda d: d}
MODS = [
    "age",
    "female",
    "edu",
    "wealth_q",
    "live_alone",
    "partnered",
    "dep",
    "iadl_any",
    "srh_poor",
    "ncond",
    "recall",
    "sp_internet",
    "ever_before",
    "work",
    "vig_weekly",
]

t = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
t["id"] = t.cohort + "_" + t.id


def subgroups(d):
    g = {
        "Age 50-64": d.age < 65,
        "Age 65-74": (d.age >= 65) & (d.age < 75),
        "Age 75+": d.age >= 75,
        "Men": d.female == 0,
        "Women": d.female == 1,
        "Education low": d.edu == 0,
        "Education mid": d.edu == 1,
        "Education high": d.edu == 2,
        "Wealth Q1-2": d.wealth_q <= 1,
        "Wealth Q3-5": d.wealth_q >= 2,
        "Lives alone": d.live_alone == 1,
        "Lives with others": d.live_alone == 0,
        "Elevated depressive symptoms": d.dep_bin == 1,
        "No elevated symptoms": d.dep_bin == 0,
        "Fair/poor health": d.srh_poor == 1,
        "Good+ health": d.srh_poor == 0,
        "Spouse online": d.sp_internet == 1,
        "Spouse offline/none": d.sp_internet != 1,
        "Never used before": d.ever_before == 0,
        "Used before (returner)": d.ever_before == 1,
    }
    if d.rural.notna().any():
        g["Rural"] = d.rural == 1
        g["Urban"] = d.rural == 0
    return g


def run():
    pseudo = {}
    sub_rows = []
    for y, f in OUT.items():
        for c in COH:
            d = f(t[t.cohort == c])
            res, P = aipw(d, y, trim=0.025, return_pseudo=True)
            dd = P["t"].copy()
            phi = P["psi1"] - P["psi0"]
            dd["phi"] = phi
            pseudo[(y, c)] = dd
            for name, mask in subgroups(dd).items():
                m = mask.fillna(False).values
                if m.sum() < 200 or dd.A.values[m].sum() < 30:
                    continue
                est = phi[m].mean()
                se = np.sqrt(cluster_var(phi[m] - est, dd.id.values[m]))
                sub_rows.append(
                    dict(
                        y=y,
                        cohort=c,
                        subgroup=name,
                        est=est,
                        se=se,
                        n=int(m.sum()),
                        n_A1=int(dd.A.values[m].sum()),
                    )
                )
            print(y, c, "done", flush=True)
    sub = pd.DataFrame(sub_rows)
    pooled = []
    for (y, s), g in sub.groupby(["y", "subgroup"]):
        m = meta_dl(g.est, g.se)
        pooled.append(
            dict(
                y=y,
                subgroup=s,
                est=m["est"],
                lo=m["lo"],
                hi=m["hi"],
                se=m["se"],
                I2=m["I2"],
                k=m["k"],
            )
        )
    pooled = pd.DataFrame(pooled)
    sub.to_csv("hte_subgroups_by_cohort.csv", index=False)
    pooled.to_csv("hte_subgroups_pooled.csv", index=False)
    pd.to_pickle(pseudo, "pseudo.pkl")
    return pseudo, pooled


def gates(phi, score, ids, q=5):
    bins = pd.qcut(pd.Series(score).rank(method="first"), q, labels=False).values
    rows = []
    for b in range(q):
        m = bins == b
        e = phi[m].mean()
        se = np.sqrt(cluster_var(phi[m] - e, ids[m]))
        rows.append(dict(q=b + 1, est=e, lo=e - 1.96 * se, hi=e + 1.96 * se))
    top, bot = bins == q - 1, bins == 0
    diff = phi[top].mean() - phi[bot].mean()
    # SE of difference (independent clusters approx)
    se_d = np.sqrt(
        cluster_var(np.where(top, phi - phi[top].mean(), 0), ids) * (len(phi) / top.sum()) ** 2
        + cluster_var(np.where(bot, phi - phi[bot].mean(), 0), ids) * (len(phi) / bot.sum()) ** 2
    )
    return pd.DataFrame(rows), diff, se_d


def cate_models(pseudo):
    HP = dict(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=300, random_state=2
    )
    g_rows, tr_rows = [], []
    for y in OUT:
        # (b) within-cohort cross-fitted CATE
        for c in COH:
            d = pseudo[(y, c)]
            X = d[MODS].astype(float)
            phi = d.phi.values
            ids = d.id.values
            s = np.zeros(len(d))
            for tr, te in GroupKFold(5).split(X, phi, ids):
                s[te] = HGBR(**HP).fit(X.iloc[tr], phi[tr]).predict(X.iloc[te])
            gt, diff, se = gates(phi, s, ids)
            gt["y"], gt["cohort"], gt["type"] = y, c, "within"
            g_rows.append(gt)
            tr_rows.append(
                dict(
                    y=y,
                    cohort=c,
                    type="within",
                    diff=diff,
                    lo=diff - 1.96 * se,
                    hi=diff + 1.96 * se,
                )
            )
        # (c) transport: train on other three cohorts (outcome standardised effect scale per cohort)
        for c in COH:
            trn = pd.concat([pseudo[(y, k)] for k in COH if k != c])
            tst = pseudo[(y, c)]
            m = HGBR(**HP).fit(trn[MODS].astype(float), trn.phi.values)
            s = m.predict(tst[MODS].astype(float))
            gt, diff, se = gates(tst.phi.values, s, tst.id.values)
            gt["y"], gt["cohort"], gt["type"] = y, c, "transport"
            g_rows.append(gt)
            tr_rows.append(
                dict(
                    y=y,
                    cohort=c,
                    type="transport",
                    diff=diff,
                    lo=diff - 1.96 * se,
                    hi=diff + 1.96 * se,
                )
            )
        print(y, "CATE done", flush=True)
    pd.concat(g_rows).to_csv("hte_gates.csv", index=False)
    tr = pd.DataFrame(tr_rows)
    tr.to_csv("hte_top_vs_bottom.csv", index=False)
    return tr


if __name__ == "__main__":
    pseudo, pooled = run()
    print(pooled[pooled.y != "Y_dep"].round(3).to_string())
    tr = cate_models(pseudo)
    print(tr.round(3).to_string())
