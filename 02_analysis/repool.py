"""Rebuild *_all.csv from per-cohort files and re-pool with HKSJ (as in the original pooled files)."""

import pandas as pd
import numpy as np
import sys
from pool_hk import hksj
from aipw import meta_dl

C = ["HRS", "ELSA", "SHARE", "CHARLS"]


def LO(h):
    return h["hk_lo"] if h["k"] >= 3 else h["dl_lo"]


def HI(h):
    return h["hk_hi"] if h["k"] >= 3 else h["dl_hi"]


def cat(pats):
    fs = []
    for p in pats:
        for c in C:
            try:
                fs.append(pd.read_csv(p.format(c)))
            except Exception:
                pass
    return pd.concat(fs, ignore_index=True)


def es(allf, pats, pooled, extra_dl=False, i2=True, dl2=True):
    a = cat(pats)
    a.to_csv(allf, index=False)
    P = []
    for (y, e), g in a.groupby(["y", "e"], sort=False):
        g = g[g.se > 0]
        if len(g) < 2:
            continue
        h = hksj(g.est, g.se)
        r = dict(
            y=y,
            e=e,
            k=h["k"],
            est=h["est"],
            lo=LO(h) if dl2 else h["hk_lo"],
            hi=HI(h) if dl2 else h["hk_hi"],
        )
        if i2:
            r["I2"] = h["I2"]
        if extra_dl:
            r.update(dl_lo=h["dl_lo"], dl_hi=h["dl_hi"])
        P.append(r)
    pd.DataFrame(P).to_csv(pooled, index=False)


def negctrl():
    a = cat(["results_negctrl_{}.csv", "results_negctrl2_{}.csv"])
    a.to_csv("results_negctrl_all.csv", index=False)
    P = []
    for (o, d), g in a.groupby(["outcome", "design"]):
        g = g[g.se > 0]
        if len(g) < 2:
            continue
        h = hksj(g.est, g.se)
        P.append(
            dict(
                outcome=o,
                design=d,
                k=h["k"],
                est=h["est"],
                lo=LO(h),
                hi=HI(h),
                cohorts=",".join(g.cohort),
            )
        )
    pd.DataFrame(P).to_csv("results_negctrl_pooled.csv", index=False)


def behav():
    a = cat(["results_behav_vig_weekly_{}.csv"])
    a.to_csv("results_behav_vig_all.csv", index=False)
    P = []
    for (o, d), g in a.groupby(["outcome", "design"]):
        g = g[g.se > 0]
        if len(g) < 2:
            continue
        h = hksj(g.est, g.se)
        P.append(
            dict(
                outcome=o,
                design=d,
                k=h["k"],
                est=h["est"],
                lo=LO(h),
                hi=HI(h),
                lo90=h["hk90_lo"],
                hi90=h["hk90_hi"],
                dl_lo=h["dl_lo"],
                dl_hi=h["dl_hi"],
                I2=h["I2"],
            )
        )
    pd.DataFrame(P).to_csv("results_behav_vig_pooled.csv", index=False)


def newout():
    a = cat(["results_newout_{}.csv"])
    a.to_csv("results_newout_all.csv", index=False)
    P = []
    for (d, o), g in a.groupby(["design", "outcome"]):
        g = g[g.se > 0]
        coh = "/".join(g.cohort)
        isrr = "rr" in g and g.rr.notna().all() and len(g) and d != "pretrend"
        if isrr:
            if len(g) == 1:
                P.append(
                    dict(
                        design=d,
                        outcome=o,
                        scale="RR",
                        cohorts=coh,
                        k=1,
                        est=g.rr.iloc[0],
                        hk_lo=np.nan,
                        hk_hi=np.nan,
                        I2=0.0,
                    )
                )
            else:
                h = hksj(np.log(g.rr), g.se_lrr)
                P.append(
                    dict(
                        design=d,
                        outcome=o,
                        scale="RR",
                        cohorts=coh,
                        k=h["k"],
                        est=np.exp(h["est"]),
                        hk_lo=np.exp(LO(h)),
                        hk_hi=np.exp(HI(h)),
                        I2=h["I2"],
                    )
                )
        if isrr:
            continue
        if len(g) >= 2:
            h = hksj(g.est, g.se)
            P.append(
                dict(
                    design=d,
                    outcome=o,
                    scale="RD/SD",
                    cohorts=coh,
                    k=h["k"],
                    est=h["est"],
                    hk_lo=LO(h),
                    hk_hi=HI(h),
                    I2=h["I2"],
                )
            )
        elif len(g) == 1:
            P.append(
                dict(
                    design=d,
                    outcome=o,
                    scale="RD/SD",
                    cohorts=coh,
                    k=1,
                    est=g.est.iloc[0],
                    hk_lo=g.lo.iloc[0],
                    hk_hi=g.hi.iloc[0],
                    I2=0.0,
                )
            )
    pd.DataFrame(P).to_csv("results_newout_pooled.csv", index=False)


def subgrp():
    a = cat(["results_subgrp_{}.csv"])
    a.to_csv("results_subgrp_all.csv", index=False)
    P = []
    for s, g in a.groupby("subgroup", sort=False):
        w = g.pivot_table(index="cohort", columns="design", values=["est", "se"])
        if "tte" not in w["est"] or "placebo" not in w["est"]:
            continue
        w = w.dropna()
        if len(w) < 2:
            continue
        t = hksj(w["est"]["tte"], w["se"]["tte"])
        p = hksj(w["est"]["placebo"], w["se"]["placebo"])
        d = hksj(
            w["est"]["tte"] - w["est"]["placebo"],
            np.sqrt(w["se"]["tte"] ** 2 + w["se"]["placebo"] ** 2),
        )
        P.append(
            dict(
                subgroup=s,
                k=len(w),
                tte=t["est"],
                tte_lo=LO(t),
                tte_hi=HI(t),
                pla=p["est"],
                pla_lo=LO(p),
                pla_hi=HI(p),
                diff=d["est"],
                diff_lo=LO(d),
                diff_hi=HI(d),
            )
        )
    pd.DataFrame(P).to_csv("results_subgrp_pooled.csv", index=False)


def corr():
    a = cat(["results_corr_{}.csv"])
    a.to_csv("results_corr_all.csv", index=False)
    P = []
    for (y, x, z), q in a.groupby(["y", "a", "b"]):
        if x >= z:
            continue
        wa, wb = 1 / q.var_a, 1 / q.var_b
        cov = (wa * wb * q["cov"]).sum() / (wa.sum() * wb.sum())
        P.append(dict(y=y, a=x, b=z, rho=cov / np.sqrt((1 / wa.sum()) * (1 / wb.sum()))))
    pd.DataFrame(P).to_csv("results_corr_pooled.csv", index=False)


def meta(tag):
    res = pd.read_csv(f"results_main_{tag}.csv")
    M = []
    for y, g in res.groupby("y"):
        m = meta_dl(g.est, g.se)
        m.update(y=y, scale="RD/MD")
        M.append(m)
        if "se_lrr" in g and g.se_lrr.notna().all():
            ml = meta_dl(np.log(g.rr), g.se_lrr)
            M.append(
                dict(
                    y=y,
                    scale="RR",
                    est=np.exp(ml["est"]),
                    lo=np.exp(ml["lo"]),
                    hi=np.exp(ml["hi"]),
                    I2=ml["I2"],
                    tau2=ml["tau2"],
                    k=ml["k"],
                )
            )
    pd.DataFrame(M).to_csv(f"results_meta_{tag}.csv", index=False)


def hte():
    sub = pd.read_csv("hte_subgroups_by_cohort.csv")
    P = []
    for (y, s), g in sub.groupby(["y", "subgroup"]):
        m = meta_dl(g.est, g.se)
        P.append(
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
    pd.DataFrame(P).to_csv("hte_subgroups_pooled.csv", index=False)


JOBS = dict(
    es=lambda: es("results_es_all.csv", ["results_es_{}.csv"], "results_es_pooled.csv", dl2=False),
    es1=lambda: es(
        "results_es1_all.csv", ["results_es1_{}.csv"], "results_es1_pooled.csv", dl2=False
    ),
    es_vig=lambda: es(
        "results_es_vig_all.csv",
        ["results_es_vig_weekly_{}.csv"],
        "results_es_vig_pooled.csv",
        i2=False,
    ),
    es_stop=lambda: es(
        "results_es_stop_all.csv", ["results_es_stop_{}.csv"], "results_es_stop_pooled.csv"
    ),
    es_stopvig=lambda: es(
        "results_es_stopvig_all.csv",
        ["results_es_stopvig_{}.csv"],
        "results_es_stopvig_pooled.csv",
        extra_dl=True,
        i2=False,
    ),
    negctrl=negctrl,
    behav=behav,
    newout=newout,
    subgrp=subgrp,
    corr=corr,
    hte=hte,
)
if __name__ == "__main__":
    for j in sys.argv[1:]:
        if j.startswith("meta:"):
            meta(j[5:])
        else:
            JOBS[j]()
        print("done", j)
