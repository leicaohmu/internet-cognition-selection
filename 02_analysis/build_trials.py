"""Build sequential nested target trials (t0 -> t1 exposure -> t2 outcome) for each cohort."""

import pandas as pd
import numpy as np

SCHED = {
    "HRS": [(w, w + 1, w + 2) for w in range(6, 14)],
    "ELSA": [(w, w + 1, w + 2) for w in range(2, 8)],
    "SHARE": [(5, 6, 8), (6, 8, 9)],
    "CHARLS": [(1, 2, 3), (2, 3, 4), (3, 4, 5)],
}
DEPCUT = {"HRS": 3, "ELSA": 3, "SHARE": 4, "CHARLS": 10, "LASI": 10}
DEPCUT_ALT = {"HRS": 4, "ELSA": 4, "SHARE": 3, "CHARLS": 12, "LASI": 12}


def load(c):
    p = pd.read_csv(f"panel_{c}.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
    p = p.drop_duplicates(["id", "wave"])
    if c == "SHARE":
        p["partnered"] = p.mstat.isin([1, 2]).astype(float).where(p.mstat.notna())
    else:
        p["partnered"] = p.mstat.isin([1, 2, 3]).astype(float).where(p.mstat.notna())
    p["srh_poor"] = (p.shlt >= 4).astype(float).where(p.shlt.notna())
    p["adl_any"] = (p.adl > 0).astype(float).where(p.adl.notna())
    p["iadl_any"] = (p.iadl > 0).astype(float).where(p.iadl.notna())
    p["live_alone"] = (p.hhsize == 1).astype(float).where(p.hhsize.notna())
    p["dep_bin"] = (p.dep >= DEPCUT[c]).astype(float).where(p.dep.notna())
    p["dep_bin_alt"] = (p.dep >= DEPCUT_ALT[c]).astype(float).where(p.dep.notna())
    # wealth quintile within wave (SHARE: within country x wave; SHARE value is already an income percentile)
    grp = ["wave", "country"] if c == "SHARE" else ["wave"]
    p["wealth_q"] = p.groupby(grp).wealth.transform(
        lambda s: (
            pd.qcut(s.rank(method="first"), 5, labels=False) if s.notna().sum() > 50 else np.nan
        )
    )
    # ever used internet at an earlier wave
    p = p.sort_values(["id", "wave"])
    p["ever_before"] = p.groupby("id").internet.transform(lambda s: s.shift().fillna(0).cummax())
    # ---- extra confounders + spouse internet use ----
    x = pd.read_csv(
        f"extra_{c}.csv.gz", dtype={"id": str, "sp_id": str, "couple": str}, low_memory=False
    ).drop_duplicates(["id", "wave"])
    p = p.merge(x, on=["id", "wave"], how="left")
    if c == "CHARLS":
        s = p.sp_id.astype(str)
        p["sp_id"] = np.where(
            s.str.len() == 11, s.str[:9] + "0" + s.str[9:], np.where(s.str.len() == 12, s, None)
        )
    if c in ("ELSA", "CHARLS"):
        sp = p[["id", "wave", "internet"]].rename(
            columns={"id": "sp_id", "internet": "sp_internet"}
        )
        p = p.merge(sp, on=["sp_id", "wave"], how="left")
    elif c == "SHARE":
        cp = p[p.couple.notna()][["couple", "wave", "id", "internet"]]
        m = cp.merge(cp, on=["couple", "wave"], suffixes=("", "_sp"))
        m = (
            m[m.id != m.id_sp]
            .drop_duplicates(["id", "wave"])[["id", "wave", "internet_sp"]]
            .rename(columns={"internet_sp": "sp_internet"})
        )
        p = p.merge(m, on=["id", "wave"], how="left")
    for col in ["hosp", "bmi", "grip", "kcnt", "lonely", "socwk", "help_recv", "sp_internet"]:
        if col not in p:
            p[col] = np.nan
    # previous-wave (t-1) health trajectory
    lagv = ["dep", "srh_poor", "iadl_any", "adl_any", "recall", "ncond", "work", "wealth_q"]
    p = p.sort_values(["id", "wave"])
    resp = p[p.iwstat == 1]
    lag = resp.groupby("id")[lagv + ["wave"]].shift()
    lag.columns = [c + "_lag" for c in lag.columns]
    p = p.join(lag)
    gap = p.wave - p.wave_lag
    for c in lagv:
        p[c + "_lag"] = p[c + "_lag"].where(gap <= 2)
    if "recall_web" not in p:
        p["recall_web"] = 0.0
    for col in ["rural", "proxy", "cancer", "ncond"]:
        if col not in p:
            p[col] = np.nan
    return p


T1 = ["srh_poor_1", "adl_any_1", "iadl_any_1", "dep_1", "ncond_1", "recall_1"]
EXTRA = ["hosp", "bmi", "grip", "kcnt", "lonely", "socwk", "help_recv", "sp_internet"]
LAG = [
    "dep_lag",
    "srh_poor_lag",
    "iadl_any_lag",
    "adl_any_lag",
    "recall_lag",
    "ncond_lag",
    "work_lag",
    "wealth_q_lag",
]
COV = [
    "age",
    "female",
    "edu",
    "partnered",
    "live_alone",
    "wealth_q",
    "work",
    "srh_poor",
    "ncond",
    "adl_any",
    "iadl_any",
    "dep",
    "recall",
    "smoke",
    "drink",
    "vig_weekly",
    "rural",
    "ever_before",
    "trial",
    "year",
]


def build(c):
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
        s1 = ix.xs(t1, level="wave")[
            [
                "iwstat",
                "proxy",
                "internet",
                "cancer",
                "srh_poor",
                "adl_any",
                "iadl_any",
                "dep",
                "ncond",
                "recall",
            ]
        ].add_suffix("_1")
        s2 = ix.xs(t2, level="wave")[
            [
                "iwstat",
                "dep",
                "dep_bin",
                "dep_bin_alt",
                "recall",
                "recall_web",
                "iadl_any",
                "cancer",
            ]
        ].add_suffix("_2")
        b = b.join(s1, on="id").join(s2, on="id")
        b = b[(b.iwstat_1 == 1) & b.internet_1.notna()]
        b["A"] = b.internet_1
        b["trial"] = k
        b["t0"], b["t1"], b["t2"] = t0, t1, t2
        alive2 = b.iwstat_2 == 1
        died2 = b.iwstat_2.isin([5, 6])
        b["Y_dep"] = b.dep_bin_2.where(alive2)
        b["Y_dep_alt"] = b.dep_bin_alt_2.where(alive2)
        b["Y_depscore"] = b.dep_2.where(alive2)
        rec = b.recall_2.where(
            alive2 & (b.recall_web_2.fillna(0) == 0)
        )  # exclude web-mode recall (HRS 2018-20)
        b["Y_recall"] = rec
        b["Y_iadl"] = b.iadl_any_2.where(alive2 & (b.iadl_any == 0))
        b["Y_death"] = np.where(died2, 1.0, np.where(alive2, 0.0, np.nan))
        # composites: death counts as a bad outcome
        b["Y_iadl_death"] = np.where(died2, 1.0, b.iadl_any_2.where(alive2)).astype(float)
        b["Y_iadl_death"] = pd.Series(b["Y_iadl_death"], index=b.index).where(b.iadl_any == 0)
        b["Y_dep_death"] = np.where(died2, 1.0, b.dep_bin_2.where(alive2)).astype(float)
        if (
            c == "ELSA" and t2 > 6
        ):  # ELSA deaths after wave 6 are not recorded in the harmonised file
            for yy in ["Y_death", "Y_iadl_death", "Y_dep_death"]:
                b[yy] = np.nan
        if c == "SHARE":  # easySHARE has no death information
            for yy in ["Y_death", "Y_iadl_death", "Y_dep_death"]:
                b[yy] = np.nan
        if c != "SHARE":
            inc = ((b.cancer_1 == 1) | (b.cancer_2 == 1)).astype(float)
            b["Y_cancer"] = inc.where(alive2 & (b.cancer == 0) & b.cancer_2.notna())
        else:
            b["Y_cancer"] = np.nan
        out.append(b)
    t = pd.concat(out, ignore_index=True)
    # standardise continuous outcomes against the cohort's baseline distribution
    for y, base in [("Y_recall", "recall"), ("Y_depscore", "dep")]:
        mu, sd = t[base].mean(), t[base].std()
        t[y] = (t[y] - mu) / sd
        t[base + "_z"] = (t[base] - mu) / sd
    t["cohort"] = c
    t["cancer0"] = t["cancer"]
    keep = (
        ["cohort", "id", "country", "t0", "t1", "t2", "A", "dep_bin", "cancer0"]
        + COV
        + LAG
        + T1
        + EXTRA
        + ["proxy_1"]
        + [x for x in t.columns if x.startswith("Y_")]
    )
    return t[keep]


if __name__ == "__main__":
    allt = []
    for c in SCHED:
        t = build(c)
        allt.append(t)
        print(
            f"\n== {c}: trials={t.trial.nunique()} rows={len(t)} persons={t.id.nunique()} initiators={int(t.A.sum())} ({100*t.A.mean():.1f}%)"
        )
        print(
            t.groupby("A")[["Y_dep", "Y_recall", "Y_iadl", "Y_death", "Y_cancer"]]
            .agg(lambda s: f"{s.notna().sum()}|{s.mean():.3f}")
            .T.to_string()
        )
    pd.concat(allt).to_csv("trials.csv.gz", index=False)
