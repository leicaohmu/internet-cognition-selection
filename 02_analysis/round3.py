"""Round-3 review analyses.
1. Early (start t1) vs very late (offline t1-t3, start t4) at t2 - late group's pre-adoption wave is t3, not the outcome wave (HRS, ELSA).
2. Late vs never at t2 adjusted for recall at t1 (late group's pre-adoption advantage at its pre-adoption wave).
3. Target trial restricted to the placebo-eligible population (alive with known internet status at t3).
4. Pre-COVID restriction: no t1..t3 wave from 2020 (main, placebo, early vs late).
"""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from build_trials import load, COV, LAG, EXTRA
from aipw import aipw
from ladder import aipw_cols
import within
import placebo
from within2 import PRECOLS

YEAR = {
    "HRS": {w: 1990 + 2 * w for w in range(1, 16)},
    "ELSA": {w: 2000 + 2 * w for w in range(1, 11)},
    "SHARE": {5: 2013, 6: 2015, 8: 2019, 9: 2021},
    "CHARLS": {1: 2011, 2: 2013, 3: 2015, 4: 2018, 5: 2020, 6: 2021},
}
# ELSA wave w fieldwork ~ 2002+2(w-1) -> wave 9 = 2018, wave 10 = 2021 ; HRS wave w = 1992+2(w-1) -> 15 = 2020
YEAR["ELSA"] = {w: 2002 + 2 * (w - 1) for w in range(1, 11)}
YEAR["HRS"] = {w: 1992 + 2 * (w - 1) for w in range(1, 16)}
OUT = [("Y_recall", lambda d: d), ("Y_iadl", lambda d: d[d.iadl_any == 0]), ("Y_dep", lambda d: d)]


def run(d, y, tag, c, rows, cols=None):
    if d[y].notna().sum() < 150 or d.A.sum() < 30 or (1 - d.A).sum() < 30:
        return
    r = aipw_cols(d, y, cols) if cols else aipw(d, y, trim=0.025)
    r.update(analysis=tag, cohort=c)
    rows.append(r)
    print(
        c,
        tag,
        y,
        f"{r['est']:+.4f} ({r['lo']:+.4f},{r['hi']:+.4f}) n={r['n']} A1={r['n_A1']}",
        flush=True,
    )


def very_late(c):
    p = load(c)
    ix = p.set_index(["id", "wave"])
    out = []
    last = max(YEAR[c])
    sched = [(w, w + 1, w + 2, w + 3, w + 4) for w in range(min(YEAR[c]), last - 3)]
    if c == "HRS":
        sched = [s for s in sched if s[0] >= 6]
    if c == "ELSA":
        sched = [s for s in sched if s[0] >= 2]
    m = within.main[within.main.cohort == c]
    for k, (t0, t1, t2, t3, t4) in enumerate(sched):
        b = p[
            (p.wave == t0)
            & (p.iwstat == 1)
            & (p.internet == 0)
            & (p.age >= 50)
            & (p.proxy.fillna(0) != 1)
        ].copy()
        for w, suf, cols in [
            (t1, "_1", ["iwstat", "internet"]),
            (t2, "_2", ["iwstat", "internet", "recall", "recall_web", "iadl_any", "dep_bin"]),
            (t3, "_3", ["iwstat", "internet"]),
            (t4, "_4", ["iwstat", "internet"]),
        ]:
            b = b.join(ix.xs(w, level="wave")[cols].add_suffix(suf), on="id")
        early = (b.iwstat_1 == 1) & (b.internet_1 == 1) & (b.iwstat_2 == 1)
        vlate = (
            (b.iwstat_1 == 1)
            & (b.internet_1 == 0)
            & (b.iwstat_2 == 1)
            & (b.internet_2 == 0)
            & (b.iwstat_3 == 1)
            & (b.internet_3 == 0)
            & (b.iwstat_4 == 1)
            & (b.internet_4 == 1)
        )
        b = b[early | vlate].copy()
        b["A"] = early[early | vlate].astype(int).values
        b["trial"] = k
        b["Y_recall"] = (
            b.recall_2.where(b.recall_web_2.fillna(0) == 0) - m.recall.mean()
        ) / m.recall.std()
        b["Y_iadl"] = b.iadl_any_2.where(b.iadl_any == 0)
        b["Y_dep"] = b.dep_bin_2
        out.append(b)
    t = pd.concat(out, ignore_index=True)
    t["cohort"] = c
    t["id"] = c + "_" + t.id.astype(str)
    return t


if __name__ == "__main__":
    rows = []
    main = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
    raw_main = main.copy()
    main["id"] = main.cohort + "_" + main.id
    for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
        Y = YEAR[c]
        # 1. early vs very late
        if c in ("HRS", "ELSA"):
            vl = very_late(c)
            print(c, "early vs very late groups", vl.A.value_counts().to_dict())
            for y, f in OUT:
                run(f(vl), y, "early_vs_verylate", c, rows)
        # 2. late vs never at t2 adjusted for recall at t1 -> use within build, need recall at t1
        w = within.build(c)
        p = load(c)
        ix = p.set_index(["id", "wave"])
        w["rawid"] = w.id.str.replace(c + "_", "", n=1)
        parts = []
        for k, (t0, t1, t2, t3) in enumerate(within.SCHED4[c]):
            s = w[w.trial == k].join(
                ix.xs(t1, level="wave")[["recall"]].add_suffix("_t1"), on="rawid"
            )
            s["t1y"] = Y[t1]
            s["t3y"] = Y[t3]
            s["t2y"] = Y[t2]
            parts.append(s)
        w = pd.concat(parts, ignore_index=True)
        ln = w[w.grp.isin(["late", "never"])].copy()
        ln["A"] = (ln.grp == "late").astype(int)
        mm = within.main[within.main.cohort == c]
        ln["recall_t1z"] = (ln.recall_t1 - mm.recall.mean()) / mm.recall.std()
        run(
            ln[ln.recall_t1.notna()],
            "Y_recall",
            "late_vs_never_adj_t1",
            c,
            rows,
            cols=PRECOLS + ["recall", "recall_t1z"],
        )
        # 4b. pre-COVID early vs late
        el = w[w.grp.isin(["early", "late"])].copy()
        el["A"] = (el.grp == "early").astype(int)
        for y, f in OUT:
            run(f(el[el.t3y < 2020]), y, "early_vs_late_precovid", c, rows)
        # 3. target trial in placebo-eligible population: known internet status at t3 (t3 = t2+1 wave in SCHED)
        mt = main[main.cohort == c].copy()
        mt["rawid"] = mt.id.str.replace(c + "_", "", n=1)
        waves = sorted(Y)
        nxt = {a: b for a, b in zip(waves[:-1], waves[1:])}
        mt["t3"] = mt.t2.map(nxt)
        st = p[["id", "wave", "iwstat", "internet"]].rename(columns={"id": "rawid", "wave": "t3"})
        mt = mt.merge(st, on=["rawid", "t3"], how="left")
        mt["t1y"] = mt.t1.map(Y)
        mt["t2y"] = mt.t2.map(Y)
        common = mt[(mt.iwstat == 1) & mt.internet.notna()]
        for y, f in OUT:
            run(f(common), y, "tte_known_t3", c, rows)
        # 4a. pre-COVID main
        for y, f in OUT:
            run(f(mt[mt.t2y < 2020]), y, "tte_precovid", c, rows)
        # 4c. pre-COVID placebo (t3 < 2020)
        pl = placebo.build(c)
        pl_t3 = [Y[s[3]] for s in placebo.SCHED4[c]]
        pl["t3y"] = pl.trial.map(dict(enumerate(pl_t3))) if "trial" in pl else np.nan
        if pl.t3y.notna().any():
            for y, f in OUT:
                run(f(pl[pl.t3y < 2020]), y, "placebo_precovid", c, rows)
    pd.DataFrame(rows).to_csv("results_round3.csv", index=False)
    print("done")
