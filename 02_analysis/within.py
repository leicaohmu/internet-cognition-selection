"""Staggered-initiation (within-adopter) contrast.
Eligible at t0: non-user, age>=50, non-proxy; interviewed with internet status known at t1, t2, t3.
Groups by timing of first use: early (online at t1), middle (offline t1, online t2), late (offline t1,t2; online t3),
never (offline t1..t3). Outcomes at t2. Main contrast: early vs late (both eventually adopt).
"""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from build_trials import load, COV, LAG, EXTRA
from aipw import aipw, meta_dl

SCHED4 = {
    "HRS": [(w, w + 1, w + 2, w + 3) for w in range(6, 13)],
    "ELSA": [(w, w + 1, w + 2, w + 3) for w in range(2, 8)],
    "SHARE": [(5, 6, 8, 9)],
    "CHARLS": [(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)],
}
main = pd.read_csv("trials.csv.gz", low_memory=False)
DELTA = {
    "Y_recall": 0.10,
    "Y_iadl": 0.02,
    "Y_dep": 0.02,
}  # prespecified equivalence margins (SD; risk difference)


def build(c):
    p = load(c)
    ix = p.set_index(["id", "wave"])
    out = []
    for k, (t0, t1, t2, t3) in enumerate(SCHED4[c]):
        b = p[
            (p.wave == t0)
            & (p.iwstat == 1)
            & (p.internet == 0)
            & (p.age >= 50)
            & (p.proxy.fillna(0) != 1)
        ].copy()
        for w, suf, cols in [
            (t1, "_1", ["iwstat", "internet"]),
            (t2, "_2", ["iwstat", "internet", "dep_bin", "recall", "recall_web", "iadl_any"]),
            (t3, "_3", ["iwstat", "internet"]),
        ]:
            b = b.join(ix.xs(w, level="wave")[cols].add_suffix(suf), on="id")
        b = b[
            (b.iwstat_1 == 1)
            & (b.iwstat_2 == 1)
            & (b.iwstat_3 == 1)
            & b.internet_1.notna()
            & b.internet_2.notna()
            & b.internet_3.notna()
        ]
        b["grp"] = np.select(
            [
                b.internet_1 == 1,
                (b.internet_1 == 0) & (b.internet_2 == 1),
                (b.internet_1 == 0) & (b.internet_2 == 0) & (b.internet_3 == 1),
            ],
            ["early", "middle", "late"],
            "never",
        )
        b["trial"] = k
        b["Y_dep"] = b.dep_bin_2
        b["Y_recall"] = b.recall_2.where(b.recall_web_2.fillna(0) == 0)
        b["Y_iadl"] = b.iadl_any_2.where(b.iadl_any == 0)
        out.append(b)
    t = pd.concat(out, ignore_index=True)
    m = main[main.cohort == c]
    t["Y_recall"] = (t.Y_recall - m.recall.mean()) / m.recall.std()
    t["cohort"] = c
    t["id"] = c + "_" + t.id.astype(str)
    return t


if __name__ == "__main__":
    rows = []
    for c in SCHED4:
        t = build(c)
        print(c, t.grp.value_counts().to_dict(), flush=True)
        for a1, a0 in [
            ("early", "late"),
            ("early", "never"),
            ("late", "never"),
            ("middle", "never"),
        ]:
            d = t[t.grp.isin([a1, a0])].copy()
            d["A"] = (d.grp == a1).astype(int)
            for y, f in [
                ("Y_recall", lambda d: d),
                ("Y_iadl", lambda d: d[d.iadl_any == 0]),
                ("Y_dep", lambda d: d),
            ]:
                dd = f(d)
                if dd.A.sum() < 40 or (1 - dd.A).sum() < 40 or dd[y].notna().sum() < 200:
                    continue
                r = aipw(dd, y, trim=0.025)
                r["contrast"] = f"{a1} vs {a0}"
                rows.append(r)
                if a0 == "late":
                    print(
                        f"  {a1} vs {a0} {y}: {r['est']:+.4f} ({r['lo']:+.4f},{r['hi']:+.4f}) n={r['n']} A1={r['n_A1']}",
                        flush=True,
                    )
    res = pd.DataFrame(rows)
    res.to_csv("results_within.csv", index=False)
    print("\nPooled (DerSimonian-Laird; 90% CI for TOST):")
    for (ct, y), g in res.groupby(["contrast", "y"]):
        m = meta_dl(g.est, g.se)
        lo90, hi90 = m["est"] - 1.645 * m["se"], m["est"] + 1.645 * m["se"]
        eq = (lo90 > -DELTA[y]) and (hi90 < DELTA[y])
        print(
            f"{ct:16s} {y:9s} {m['est']:+.4f} 95%({m['lo']:+.4f},{m['hi']:+.4f}) 90%({lo90:+.4f},{hi90:+.4f}) k={m['k']} I2={m['I2']:.2f} equivalent(+-{DELTA[y]})={eq}"
        )
