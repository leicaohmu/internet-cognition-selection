"""Negative-control (placebo) exposure: initiation one wave AFTER the outcome.
Eligible at t0: non-user, age>=50, non-proxy; still non-user and alive at t1 and t2.
Placebo exposure B = internet use at t3. Outcomes at t2 (before B can act). Same covariates as primary model.
"""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from build_trials import load, COV, LAG, EXTRA, T1
from aipw import aipw, meta_dl

SCHED4 = {
    "HRS": [(w, w + 1, w + 2, w + 3) for w in range(6, 13)],
    "ELSA": [(w, w + 1, w + 2, w + 3) for w in range(2, 8)],
    "SHARE": [(5, 6, 8, 9)],
    "CHARLS": [(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)],
}

main = pd.read_csv("trials.csv.gz", low_memory=False)


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
        s1 = ix.xs(t1, level="wave")[
            ["iwstat", "internet", "srh_poor", "adl_any", "iadl_any", "dep", "ncond", "recall"]
        ].add_suffix("_1")
        s2 = ix.xs(t2, level="wave")[
            ["iwstat", "internet", "dep_bin", "recall", "recall_web", "iadl_any"]
        ].add_suffix("_2")
        s3 = ix.xs(t3, level="wave")[["iwstat", "internet"]].add_suffix("_3")
        b = b.join(s1, on="id").join(s2, on="id").join(s3, on="id")
        b = b[
            (b.iwstat_1 == 1)
            & (b.internet_1 == 0)
            & (b.iwstat_2 == 1)
            & (b.internet_2 == 0)
            & (b.iwstat_3 == 1)
            & b.internet_3.notna()
        ]
        b["A"] = b.internet_3
        b["trial"] = k
        b["Y_dep"] = b.dep_bin_2
        b["Y_recall"] = b.recall_2.where(b.recall_web_2.fillna(0) == 0)
        b["Y_iadl"] = b.iadl_any_2.where(b.iadl_any == 0)
        out.append(b)
    t = pd.concat(out, ignore_index=True)
    m = main[main.cohort == c]
    mu, sd = m.recall.mean(), m.recall.std()  # same scale as the main analysis
    t["Y_recall"] = (t.Y_recall - mu) / sd
    t["cohort"] = c
    t["id"] = c + "_" + t.id.astype(str)
    return t[
        ["cohort", "id", "country", "A", "dep_bin"]
        + COV
        + LAG
        + EXTRA
        + T1
        + ["Y_dep", "Y_recall", "Y_iadl"]
    ]


if __name__ == "__main__":
    rows = []
    for c in SCHED4:
        t = build(c)
        print(
            f"== {c}: rows={len(t)} future initiators={int(t.A.sum())} ({100*t.A.mean():.1f}%)",
            flush=True,
        )
        for y, f in [
            ("Y_recall", lambda d: d),
            ("Y_iadl", lambda d: d[d.iadl_any == 0]),
            ("Y_dep", lambda d: d),
        ]:
            d = f(t)
            if d[y].notna().sum() < 200 or d.A.sum() < 50:
                print("   skip", y)
                continue
            r = aipw(d, y, trim=0.025)
            rows.append(r)
            print(
                f"   {y:9s} n={r['n']} A1={r['n_A1']} est={r['est']:+.4f} ({r['lo']:+.4f},{r['hi']:+.4f})"
                + (f" RR={r['rr']:.2f} ({r['rr_lo']:.2f}-{r['rr_hi']:.2f})" if "rr" in r else ""),
                flush=True,
            )
    res = pd.DataFrame(rows)
    res.to_csv(
        "results_placebo%s.csv" % ("_t1" if os.environ.get("USE_T1") == "1" else ""), index=False
    )
    prim = pd.read_csv(
        "results_meta_hgb_trim0.025_lag%s_extra.csv"
        % ("_t1" if os.environ.get("USE_T1") == "1" else "")
    )
    print("\nPooled placebo vs main (primary):")
    for y, g in res.groupby("y"):
        m = meta_dl(g.est, g.se)
        pm = prim[(prim.y == y) & (prim.scale == "RD/MD")].iloc[0]
        line = f"{y}: placebo RD/MD {m['est']:+.4f} ({m['lo']:+.4f},{m['hi']:+.4f}) I2={m['I2']:.2f} k={m['k']}  | main {pm.est:+.4f} ({pm.lo:+.4f},{pm.hi:+.4f})"
        if "se_lrr" in g and g.se_lrr.notna().all():
            ml = meta_dl(np.log(g.rr), g.se_lrr)
            pr = prim[(prim.y == y) & (prim.scale == "RR")].iloc[0]
            line += f"\n      placebo RR {np.exp(ml['est']):.3f} ({np.exp(ml['lo']):.3f}-{np.exp(ml['hi']):.3f})  | main RR {pr.est:.3f} ({pr.lo:.3f}-{pr.hi:.3f})"
        print(line)
