"""Placebo with initiation two waves after the outcome: non-user at t0..t3, B = use at t4, outcomes at t2."""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from build_trials import load, COV, LAG, EXTRA
from aipw import aipw, meta_dl

S5 = {
    "HRS": [(w, w + 1, w + 2, w + 3, w + 4) for w in range(6, 12)],
    "ELSA": [(w, w + 1, w + 2, w + 3, w + 4) for w in range(2, 7)],
}
main = pd.read_csv("trials.csv.gz", low_memory=False)
rows = []
for c, sched in S5.items():
    p = load(c)
    ix = p.set_index(["id", "wave"])
    out = []
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
            (t2, "_2", ["iwstat", "internet", "dep_bin", "recall", "recall_web", "iadl_any"]),
            (t3, "_3", ["iwstat", "internet"]),
            (t4, "_4", ["iwstat", "internet"]),
        ]:
            b = b.join(ix.xs(w, level="wave")[cols].add_suffix(suf), on="id")
        b = b[
            (b.iwstat_1 == 1)
            & (b.internet_1 == 0)
            & (b.iwstat_2 == 1)
            & (b.internet_2 == 0)
            & (b.iwstat_3 == 1)
            & (b.internet_3 == 0)
            & (b.iwstat_4 == 1)
            & b.internet_4.notna()
        ]
        b["A"] = b.internet_4
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
    print(c, "rows", len(t), "future initiators", int(t.A.sum()))
    for y, f in [
        ("Y_recall", lambda d: d),
        ("Y_iadl", lambda d: d[d.iadl_any == 0]),
        ("Y_dep", lambda d: d),
    ]:
        r = aipw(f(t), y, trim=0.025)
        rows.append(r)
        print(
            "  ",
            y,
            f"{r['est']:+.4f} ({r['lo']:+.4f},{r['hi']:+.4f})",
            f"RR={r.get('rr', np.nan):.2f}",
        )
pd.DataFrame(rows).to_csv("results_placebo_lead2.csv", index=False)
