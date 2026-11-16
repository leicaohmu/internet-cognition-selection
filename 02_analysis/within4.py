"""Change-score (difference-in-differences) early vs late: outcome = recall(t2) - recall(t0), adjusted for t-1 and demographics."""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from within2 import build, PRECOLS
from ladder import aipw_cols
from pool_hk import hksj

rows = []
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    t = build(c)
    el = t[t.grp.isin(["early", "late"])].copy()
    el["A"] = (el.grp == "early").astype(int)
    el["Y_chg"] = el.Y_recall - el.Y_pre
    for name, dd in [
        ("primary", el),
        ("sustained", el[(el.grp == "late") | ((el.internet_2 == 1) & (el.internet_3 == 1))]),
    ]:
        x = dd[dd.recall_lag.notna()]
        if x.Y_chg.notna().sum() < 150 or x.A.sum() < 30:
            continue
        r = aipw_cols(x, "Y_chg", PRECOLS)
        r.update(variant="did_" + name)
        rows.append(r)
        print(
            c,
            name,
            f"{r['est']:+.3f} ({r['lo']:+.3f},{r['hi']:+.3f}) n={r['n']} A1={r['n_A1']}",
            flush=True,
        )
d = pd.DataFrame(rows)
d.to_csv("results_within4.csv", index=False)
for v, g in d.groupby("variant"):
    r = hksj(g.est, g.se)
    print(
        "POOLED",
        v,
        len(g),
        f"{r['est']:+.3f} DL({r['dl_lo']:+.3f},{r['dl_hi']:+.3f}) 90DL({r['est']-1.645*(r['dl_hi']-r['est'])/1.96:+.3f},{r['est']+1.645*(r['dl_hi']-r['est'])/1.96:+.3f}) HK({r['hk_lo']:+.3f},{r['hk_hi']:+.3f}) HK90({r['hk90_lo']:+.3f},{r['hk90_hi']:+.3f})",
    )
