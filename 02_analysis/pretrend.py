"""Pre-trend: before anyone in the trial has started, did future initiators already decline less between t-1 and t0?
Outcome = recall at t0 (z) adjusting recall at t-1 and other t-1 / demographic covariates; A = initiation by t1.
"""

import os

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from ladder import aipw_cols
from aipw import meta_dl

t = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
t["id"] = t.cohort + "_" + t.id
cols = [
    "age",
    "female",
    "edu",
    "partnered",
    "live_alone",
    "rural",
    "year",
    "dep_lag",
    "srh_poor_lag",
    "iadl_any_lag",
    "adl_any_lag",
    "recall_lag",
    "ncond_lag",
    "work_lag",
    "wealth_q_lag",
]
rows = []
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    d = t[(t.cohort == c) & t.recall_lag.notna()].copy()
    m = t[t.cohort == c]
    d["Y_pre"] = (d.recall - m.recall.mean()) / m.recall.std()
    d["Y_iadl_pre"] = d.iadl_any.where(d.iadl_any_lag == 0)
    for y in ["Y_pre", "Y_iadl_pre"]:
        dd = d if y == "Y_pre" else d[d.iadl_any_lag == 0]
        if dd[y].notna().sum() < 300:
            continue
        r = aipw_cols(dd, y, cols)
        rows.append(r)
        print(c, y, f"{r['est']:+.4f} ({r['lo']:+.4f},{r['hi']:+.4f}) n={r['n']}", flush=True)
res = pd.DataFrame(rows)
res.to_csv("results_pretrend.csv", index=False)
for y, g in res.groupby("y"):
    m = meta_dl(g.est, g.se)
    print(
        "POOLED", y, f"{m['est']:+.4f} ({m['lo']:+.4f},{m['hi']:+.4f}) k={m['k']} I2={m['I2']:.2f}"
    )
