"""Adoption and persistence flows among people offline at t0 (Fig. 4c)."""

import os

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from build_trials import load
from within import SCHED4

rows = []
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    p = load(c)
    ix = p.set_index(["id", "wave"])
    for t0, t1, t2, t3 in SCHED4[c]:
        b = p[
            (p.wave == t0)
            & (p.iwstat == 1)
            & (p.internet == 0)
            & (p.age >= 50)
            & (p.proxy.fillna(0) != 1)
        ].copy()
        for w, s in [(t1, "_1"), (t2, "_2"), (t3, "_3")]:
            b = b.join(ix.xs(w, level="wave")[["iwstat", "internet"]].add_suffix(s), on="id")
        b = b[(b.iwstat_1 == 1) & (b.iwstat_2 == 1) & b.internet_1.notna() & b.internet_2.notna()]
        k3 = (b.iwstat_3 == 1) & b.internet_3.notna()
        s1 = np.where(b.internet_1 == 1, "online", "offline")
        s2 = np.where(b.internet_2 == 1, "online", "offline")
        s3 = np.where(~k3, "unknown", np.where(b.internet_3 == 1, "online", "offline"))
        rows.append(pd.DataFrame({"cohort": c, "s1": s1, "s2": s2, "s3": s3}))
d = pd.concat(rows)
f = d.groupby(["s1", "s2", "s3"]).size().rename("n").reset_index()
f.to_csv("flows_t0.csv", index=False)
print(len(d))
print(f)
print(d.groupby("cohort").size())
e = d[d.s1 == "online"]
print(
    "early kept at t2 by cohort",
    e.groupby("cohort").apply(lambda z: (z.s2 == "online").mean()).round(2).to_dict(),
)
