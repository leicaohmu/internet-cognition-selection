import pandas as pd
import numpy as np
from pool_hk import hksj

d = pd.read_csv("results_sens_overlap.csv")
rows = []
for (des, y, m), g in d.groupby(["design", "y", "method"]):
    g = g.dropna(subset=["se"])
    if len(g) < 2:
        r = dict(
            est=g.est.iloc[0],
            dl_lo=g.est.iloc[0] - 1.96 * g.se.iloc[0],
            dl_hi=g.est.iloc[0] + 1.96 * g.se.iloc[0],
            k=1,
            hk_lo=np.nan,
            hk_hi=np.nan,
            I2=np.nan,
        )
    else:
        r = hksj(g.est, g.se)
    r.update(design=des, y=y, method=m, cohorts=",".join(g.cohort), n=int(g.n.sum()))
    rows.append(r)
r = pd.DataFrame(rows)[
    ["design", "y", "method", "cohorts", "k", "n", "est", "dl_lo", "dl_hi", "hk_lo", "hk_hi", "I2"]
]
r.to_csv("results_sens_pooled.csv", index=False)
pd.set_option("display.width", 220)
print(r.round(3).to_string(index=False))
