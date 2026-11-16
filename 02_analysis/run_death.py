import os
import sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from aipw import aipw, meta_dl

t = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
t["id"] = t.cohort + "_" + t.id
tag = sys.argv[1]
rows = []
for y, sub in [
    ("Y_death", lambda d: d),
    ("Y_iadl_death", lambda d: d[d.iadl_any == 0]),
    ("Y_dep_death", lambda d: d),
]:
    for c in ["HRS", "ELSA", "CHARLS"]:
        d = sub(t[t.cohort == c])
        if d[y].notna().sum() < 100:
            continue
        r = aipw(d, y, trim=0.025)
        rows.append(r)
        print(
            tag,
            c,
            y,
            "n_obs",
            r["n_obs"],
            "RR %.2f (%.2f-%.2f)" % (r["rr"], r["rr_lo"], r["rr_hi"]),
            flush=True,
        )
res = pd.DataFrame(rows)
res.to_csv(f"results_death_{tag}.csv", index=False)
for y, g in res.groupby("y"):
    m = meta_dl(np.log(g.rr), g.se_lrr)
    print(
        tag,
        y,
        "pooled RR %.3f (%.3f-%.3f) I2=%.2f"
        % (np.exp(m["est"]), np.exp(m["lo"]), np.exp(m["hi"]), m["I2"]),
    )
