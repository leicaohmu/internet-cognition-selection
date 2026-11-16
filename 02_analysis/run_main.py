import pandas as pd
import numpy as np
import time
import sys
import warnings

warnings.filterwarnings("ignore")
from aipw import aipw, balance, meta_dl

t = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
t["id"] = t.cohort + "_" + t.id
kind = sys.argv[1] if len(sys.argv) > 1 else "hgb"
TRIM = float(sys.argv[2]) if len(sys.argv) > 2 else None
import os

tag = (
    kind
    + ("" if TRIM is None else f"_trim{TRIM}")
    + ("_lag" if os.environ.get("USE_LAG") == "1" else "")
    + ("_t1" if os.environ.get("USE_T1") == "1" else "")
    + ("_extra" if os.environ.get("USE_EXTRA") == "1" else "")
)

OUTCOMES = {
    "Y_dep": lambda d: d,
    "Y_depscore": lambda d: d,
    "Y_recall": lambda d: d,
    "Y_iadl": lambda d: d[d.iadl_any == 0],
    "Y_death": lambda d: d[d.cohort != "SHARE"],
    "Y_cancer": lambda d: d[(d.cancer0 == 0) & (d.cohort != "SHARE")],
    "Y_dep_inc": lambda d: d[d.dep_bin == 0],  # incident analysis (sensitivity)
    "Y_dep_alt": lambda d: d,  # alternative cut-off (sensitivity)
}
t["Y_dep_inc"] = t.Y_dep

rows = []
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    for y, f in OUTCOMES.items():
        d = f(t[t.cohort == c])
        if len(d) == 0 or d[y].notna().sum() < 100:
            continue
        t0 = time.time()
        r = aipw(d, y, kind=kind, trim=TRIM)
        rows.append(r)
        print(
            f"{c:7s} {y:11s} n={r['n']:6d} A1={r['n_A1']:5d} est={r['est']:+.4f} ({r['lo']:+.4f},{r['hi']:+.4f})"
            + (f" RR={r['rr']:.2f} ({r['rr_lo']:.2f}-{r['rr_hi']:.2f})" if "rr" in r else "")
            + f"  [{time.time()-t0:.0f}s]",
            flush=True,
        )
res = pd.DataFrame(rows)
res.to_csv(f"results_main_{tag}.csv", index=False)

meta = []
for y, g in res.groupby("y"):
    m = meta_dl(g.est, g.se)
    m.update(y=y, scale="RD/MD")
    meta.append(m)
    if "se_lrr" in g and g.se_lrr.notna().all():
        ml = meta_dl(np.log(g.rr), g.se_lrr)
        meta.append(
            dict(
                y=y,
                scale="RR",
                est=np.exp(ml["est"]),
                lo=np.exp(ml["lo"]),
                hi=np.exp(ml["hi"]),
                I2=ml["I2"],
                tau2=ml["tau2"],
                k=ml["k"],
            )
        )
meta = pd.DataFrame(meta)
meta.to_csv(f"results_meta_{tag}.csv", index=False)
print(meta.round(4).to_string())

if kind == "hgb" and os.environ.get("SKIPBAL") != "1":
    bal = pd.concat(
        [balance(t[t.cohort == c], trim=TRIM)[0] for c in ["HRS", "ELSA", "SHARE", "CHARLS"]]
    )
    bal.to_csv(f"balance_{tag}.csv", index=False)
    print(bal.groupby("cohort")[["smd_raw", "smd_w"]].agg(lambda s: np.abs(s).max()).round(3))
