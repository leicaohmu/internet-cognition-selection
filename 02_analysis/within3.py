import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from within2 import build, PRECOLS
from ladder import aipw_cols
from aipw import aipw
from pool_hk import hksj

rows = []
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    t = build(c)
    el = t[t.grp.isin(["early", "late"])].copy()
    el["A"] = (el.grp == "early").astype(int)
    sus = el[(el.grp == "late") | ((el.internet_2 == 1) & (el.internet_3 == 1))]
    nonp = el[(el.grp == "late") | ((el.internet_2 == 0))]  # early starters who stopped by t2
    for name, dd in [("sustained", sus), ("stopped_by_t2", nonp)]:
        for y in ["Y_pre", "Y_iadl_pre"]:
            x = dd[dd.recall_lag.notna()] if y == "Y_pre" else dd[dd.iadl_any_lag == 0]
            if x[y].notna().sum() < 150 or x.A.sum() < 30 or (1 - x.A).sum() < 30:
                continue
            r = aipw_cols(x, y, PRECOLS)
            r.update(variant="pretrend_" + name)
            rows.append(r)
            print(
                c,
                name,
                y,
                f"{r['est']:+.3f} ({r['lo']:+.3f},{r['hi']:+.3f}) n={r['n']} A1={r['n_A1']}",
                flush=True,
            )
        if name == "stopped_by_t2":
            for y in ["Y_recall"]:
                x = dd
                if x.A.sum() < 30:
                    continue
                r = aipw(x, y, trim=0.025)
                r.update(variant="stopped_by_t2")
                rows.append(r)
                print(
                    c,
                    "stopped",
                    y,
                    f"{r['est']:+.3f} ({r['lo']:+.3f},{r['hi']:+.3f}) n={r['n']} A1={r['n_A1']}",
                    flush=True,
                )
d = pd.DataFrame(rows)
d.to_csv("results_within3.csv", index=False)
for (v, y), g in d.groupby(["variant", "y"]):
    if len(g) < 2:
        continue
    r = hksj(g.est, g.se)
    print(
        "POOLED",
        v,
        y,
        len(g),
        f"{r['est']:+.3f} DL({r['dl_lo']:+.3f},{r['dl_hi']:+.3f}) HK({r['hk_lo']:+.3f},{r['hk_hi']:+.3f})",
    )
