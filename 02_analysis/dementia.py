"""v1.1: HRS dementia (Langa-Weir, imputed 27-point score <=6 among self-respondents) with the four designs."""

import os
import sys

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
os.environ["OUTS"] = "dem,cind"
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import newout

F = "/mnt/user-data/uploads/小王新/数据/3. HRS  美国/HRS_美国/Raw_data/Cross-Wave Imputation of Cognitive Functioning Measures 1992-2020/COGIMP9220A_R.dta"
d = pd.read_stata(F, convert_categoricals=False)
d["id"] = (d.HHID.astype(int) * 1000 + d.PN.astype(int)).astype(str)
L = []
for w in range(6, 16):
    if f"R{w}COG27" in d:
        L.append(pd.DataFrame({"id": d.id, "wave": w, "cog27i": d[f"R{w}COG27"]}))
L = pd.concat(L).dropna()
_panel = newout.panel


def panel(c):
    p = _panel(c).merge(L, on=["id", "wave"], how="left")
    p.loc[p.recall_web.fillna(0) == 1, "cog27i"] = np.nan
    p["dem"] = (p.cog27i <= 6).astype(float).where(p.cog27i.notna())
    p["cind"] = (p.cog27i <= 11).astype(float).where(p.cog27i.notna())
    return p


_oc = newout.outcome_cols
newout.outcome_cols = lambda x: x if x in ("dem", "cind") else _oc(x)
newout.BIN.update({"dem", "cind"})

if __name__ == "__main__":
    p = panel("HRS")
    print(
        "matched",
        p.cog27i.notna().mean().round(3),
        "dementia prevalence",
        p.dem.mean().round(3),
        flush=True,
    )
    rows = newout.tte("HRS", p) + newout.falsif("HRS", p) + newout.pretrend("HRS", p)
    pd.DataFrame(rows).to_csv("results_dementia_HRS.csv", index=False)
    print(
        pd.DataFrame(rows)[["design", "outcome", "est", "lo", "hi", "rr", "rr_lo", "rr_hi", "n"]]
        .round(4)
        .to_string()
    )
