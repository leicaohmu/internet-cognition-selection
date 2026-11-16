# LASI wave 1 (India) cross-sectional panel, same column names as other cohorts
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
R = os.path.expanduser("~/mnt/小王新/数据/5. LASI 印度/LASI_印度/Raw_data/")
cols = [
    "prim_key",
    "ragender",
    "raeducl",
    "rabyear",
    "r1iwstat",
    "r1proxy",
    "r1agey",
    "r1mstat",
    "hh1rural",
    "r1shlt",
    "r1adla",
    "r1iadlza",
    "r1cesd10",
    "r1tr20",
    "r1smoken",
    "r1vgactx",
    "r1work",
    "hh1cperc",
    "hh1hhres",
    "r1wtresp",
] + [
    f"r1{c}" for c in ["hibpe", "diabe", "cancre", "lunge", "hearte", "stroke", "psyche", "arthre"]
]
h = pd.read_stata(
    R + "Harmonized LASI (A.3)/H_LASI_a3.dta", columns=cols, convert_categoricals=False
)
raw = pd.read_stata(
    R + "LASI Wave 1 (Version B) Data/lasi_w1b_ind_bm.dta",
    columns=["prim_key", "fs514"],
    convert_categoricals=False,
)
print("fs514", raw.fs514.value_counts(dropna=False).to_dict())
d = h.merge(raw, on="prim_key", how="left")
p = pd.DataFrame(
    {
        "id": d.prim_key.astype(str),
        "wave": 1,
        "iwstat": d.r1iwstat,
        "proxy": d.r1proxy,
        "age": d.r1agey,
        "female": (d.ragender == 2).astype(float),
        "edu": d.raeducl - 1,
        "mstat": d.r1mstat,
        "work": d.r1work,
        "shlt": d.r1shlt,
        "adl": d.r1adla,
        "iadl": d.r1iadlza,
        "dep": d.r1cesd10,
        "recall": d.r1tr20,
        "smoke": d.r1smoken,
        "vig_weekly": (d.r1vgactx <= 3).astype(float).where(d.r1vgactx.notna()),
        "wealth": d.hh1cperc,
        "hhsize": d.hh1hhres,
        "rural": d.hh1rural,
        "weight": d.r1wtresp,
        "internet": np.where(
            d.fs514.between(1, 5), 1.0, np.where(d.fs514.isin([6, 7]), 0.0, np.nan)
        ),
    }
)
p["ncond"] = d[
    [
        f"r1{c}"
        for c in ["hibpe", "diabe", "cancre", "lunge", "hearte", "stroke", "psyche", "arthre"]
    ]
].sum(axis=1, min_count=6)
p["cohort"] = "LASI"
p["country"] = "India"
p.to_csv(os.path.expanduser("~/mnt/小王新/analysis/derived/panel_LASI.csv.gz"), index=False)
print(p.shape, p[["internet", "dep", "recall", "iadl", "age"]].describe().round(2).to_string())
