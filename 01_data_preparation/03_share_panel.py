# SHARE long panel: easySHARE rel 9.0.0 (w1-9) + IT module internet (w5-9)
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
S = os.path.expanduser("~/mnt/小王新/数据/7.SHARE 欧洲/SHARE_欧洲/Raw_data/")
OUT = os.path.expanduser("~/mnt/小王新/analysis/derived/")
keep = [
    "mergeid",
    "wave",
    "int_year",
    "country",
    "female",
    "dn003_mod",
    "age",
    "isced1997_r",
    "mar_stat",
    "hhsize",
    "partnerinhh",
    "sphus",
    "chronic_mod",
    "eurod",
    "adla",
    "iadlza",
    "recall_1",
    "recall_2",
    "smoking",
    "br010_mod",
    "br015_",
    "ep005_",
    "thinc_m",
] + [f"income_pct_w{w}" for w in [1, 2, 4, 5, 6, 7, 8, 9]]
out = []
with pd.io.stata.StataReader(
    S + "easySHARE Release 9.0.0/easySHARE_rel9-0-0.dta",
    convert_categoricals=False,
    chunksize=20000,
    columns=keep,
) as r:
    for ch in r:
        out.append(ch[ch.wave >= 4])
e = pd.concat(out, ignore_index=True)
num = e.columns.drop("mergeid")
e[num] = e[num].apply(pd.to_numeric, errors="coerce")
e[num] = e[num].mask(e[num] < 0)
pct = np.select(
    [e.wave == w for w in [4, 5, 6, 7, 8, 9]],
    [e[f"income_pct_w{w}"] for w in [4, 5, 6, 7, 8, 9]],
    np.nan,
)
isc = e.isced1997_r.where(e.isced1997_r <= 6)
p = pd.DataFrame(
    {
        "id": e.mergeid,
        "wave": e.wave,
        "year": e.int_year,
        "iwstat": 1.0,
        "proxy": np.nan,
        "age": e.age,
        "female": e.female,
        "edu": np.select([isc <= 2, isc <= 4, isc >= 5], [0, 1, 2], np.nan).astype(float),
        "byear": e.dn003_mod,
        "mstat": e.mar_stat,
        "partnered_raw": e.partnerinhh,
        "work": (e.ep005_ == 2).astype(float).where(e.ep005_.notna()),
        "shlt": e.sphus,
        "adl": e.adla,
        "iadl": e.iadlza,
        "dep": e.eurod,
        "smoke": (e.smoking == 1).astype(float).where(e.smoking.notna()),
        "drink": (e.br010_mod > 1).astype(float).where(e.br010_mod.notna()),
        "vig_weekly": (e.br015_ <= 2).astype(float).where(e.br015_.notna()),
        "recall": e.recall_1 + e.recall_2,
        "cancer": np.nan,
        "ncond": e.chronic_mod,
        "wealth": pct,
        "hhsize": e.hhsize,
        "country": e.country.astype("Int64").astype(str),
    }
)
p["wealth_is_pct"] = 1
net = []
for w in [5, 6, 7, 8, 9]:
    d = pd.read_stata(
        S + f"Wave {w} Release 9.0.0/sharew{w}_rel9-0-0_it.dta",
        columns=["mergeid", "it004_", "it003_"],
        convert_categoricals=False,
    )
    net.append(
        pd.DataFrame(
            {
                "id": d.mergeid,
                "wave": w,
                "internet": d.it004_.map({1: 1, 5: 0}),
                "pc_skill": d.it003_.where(d.it003_ > 0),
            }
        )
    )
p = p.merge(pd.concat(net), on=["id", "wave"], how="left")
p["cohort"] = "SHARE"
p.to_csv(OUT + "panel_SHARE.csv.gz", index=False)
print(p.shape)
print(p.groupby("wave")[["internet", "dep", "recall", "iadl", "wealth", "edu"]].count())
print(p.country.value_counts().head(30).to_dict())
