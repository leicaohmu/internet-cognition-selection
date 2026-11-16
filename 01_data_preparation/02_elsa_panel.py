# ELSA long panel: Harmonized ELSA G3 (w2-9) + raw internet items (w2-10)
import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings("ignore")
R = os.path.expanduser("~/mnt/小王新/数据/2. ELSA 英国/Raw_data/")
E = R + "UKDA-5050-stata/stata/stata13_se/"
OUT = os.path.expanduser("~/mnt/小王新/analysis/derived/")
W = range(2, 10)
rstem = [
    "iwstat",
    "proxy",
    "agey",
    "mstat",
    "work",
    "shlt",
    "adla",
    "iadlza",
    "cesd",
    "smoken",
    "drink",
    "vgactx_e",
    "tr20",
    "cancre",
    "hibpe",
    "diabe",
    "lunge",
    "hearte",
    "stroke",
    "psyche",
    "arthre",
    "iwindy",
    "cwtresp",
]
cols = (
    ["idauniq", "ragender", "raeducl", "rabyear"]
    + [f"r{w}{s}" for w in W for s in rstem]
    + [f"h{w}{s}" for w in W for s in ["atotb", "hhres"]]
)
f = R + "Harmonized ELSA/h_elsa_g3.dta"
with pd.io.stata.StataReader(f) as r:
    have = set(r.variable_labels())
cols = [c for c in cols if c in have]
out = []
with pd.io.stata.StataReader(f, convert_categoricals=False, chunksize=4000, columns=cols) as r:
    for ch in r:
        out.append(ch)
df = pd.concat(out, ignore_index=True)
na = pd.Series(np.nan, index=df.index)
rows = []
for w in W:
    g = lambda s: df[f"r{w}{s}"] if f"r{w}{s}" in df else na
    h = lambda s: df[f"h{w}{s}"] if f"h{w}{s}" in df else na
    d = pd.DataFrame(
        {
            "id": df.idauniq.astype("int64").astype(str),
            "wave": w,
            "year": g("iwindy"),
            "iwstat": g("iwstat"),
            "proxy": g("proxy"),
            "age": g("agey"),
            "female": (df.ragender == 2).astype(float),
            "edu": df.raeducl - 1,
            "byear": df.rabyear,
            "mstat": g("mstat"),
            "work": g("work"),
            "shlt": g("shlt"),
            "adl": g("adla"),
            "iadl": g("iadlza"),
            "dep": g("cesd"),
            "smoke": g("smoken"),
            "drink": g("drink"),
            "vig_weekly": (g("vgactx_e") <= 3).where(g("vgactx_e").notna()).astype(float),
            "recall": g("tr20"),
            "cancer": g("cancre"),
            "wealth": h("atotb"),
            "hhsize": h("hhres"),
            "weight": g("cwtresp"),
        }
    )
    d["ncond"] = pd.concat(
        [
            g(c)
            for c in ["hibpe", "diabe", "cancre", "lunge", "hearte", "stroke", "psyche", "arthre"]
        ],
        axis=1,
    ).sum(axis=1, min_count=6)
    rows.append(d)
p = pd.concat(rows, ignore_index=True)
p = p[p.iwstat.notna() & (p.iwstat != 0)]
# internet from raw core files
files = {
    2: ("wave2/wave_2_core_data_v4.dta", "scpt07"),
    3: ("wave3/wave_3_elsa_data_v4.dta", "scpt07"),
    4: ("wave4/wave_4_elsa_data_v3.dta", "scpt06"),
    5: ("wave5/wave_5_elsa_data_v4.dta", "scpt06"),
    6: ("wave6/wave_6_elsa_data_v2.dta", "scint"),
    7: (None, "scint"),
    8: ("wave8/wave_8_elsa_data_eul_v2.dta", "scint"),
    9: ("wave9/wave_9_elsa_data_eul_v1.dta", "scint"),
    10: ("wave10/wave_10_elsa_data_eul_v1.dta", "SCINT"),
}
net = []
for w, (fn, v) in files.items():
    pth = E + fn if fn else glob.glob(E + f"wave{w}/*elsa_data*.dta")[0]
    d = pd.read_stata(pth, columns=["idauniq", v], convert_categoricals=False)
    x = d[v]
    val = (
        np.where(x.between(1, 5), 1, np.where(x == 6, 0, np.nan))
        if v.lower() == "scint"
        else np.where(x == 1, 1, np.where(x == 0, 0, np.nan))
    )
    net.append(
        pd.DataFrame({"id": d.idauniq.astype("int64").astype(str), "wave": w, "internet": val})
    )
net = pd.concat(net)
p = p.merge(
    net, on=["id", "wave"], how="outer"
)  # keeps w10 internet rows (outcomes for w10 added separately)
p["cohort"] = "ELSA"
p["country"] = "England"
p.to_csv(OUT + "panel_ELSA.csv.gz", index=False)
print(p.shape)
print(p.groupby("wave")[["internet", "dep", "recall", "iadl", "wealth", "age"]].count())
