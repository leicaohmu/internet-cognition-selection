# Build HRS long person-wave panel (waves 6-15, 2002-2020) from RAND HRS 2020 + Gateway Harmonized HRS D
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
B = os.path.expanduser("~/mnt/小王新/数据/3. HRS  美国/HRS_美国/Raw_data/")
OUT = os.path.expanduser("~/mnt/小王新/analysis/derived/")
W = range(6, 16)
rstem = [
    "iwstat",
    "proxy",
    "agey_e",
    "mstat",
    "work",
    "shlt",
    "adl5a",
    "iadl5a",
    "cesd",
    "smoken",
    "drink",
    "vgactx",
    "tr20",
    "cancre",
    "hibpe",
    "diabe",
    "lunge",
    "hearte",
    "stroke",
    "psyche",
    "arthre",
    "wtresp",
    "iwendy",
]
hstem = ["atotb", "hhres"]
cols = (
    ["hhidpn", "ragender", "raeduc", "rabyear"]
    + [f"r{w}{s}" for w in W for s in rstem]
    + [f"h{w}{s}" for w in W for s in hstem]
)


def read(f, cols):
    with pd.io.stata.StataReader(f, convert_categoricals=False) as r:
        have = set(r.variable_labels())
    cols = [c for c in cols if c in have]
    out = []
    with pd.io.stata.StataReader(f, convert_categoricals=False, chunksize=4000, columns=cols) as r:
        for ch in r:
            out.append(ch)
    return pd.concat(out, ignore_index=True)


rand = read(B + "RAND HRS Longitudinal File 2020/randhrs1992_2020v2.dta", cols)
gw = read(
    B + "Gateway Harmonized HRS/H_HRS_d.dta",
    ["hhidpn", "raeducl"] + [f"r{w}email" for w in W] + [f"h{w}rural" for w in W],
)
df = rand.merge(gw, on="hhidpn", how="left")
rows = []
for w in W:
    na = pd.Series(np.nan, index=df.index)
    g = lambda s: df[f"r{w}{s}"] if f"r{w}{s}" in df else na
    d = pd.DataFrame(
        {
            "id": df.hhidpn.astype("int64").astype(str),
            "wave": w,
            "year": g("iwendy"),
            "iwstat": g("iwstat"),
            "proxy": g("proxy"),
            "age": g("agey_e"),
            "female": (df.ragender == 2).astype(float),
            "edu": df.raeducl - 1,
            "byear": df.rabyear,
            "mstat": g("mstat"),
            "work": g("work"),
            "shlt": g("shlt"),
            "adl": g("adl5a"),
            "iadl": g("iadl5a"),
            "dep": g("cesd"),
            "smoke": g("smoken"),
            "drink": g("drink"),
            "vig_weekly": (g("vgactx") <= 3).where(g("vgactx").notna()).astype(float),
            "recall": g("tr20"),
            "cancer": g("cancre"),
            "wealth": (df[f"h{w}atotb"] if f"h{w}atotb" in df else na),
            "hhsize": (df[f"h{w}hhres"] if f"h{w}hhres" in df else na),
            "rural": (df[f"h{w}rural"] if f"h{w}rural" in df else na),
            "weight": g("wtresp"),
            "internet": (df[f"r{w}email"] if f"r{w}email" in df else na),
        }
    )
    conds = [
        g(c) for c in ["hibpe", "diabe", "cancre", "lunge", "hearte", "stroke", "psyche", "arthre"]
    ]
    d["ncond"] = pd.concat(conds, axis=1).sum(axis=1, min_count=6)
    rows.append(d)
p = pd.concat(rows, ignore_index=True)
p = p[p.iwstat.notna() & (p.iwstat != 0)]
p["cohort"] = "HRS"
p["country"] = "US"
p.to_csv(OUT + "panel_HRS.csv.gz", index=False)
print(p.shape)
print(
    p.groupby("wave")[["internet", "dep", "recall", "iadl", "wealth"]].apply(
        lambda x: x.notna().mean().round(2)
    )
)
print(p.iwstat.value_counts().to_dict())

# ---- replace recall with cross-wave cognition file (non-imputed values; flag web mode) ----
C = B + "Cross-Wave Imputation of Cognitive Functioning Measures 1992-2020/COGIMP9220A_R.dta"
with pd.io.stata.StataReader(C) as r:
    have = set(r.variable_labels())
want = ["HHID", "PN"] + [
    c
    for w in W
    for c in [
        f"R{w}TR20",
        f"R{w}FIMRC",
        f"R{w}FDLRC",
        f"R{w}TR20P",
        f"R{w}TR20W",
        f"R{w}FIMRCP",
        f"R{w}FDLRCP",
        f"R{w}FIMRCW",
        f"R{w}FDLRCW",
    ]
    if c in have
]
cg = pd.read_stata(C, columns=want, convert_categoricals=False)
cg["id"] = (
    cg.HHID.astype(int).astype(str) + cg.PN.astype(int).astype(str).str.zfill(3)
).str.lstrip("0")
rows = []
for w in W:
    if f"R{w}TR20" in cg:
        rec = cg[f"R{w}TR20"]
        imp = (cg[f"R{w}FIMRC"].fillna(0) > 0) | (cg[f"R{w}FDLRC"].fillna(0) > 0)
        web = pd.Series(0.0, index=cg.index)
    else:
        P, Wb = cg[f"R{w}TR20P"], cg[f"R{w}TR20W"]
        rec = P.fillna(Wb)
        web = (P.isna() & Wb.notna()).astype(float)
        impP = (cg[f"R{w}FIMRCP"].fillna(0) > 0) | (cg[f"R{w}FDLRCP"].fillna(0) > 0)
        impW = (cg[f"R{w}FIMRCW"].fillna(0) > 0) | (cg[f"R{w}FDLRCW"].fillna(0) > 0)
        imp = np.where(web == 1, impW, impP)
    rows.append(
        pd.DataFrame(
            {
                "id": cg.id,
                "wave": w,
                "recall2": rec.where(~pd.Series(imp, index=cg.index)),
                "recall_web": web.where(rec.notna()),
            }
        )
    )
cr = pd.concat(rows)
p["id"] = p.id.str.lstrip("0")
p = (
    p.drop(columns="recall")
    .merge(cr, on=["id", "wave"], how="left")
    .rename(columns={"recall2": "recall"})
)
p.to_csv(OUT + "panel_HRS.csv.gz", index=False)
print("after cog merge", p.shape)
print(p[p.iwstat == 1].groupby("wave")[["recall", "recall_web"]].agg(["count", "mean"]).round(2))
print("internet x web (w14-15):")
print(pd.crosstab(p.loc[p.wave >= 14, "internet"], p.loc[p.wave >= 14, "recall_web"]))
