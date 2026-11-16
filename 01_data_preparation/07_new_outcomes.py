# Extra outcomes for v0.9: serial 7s, verbal fluency, orientation, HRS 27-pt cognition, life satisfaction (+CASP)
import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings("ignore")
D = os.path.expanduser("~/mnt/小王新/数据/")
OUT = os.path.expanduser("~/mnt/小王新/analysis/derived/")


def rd(f, want):
    with pd.io.stata.StataReader(f) as r:
        have = set(r.variable_labels())
    cols = [c for c in want if c in have]
    out = []
    with pd.io.stata.StataReader(f, convert_categoricals=False, chunksize=20000, columns=cols) as r:
        for ch in r:
            out.append(ch)
    return pd.concat(out, ignore_index=True)


def long(df, idcol, W, m):
    rows = []
    for w in W:
        d = {"id": df[idcol], "wave": w}
        for k, v in m.items():
            col = v.format(w=w)
            d[k] = df[col] if col in df else np.nan
        rows.append(pd.DataFrame(d))
    o = pd.concat(rows, ignore_index=True)
    num = [c for c in o if c not in ("id", "wave")]
    o[num] = o[num].apply(pd.to_numeric, errors="coerce").mask(lambda x: x < 0)
    return o


# ---------------- HRS
H = D + "3. HRS  美国/HRS_美国/Raw_data/"
W = range(6, 16)
rv = ["hhidpn"] + [
    f"r{w}{s}"
    for w in W
    for s in [
        "ser7",
        "ser7p",
        "fser7",
        "fser7p",
        "cog27",
        "cog27p",
        "fimrc",
        "fdlrc",
        "fbwc20",
        "fimrcp",
        "fdlrcp",
        "fbwc20p",
    ]
]
r = rd(H + "RAND HRS Longitudinal File 2020/randhrs1992_2020v2.dta", rv)
g = rd(
    H + "Gateway Harmonized HRS/H_HRS_d.dta",
    ["hhidpn"] + [f"r{w}{s}" for w in W for s in ["orient", "lsatsc", "verbf"]],
)
df = r.merge(g, on="hhidpn", how="left")
df["id"] = df.hhidpn.astype("int64").astype(str)
rows = []
for w in W:
    s = lambda k: df[f"r{w}{k}"] if f"r{w}{k}" in df else pd.Series(np.nan, index=df.index)
    ser7 = s("ser7").fillna(s("ser7p"))
    fs = s("fser7").fillna(s("fser7p"))
    cog = s("cog27").fillna(s("cog27p"))
    imp = (
        (s("fimrc").fillna(s("fimrcp")).fillna(0) > 0)
        | (s("fdlrc").fillna(s("fdlrcp")).fillna(0) > 0)
        | (s("fbwc20").fillna(s("fbwc20p")).fillna(0) > 0)
        | (fs.fillna(0) > 0)
    )
    rows.append(
        pd.DataFrame(
            {
                "id": df.id,
                "wave": w,
                "ser7": ser7.where(fs.fillna(0) == 0),
                "cog27": cog.where(~imp),
                "orient": s("orient"),
                "lsat": s("lsatsc"),
                "verbf": s("verbf"),
            }
        )
    )
h = pd.concat(rows, ignore_index=True)
n = [c for c in h if c not in ("id", "wave")]
h[n] = h[n].apply(pd.to_numeric, errors="coerce").mask(lambda x: x < 0)
h.to_csv(OUT + "newout_HRS.csv.gz", index=False)
print("HRS", h.groupby("wave")[n].count().to_string())
# ---------------- ELSA
f = glob.glob(D + "2. ELSA 英国/**/h_elsa_g3.dta", recursive=True)[0]
W = range(1, 10)
e = rd(
    f,
    ["idauniq"] + [f"r{w}{s}" for w in W for s in ["verbf", "ser7", "orient", "lsatsc", "casp19"]],
)
e["id"] = e.idauniq.astype("int64").astype(str)
o = long(
    e,
    "id",
    W,
    {
        "verbf": "r{w}verbf",
        "ser7": "r{w}ser7",
        "orient": "r{w}orient",
        "lsat": "r{w}lsatsc",
        "casp": "r{w}casp19",
    },
)
o.to_csv(OUT + "newout_ELSA.csv.gz", index=False)
print("ELSA", o.groupby("wave").count().to_string())
# ---------------- SHARE
S = D + "7.SHARE 欧洲/SHARE_欧洲/Raw_data/"
es = rd(
    S + "easySHARE Release 9.0.0/easySHARE_rel9-0-0.dta",
    ["mergeid", "wave", "numeracy_2", "orienti", "casp"],
)
es = es.rename(columns={"mergeid": "id", "numeracy_2": "ser7", "orienti": "orient"})
es[["ser7", "orient", "casp"]] = (
    es[["ser7", "orient", "casp"]].apply(pd.to_numeric, errors="coerce").mask(lambda x: x < 0)
)
hs = rd(
    S + "Harmonized SHARE/H_SHARE_f2.dta",
    ["mergeid"] + [f"r{w}{s}" for w in [4, 5, 6, 7, 8] for s in ["verbf", "satlife_s"]],
)
hl = (
    long(hs, "mergeid", [4, 5, 6, 7, 8], {"verbf": "r{w}verbf", "lsat": "r{w}satlife_s"}).rename(
        columns={"mergeid": "id"}
    )
    if "mergeid" in hs
    else None
)
hl = long(
    hs.rename(columns={"mergeid": "id"}),
    "id",
    [4, 5, 6, 7, 8],
    {"verbf": "r{w}verbf", "lsat": "r{w}satlife_s"},
)
w9 = []
for mod, cols in [("cf", ["mergeid", "cf010_"]), ("ac", ["mergeid", "ac012_"])]:
    fn = glob.glob(S + f"Wave 9 Release 9.0.0/sharew9_rel9-0-0_{mod}.dta")
    if fn:
        x = rd(fn[0], cols)
        w9.append(x.set_index("mergeid"))
if w9:
    x = (
        pd.concat(w9, axis=1)
        .reset_index()
        .rename(columns={"mergeid": "id", "cf010_": "verbf", "ac012_": "lsat"})
    )
    x["wave"] = 9
    for c in ["verbf", "lsat"]:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce").mask(lambda v: v < 0)
    hl = pd.concat(
        [hl, x[[c for c in ["id", "wave", "verbf", "lsat"] if c in x]]], ignore_index=True
    )
s = es[es.wave >= 4].merge(hl, on=["id", "wave"], how="outer")
s.to_csv(OUT + "newout_SHARE.csv.gz", index=False)
print("SHARE", s.groupby("wave").count().to_string())
# ---------------- CHARLS
c = rd(
    D + "H_CHARLS_D_Data.dta",
    ["ID"] + [f"r{w}{s}" for w in range(1, 5) for s in ["ser7", "orient", "satlife", "draw"]],
)
c["id"] = c.ID.astype(str)
o = long(
    c,
    "id",
    range(1, 5),
    {"ser7": "r{w}ser7", "orient": "r{w}orient", "lsat": "r{w}satlife", "draw": "r{w}draw"},
)
o.to_csv(OUT + "newout_CHARLS.csv.gz", index=False)
print("CHARLS", o.groupby("wave").count().to_string())
