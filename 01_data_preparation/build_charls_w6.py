"""Extend CHARLS files with wave 5 (2020) items not previously built and with wave 6 (fieldwork 2021-2023).
Wave 6: single-trial 10-word immediate + delayed recall (comparable with 2011-2018), CES-D 10, ADL/IADL, self-rated health,
chronic conditions (new diagnoses added to prior status), hospital stay, weekly vigorous activity, serial 7s, life satisfaction,
internet use (WeChat, online chatting or any online activity), proxy, death (2023 sample file).
Wave 5 additions: self-rated health, hospital stay, weekly vigorous activity, serial 7s, life satisfaction (raw 2020).
"""

import pandas as pd
import numpy as np

W6 = "/tmp/w6/CHARLS-W6-{}/"
X20 = "/mnt/user-data/uploads/小王新/analysis/derived/charls2020_extra.csv"


def z12(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)


def ser7(df, cols):
    if not all(c in df for c in cols):
        return pd.Series(np.nan, index=df.index)
    prev = pd.Series(100.0, index=df.index)
    n = pd.Series(0.0, index=df.index)
    any_ = df[cols].notna().any(axis=1)
    for c in cols:
        a = df[c].where(df[c] < 900)
        n += (a == prev - 7).astype(float)
        prev = a.where(a.notna(), prev)
    return n.where(any_)


# ---------------- wave 6 ----------------
V = {"2021": ("dc039", "dc061"), "2022": ("dc151", "dc153"), "2023": ("dc151", "dc153")}
S7 = {
    "2021": [f"dc0{i}_1" for i in range(17, 22)],
    "2022": [f"dc1{i}_1" for i in range(28, 33)],
    "2023": [f"dc1{i}_1" for i in range(28, 33)],
}
COND = {
    1: "hyp",
    3: "diab",
    4: "cancer",
    5: "lung",
    7: "heart",
    8: "stroke",
    11: "psych",
    15: "arth",
}
parts = []
for y, (a, b) in V.items():
    d = W6.format(y)
    c = pd.read_stata(d + "Cognition_Depression.dta", convert_categoricals=False)
    A = c[[f"{a}_s{i}" for i in range(1, 13)]]
    B = c[[f"{b}_s{i}" for i in range(1, 13)]]
    okA = A.notna().any(axis=1) & A[f"{a}_s12"].fillna(0).eq(0)
    okB = B.notna().any(axis=1) & B[f"{b}_s12"].fillna(0).eq(0)
    rec = (
        A.iloc[:, :10].fillna(0).gt(0).sum(axis=1) + B.iloc[:, :10].fillna(0).gt(0).sum(axis=1)
    ).where(okA & okB)
    it = c[[f"dc0{i}" for i in range(51, 61)]].where(lambda x: (x >= 1) & (x <= 4)) - 1
    for q in ["dc055", "dc058"]:
        it[q] = 3 - it[q]
    cog = pd.DataFrame(
        {
            "ID": c.ID,
            "recall": rec,
            "dep": it.sum(axis=1, min_count=10),
            "ser7": ser7(c, S7[y]),
            "lsat": 6 - c.dc062.where(c.dc062.between(1, 5)),
        }
    )
    hcols = (
        ["ID", "da001", "da082_1_", "da091", "da097", "da098_s7"]
        + [f"db0{i:02d}" for i in [1, 3, 5, 7, 9, 11, 12, 14, 16, 20, 22]]
        + [f"da011_{k}_" for k in COND]
    )
    h = pd.read_stata(
        d + "Health_Status_and_Functioning.dta", columns=hcols, convert_categoricals=False
    )
    dm = pd.read_stata(d + "Demographics.dta", columns=["ID", "xrtype"], convert_categoricals=False)
    hc = pd.read_stata(
        d + "Health_Care_Insurance.dta", columns=["ID", "ec001"], convert_categoricals=False
    )
    x = (
        cog.merge(h, on="ID", how="outer")
        .merge(dm, on="ID", how="left")
        .merge(hc, on="ID", how="left")
    )
    online = x.da091.eq(1) | x.da097.isin([1, 2]) | (x.da098_s7.notna() & x.da098_s7.ne(7))
    x["internet"] = np.where(
        online, 1.0, np.where(x.da091.notna() | x.da098_s7.notna(), 0.0, np.nan)
    )
    adl = x[[f"db0{i:02d}" for i in [1, 3, 5, 7, 9, 11]]]
    iadl = x[[f"db0{i:02d}" for i in [12, 14, 16, 20, 22]]]
    x["adl"] = adl.where((adl >= 1) & (adl <= 4)).ge(2).sum(axis=1).where(adl.notna().all(axis=1))
    x["iadl"] = (
        iadl.where((iadl >= 1) & (iadl <= 4)).ge(2).sum(axis=1).where(iadl.notna().all(axis=1))
    )
    x["shlt"] = (x.da001.where(x.da001.between(1, 5)) + 1).clip(
        upper=5
    )  # Chinese 5-point scale -> harmonized coding (fair or worse = 4-5)
    x["vig_weekly"] = x.da082_1_.map({1: 1.0, 2: 0.0})
    x["hosp"] = x.ec001.map({1: 1.0, 2: 0.0})
    for k, nm in COND.items():
        x["new_" + nm] = x[f"da011_{k}_"].map({1: 1.0, 2: 0.0})
    x["proxy"] = (x.xrtype == 2).astype(float)
    x["year"] = float(y)
    x["part"] = int(y)
    parts.append(x)
w = pd.concat(parts, ignore_index=True)
w["ID"] = z12(w.ID)
w = w.sort_values(["ID", "part"]).groupby("ID", as_index=False).first()
P = pd.read_csv("panel_CHARLS.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
P = P[P.wave <= 5].copy()
# time-invariant and prior chronic-condition status from the harmonized waves
H = (
    pd.read_stata(
        "/mnt/user-data/uploads/H_CHARLS_D_Data.dta",
        columns=["ID", "rabyear"],
        convert_categoricals=False,
    )
    if False
    else None
)
first = P.dropna(subset=["female"]).drop_duplicates("id").set_index("id")
last4 = (
    P[(P.wave <= 4) & P.ncond.notna()]
    .sort_values("wave")
    .drop_duplicates("id", keep="last")
    .set_index("id")
)
canc_prev = P[P.cancer.notna()].sort_values("wave").groupby("id").cancer.max()
w = w.set_index("ID")
new8 = w[[c for c in w if c.startswith("new_")]]
w["ncond"] = (last4.ncond.reindex(w.index) + new8.fillna(0).sum(axis=1)).where(
    last4.ncond.reindex(w.index).notna()
)
w["cancer"] = np.where(
    w.new_cancer == 1,
    1.0,
    np.where(
        canc_prev.reindex(w.index) == 1,
        1.0,
        np.where(w.new_cancer == 0, 0.0, canc_prev.reindex(w.index)),
    ),
)
w6 = pd.DataFrame(
    {
        "id": w.index,
        "wave": 6,
        "year": w.year.values,
        "iwstat": 1.0,
        "proxy": w.proxy.values,
        "female": first.female.reindex(w.index).values,
        "edu": first.edu.reindex(w.index).values,
        "byear": first.byear.reindex(w.index).values,
        "shlt": w.shlt.values,
        "adl": w.adl.values,
        "iadl": w.iadl.values,
        "dep": w.dep.values,
        "vig_weekly": w.vig_weekly.values,
        "recall": w.recall.values,
        "cancer": w.cancer.values,
        "ncond": w.ncond.values,
        "internet": w.internet.values,
        "cohort": "CHARLS",
        "country": "China",
    }
)
w6["age"] = w6.year - w6.byear
si = pd.read_stata(W6.format("2023") + "Sample_Infor.dta", convert_categoricals=False)
dead = z12(si.loc[si.died == 1, "ID"])
dead = dead[~dead.isin(w6.id)].unique()
w6 = pd.concat(
    [
        w6,
        pd.DataFrame(
            {"id": dead, "wave": 6, "iwstat": 5.0, "cohort": "CHARLS", "country": "China"}
        ),
    ],
    ignore_index=True,
)
# ---------------- wave 5 additions ----------------
x20 = pd.read_csv(X20)
x20["id"] = z12(x20.ID)
s7 = ser7(x20, ["dc007_1_1", "dc007_2_1", "dc007_3_1", "dc007_4_1", "dc007_5_1"])
add5 = pd.DataFrame(
    {
        "id": x20.id,
        "shlt5": (x20.da001.where(x20.da001.between(1, 5)) + 1).clip(upper=5),
        "vig5": x20.da032_1_.map({1: 1.0, 2: 0.0}),
        "hosp5": x20.da007.map({1: 1.0, 2: 0.0}),
        "ser7_5": s7,
        "lsat5": 6 - x20.dc026.where(x20.dc026.between(1, 5)),
    }
).drop_duplicates("id")
m5 = P.wave == 5
for col in ["female", "edu", "byear"]:
    P.loc[m5, col] = P.loc[m5, "id"].map(first[col])
P.loc[m5, "age"] = 2020 - P.loc[m5, "byear"]
P = P.merge(add5[["id", "shlt5", "vig5"]], on="id", how="left")
P.loc[m5.values, "shlt"] = P.loc[m5.values, "shlt5"]
P.loc[m5.values, "vig_weekly"] = P.loc[m5.values, "vig5"]
P = P.drop(columns=["shlt5", "vig5"])
P = pd.concat([P, w6], ignore_index=True)
P.to_csv("panel_CHARLS.csv.gz", index=False)
# extra (hospital stay) and newout (serial 7s, life satisfaction)
E = pd.read_csv("extra_CHARLS.csv.gz", dtype={"id": str, "sp_id": str}, low_memory=False)
E = E[E.wave <= 4]
E = pd.concat(
    [
        E,
        add5.assign(wave=5)[["id", "wave", "hosp5"]].rename(columns={"hosp5": "hosp"}),
        pd.DataFrame({"id": w.index, "wave": 6, "hosp": w.hosp.values}),
    ],
    ignore_index=True,
)
E.to_csv("extra_CHARLS.csv.gz", index=False)
N = pd.read_csv("newout_CHARLS.csv.gz", dtype={"id": str}, low_memory=False)
N = N[N.wave <= 4]
N = pd.concat(
    [
        N,
        add5.assign(wave=5)[["id", "wave", "ser7_5", "lsat5"]].rename(
            columns={"ser7_5": "ser7", "lsat5": "lsat"}
        ),
        pd.DataFrame({"id": w.index, "wave": 6, "ser7": w.ser7.values, "lsat": w.lsat.values}),
    ],
    ignore_index=True,
)
N.to_csv("newout_CHARLS.csv.gz", index=False)
r = P[P.iwstat == 1]
print(
    r.groupby("wave")[
        ["recall", "dep", "iadl", "adl", "shlt", "ncond", "cancer", "vig_weekly", "internet", "age"]
    ]
    .agg(["count", "mean"])
    .round(2)
    .T.to_string()
)
print("deaths w6", int((P[(P.wave == 6)].iwstat == 5).sum()))
print(N.groupby("wave")[["ser7", "lsat"]].agg(["count", "mean"]).round(2))
print(E.groupby("wave").hosp.agg(["count", "mean"]).round(3))
