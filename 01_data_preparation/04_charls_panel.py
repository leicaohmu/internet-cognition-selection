# CHARLS long panel: Harmonized CHARLS D (w1-4: 2011-2018) + raw internet (2011-2020) + 2020 outcomes built from raw
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
D = os.path.expanduser("~/mnt/小王新/数据/")
C = D + "chalrs数据/"
OUT = os.path.expanduser("~/mnt/小王新/analysis/derived/")
W = range(1, 5)
rstem = [
    "iwstat",
    "proxy",
    "agey",
    "mstat",
    "work",
    "shlt",
    "adlab_c",
    "iadlza",
    "cesd10",
    "smoken",
    "drinkl",
    "vgact_c",
    "tr20",
    "cancre",
    "hibpe",
    "diabe",
    "lunge",
    "hearte",
    "stroke",
    "psyche",
    "arthre",
    "iwy",
    "wtrespb",
]
f = D + "H_CHARLS_D_Data.dta"
with pd.io.stata.StataReader(f) as r:
    have = set(r.variable_labels())
cols = [
    c
    for c in ["ID", "ragender", "raeducl", "rabyear"]
    + [f"r{w}{s}" for w in W for s in rstem]
    + [f"hh{w}cperc" for w in W]
    + [f"h{w}hhres" for w in W]
    + [f"h{w}rural" for w in W]
    if c in have
]
df = pd.read_stata(f, columns=cols, convert_categoricals=False)
na = pd.Series(np.nan, index=df.index)
rows = []
for w in W:
    g = lambda s: df[f"r{w}{s}"] if f"r{w}{s}" in df else na
    d = pd.DataFrame(
        {
            "id": df.ID.astype(str),
            "wave": w,
            "year": g("iwy"),
            "iwstat": g("iwstat"),
            "proxy": g("proxy"),
            "age": g("agey"),
            "female": (df.ragender == 2).astype(float),
            "edu": df.raeducl - 1,
            "byear": df.rabyear,
            "mstat": g("mstat"),
            "work": g("work"),
            "shlt": g("shlt"),
            "adl": g("adlab_c"),
            "iadl": g("iadlza"),
            "dep": g("cesd10"),
            "smoke": g("smoken"),
            "drink": g("drinkl"),
            "vig_weekly": g("vgact_c"),
            "recall": g("tr20"),
            "cancer": g("cancre"),
            "wealth": df.get(f"hh{w}cperc", na),
            "hhsize": df.get(f"h{w}hhres", na),
            "rural": df.get(f"h{w}rural", na),
            "weight": g("wtrespb"),
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
# ---- 2020 (wave 5) outcomes from raw ----
R = D + "CHARLS2020r/"
hs = pd.read_stata(
    R + "Health_Status_and_Functioning.dta",
    convert_categoricals=False,
    columns=["ID"]
    + [f"dc0{i}" for i in range(16, 26)]
    + [f"dc012_s{i}" for i in range(1, 14)]
    + [f"dc028_s{i}" for i in range(1, 13)]
    + [
        "db012",
        "db014",
        "db016",
        "db020",
        "db022",
        "db001",
        "db003",
        "db005",
        "db007",
        "db009",
        "db011",
        "da003_4_",
        "da040",
        "xrtype",
    ],
)
it = hs[[f"dc0{i}" for i in range(16, 26)]].where(lambda x: (x >= 1) & (x <= 4)) - 1
for c in ["dc020", "dc023"]:
    it[c] = 3 - it[c]  # reverse-coded positive items
cesd = it.sum(axis=1, min_count=10)
imm = (
    hs[[f"dc012_s{i}" for i in range(1, 11)]]
    .gt(0)
    .sum(axis=1)
    .where(hs.dc012_s1.notna() & hs.dc012_s12.fillna(0).eq(0) & hs.dc012_s13.fillna(0).eq(0))
)
dly = (
    hs[[f"dc028_s{i}" for i in range(1, 11)]]
    .gt(0)
    .sum(axis=1)
    .where(hs.dc028_s1.notna() & hs.dc028_s12.fillna(0).eq(0))
)
iadl = (
    hs[["db012", "db014", "db016", "db020", "db022"]]
    .where(lambda x: (x >= 1) & (x <= 4))
    .ge(2)
    .sum(axis=1, min_count=5)
    .where(hs[["db012", "db014", "db016", "db020", "db022"]].notna().all(axis=1))
)
adl = (
    hs[["db001", "db003", "db005", "db007", "db009", "db011"]]
    .where(lambda x: (x >= 1) & (x <= 4))
    .ge(2)
    .sum(axis=1)
    .where(hs[["db001", "db003", "db005", "db007", "db009", "db011"]].notna().all(axis=1))
)
si = pd.read_stata(R + "Sample_Infor.dta", convert_categoricals=False)
w5 = pd.DataFrame(
    {
        "id": hs.ID.astype(str),
        "wave": 5,
        "year": 2020.0,
        "iwstat": 1.0,
        "proxy": (hs.xrtype == 2).astype(float),
        "dep": cesd,
        "recall": imm + dly,
        "iadl": iadl,
        "adl": adl,
        "cancer": hs.da003_4_.map({1: 1, 2: 0}),
        "internet": hs.da040.map({1: 1, 2: 0}),
    }
)
dead = si.loc[si.died == 1, "ID"].astype(str)
w5 = pd.concat([w5, pd.DataFrame({"id": dead.values, "wave": 5, "iwstat": 5.0})])
# ---- internet 2011-2018 from raw da056 (option 10) ----
net = []
for w, (fn, pre, conv) in {
    1: ("2011/health_status_and_functioning.dta", "da056s", True),
    2: ("2013/CHARLS2013_Dataset.zip", "da056s", False),
    3: ("2015/CHARLS2015r.zip", "da056s", False),
    4: ("2018/CHARLS2018r.zip", "da056_s", False),
}.items():
    cc = ["ID"] + [f"{pre}{i}" for i in range(1, 13)]
    if fn.endswith(".zip"):
        import zipfile
        import io

        with zipfile.ZipFile(C + fn) as z:
            d = pd.read_stata(
                io.BytesIO(z.read("Health_Status_and_Functioning.dta")),
                columns=cc,
                convert_categoricals=False,
            )
    else:
        d = pd.read_stata(C + fn, columns=cc, convert_categoricals=False)
    ids = d.ID.astype(str)
    ids = ids.str[:9] + "0" + ids.str[9:] if conv else ids
    sel = d[cc[1:]].fillna(0)
    ans = (sel > 0).any(axis=1)
    net.append(
        pd.DataFrame(
            {
                "id": ids,
                "wave": w,
                "internet": np.where(sel[f"{pre}10"].eq(10), 1, np.where(ans, 0, np.nan)),
            }
        )
    )
p = p.merge(pd.concat(net), on=["id", "wave"], how="left")
p = pd.concat([p, w5], ignore_index=True)
p["cohort"] = "CHARLS"
p["country"] = "China"
p.to_csv(OUT + "panel_CHARLS.csv.gz", index=False)
print(p.shape)
print(
    p[p.iwstat == 1]
    .groupby("wave")[["internet", "dep", "recall", "iadl", "wealth", "cancer"]]
    .agg(["count", "mean"])
    .round(2)
)
