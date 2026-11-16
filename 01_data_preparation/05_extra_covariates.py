# Extra baseline confounders: hospitalisation, BMI, grip, child contact, loneliness, social activity, spouse id / spouse internet
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
D = os.path.expanduser("~/mnt/小王新/数据/")
OUT = os.path.expanduser("~/mnt/小王新/analysis/derived/")


def rd(f, cols, idc):
    with pd.io.stata.StataReader(f) as r:
        have = set(r.variable_labels())
    cols = [idc] + [c for c in cols if c in have and c != idc]
    out = []
    with pd.io.stata.StataReader(f, convert_categoricals=False, chunksize=5000, columns=cols) as r:
        for ch in r:
            out.append(ch)
    return pd.concat(out, ignore_index=True)


def longify(df, idc, W, spec):
    rows = []
    for w in W:
        d = pd.DataFrame({"id": df[idc], "wave": w})
        for new, pat in spec.items():
            c = pat.format(w=w)
            d[new] = df[c] if c in df else np.nan
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


# HRS
H = D + "3. HRS  美国/HRS_美国/Raw_data/"
W = range(6, 16)
a = rd(
    H + "RAND HRS Longitudinal File 2020/randhrs1992_2020v2.dta",
    [f"r{w}{s}" for w in W for s in ["hosp", "bmi"]],
    "hhidpn",
)
b = rd(
    H + "Gateway Harmonized HRS/H_HRS_d.dta",
    [
        f"{p}{w}{s}"
        for w in W
        for p, s in [("s", "email"), ("r", "kcnt"), ("r", "gripsum"), ("r", "lnlys3")]
    ],
    "hhidpn",
)
x = a.merge(b, on="hhidpn", how="outer")
x["hhidpn"] = x.hhidpn.astype("int64").astype(str)
e = longify(
    x,
    "hhidpn",
    W,
    {
        "hosp": "r{w}hosp",
        "bmi": "r{w}bmi",
        "sp_internet": "s{w}email",
        "kcnt": "r{w}kcnt",
        "grip": "r{w}gripsum",
        "lonely": "r{w}lnlys3",
    },
)
e.to_csv(OUT + "extra_HRS.csv.gz", index=False)
print("HRS", e.drop(columns=["id", "wave"]).notna().mean().round(2).to_dict())
# ELSA
W = range(2, 10)
x = rd(
    D + "2. ELSA 英国/Raw_data/Harmonized ELSA/h_elsa_g3.dta",
    [
        f"{p}{w}{s}"
        for w in W
        for p, s in [
            ("s", "idauniq"),
            ("r", "kcnt"),
            ("r", "gripsum"),
            ("r", "mbmi"),
            ("r", "lnlys3"),
        ]
    ],
    "idauniq",
)
x["idauniq"] = x.idauniq.astype("int64").astype(str)
e = longify(
    x,
    "idauniq",
    W,
    {
        "sp_id": "s{w}idauniq",
        "kcnt": "r{w}kcnt",
        "grip": "r{w}gripsum",
        "bmi": "r{w}mbmi",
        "lonely": "r{w}lnlys3",
    },
)
e["sp_id"] = e.sp_id.where(e.sp_id > 0).astype("Int64").astype(str).replace("<NA>", np.nan)
e.to_csv(OUT + "extra_ELSA.csv.gz", index=False)
print("ELSA", e.drop(columns=["id", "wave"]).notna().mean().round(2).to_dict())
# CHARLS
W = range(1, 5)
x = rd(
    D + "H_CHARLS_D_Data.dta",
    [
        f"{p}{w}{s}"
        for w in W
        for p, s in [
            ("s", "id"),
            ("r", "hosp1y"),
            ("r", "gripsum"),
            ("r", "mbmi"),
            ("r", "socwk"),
            ("h", "kcnt"),
        ]
    ],
    "ID",
)
e = longify(
    x,
    "ID",
    W,
    {
        "sp_id": "s{w}id",
        "hosp": "r{w}hosp1y",
        "grip": "r{w}gripsum",
        "bmi": "r{w}mbmi",
        "socwk": "r{w}socwk",
        "kcnt": "h{w}kcnt",
    },
)
e["sp_id"] = e.sp_id.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, ".": np.nan})
e.to_csv(OUT + "extra_CHARLS.csv.gz", index=False)
print("CHARLS", e.drop(columns=["id", "wave"]).notna().mean().round(2).to_dict())
# SHARE
cols = ["mergeid", "wave", "coupleid", "hc012_", "bmi", "maxgrip", "ch007_km", "sp002_mod"]
out = []
with pd.io.stata.StataReader(
    D + "7.SHARE 欧洲/SHARE_欧洲/Raw_data/easySHARE Release 9.0.0/easySHARE_rel9-0-0.dta",
    convert_categoricals=False,
    chunksize=20000,
    columns=cols,
) as r:
    for ch in r:
        out.append(ch[ch.wave >= 4])
s = pd.concat(out, ignore_index=True)
for c in ["hc012_", "bmi", "maxgrip", "ch007_km", "sp002_mod"]:
    s[c] = pd.to_numeric(s[c], errors="coerce").mask(lambda v: v < 0)
e = pd.DataFrame(
    {
        "id": s.mergeid,
        "wave": s.wave,
        "couple": s.coupleid.astype(str),
        "hosp": (s.hc012_ == 1).astype(float).where(s.hc012_.notna()),
        "bmi": s.bmi,
        "grip": s.maxgrip,
        "kcnt": s.ch007_km,
        "help_recv": s.sp002_mod,
    }
)
e["couple"] = e.couple.where(~e.couple.isin(["", "nan", "None"]))
e.to_csv(OUT + "extra_SHARE.csv.gz", index=False)
print("SHARE", e.drop(columns=["id", "wave"]).notna().mean().round(2).to_dict())
