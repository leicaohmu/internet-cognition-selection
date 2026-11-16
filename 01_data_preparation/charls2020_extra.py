"""Extract 2020 CHARLS items not in the harmonized file (self-rated health, in-patient stay, diagnoses, vigorous activity, serial 7s, life satisfaction).
Run in 数据/CHARLS2020r; output used by build_charls_w6.py."""

import pandas as pd
import os

with pd.io.stata.StataReader("Health_Status_and_Functioning.dta") as r:
    lab = r.variable_labels()
ser = [k for k in lab if k.startswith("dc007")]
cols = (
    ["ID", "da001", "da007"]
    + [f"da003_{i}_" for i in range(1, 16)]
    + ["da032_1_", "da033_1_", "dc026"]
    + ser
)
d = pd.read_stata("Health_Status_and_Functioning.dta", columns=cols, convert_categoricals=False)
d.to_csv(os.path.expanduser("~/mnt/小王新/analysis/derived/charls2020_extra.csv"), index=False)
