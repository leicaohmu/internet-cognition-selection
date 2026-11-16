import os

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
os.environ["USE_T1"] = "1"
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from aipw import aipw, meta_dl

t = pd.read_csv("trials.csv.gz", dtype={"id": str, "country": str}, low_memory=False)
t["id"] = t.cohort + "_" + t.id
rows = []
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    for y, f in [
        ("Y_recall", lambda d: d),
        ("Y_iadl", lambda d: d[d.iadl_any == 0]),
        ("Y_dep", lambda d: d),
    ]:
        r = aipw(f(t[t.cohort == c]), y, trim=0.025)
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv("results_main_t1recall.csv", index=False)
for y, g in res.groupby("y"):
    m = meta_dl(g.est, g.se)
    print("MAIN+t1", y, "%+.4f (%+.4f,%+.4f) I2=%.2f" % (m["est"], m["lo"], m["hi"], m["I2"]))
