"""Target trial restricted to people alive with known internet status at t3, minus lead-1 placebo, with person-clustered covariance."""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import placebo

src = open("diff_tte_placebo.py").read()
exec(src[: src.index("rows = []")])


class _D:
    pass


D = _D()
D.pseudo = pseudo
D.main = main
from build_trials import load
from round3 import YEAR as YMAP

rows = []
for c in ["HRS", "ELSA", "SHARE", "CHARLS"]:
    Y = YMAP[c]
    p = load(c)
    mt = D.main[D.main.cohort == c].copy()
    mt["rawid"] = mt.id.str.replace(c + "_", "", n=1)
    waves = sorted(Y)
    nxt = {a: b for a, b in zip(waves[:-1], waves[1:])}
    mt["t3"] = mt.t2.map(nxt)
    st = p[["id", "wave", "iwstat", "internet"]].rename(
        columns={"id": "rawid", "wave": "t3", "iwstat": "iw3", "internet": "net3"}
    )
    mt = mt.merge(st, on=["rawid", "t3"], how="left")
    common = mt[(mt.iw3 == 1) & mt.net3.notna()]
    pl = placebo.build(c)
    dm, im = D.pseudo(common, "Y_recall")
    dp, ip = D.pseudo(pl, "Y_recall")
    em, ep = dm.mean(), dp.mean()
    a = pd.Series((dm - em) / len(dm)).groupby(im).sum()
    b = pd.Series((dp - ep) / len(dp)).groupby(ip).sum()
    s = a.sub(b, fill_value=0)
    rows.append(
        dict(cohort=c, y="Y_recall", tte=em, placebo=ep, est=em - ep, se=np.sqrt((s**2).sum()))
    )
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results_tte_known_t3_minus_placebo.csv", index=False)
