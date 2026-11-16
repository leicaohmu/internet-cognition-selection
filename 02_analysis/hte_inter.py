import pandas as pd
import numpy as np
import sys
from pool_hk import hksj
from scipy import stats

f = sys.argv[1]
S = pd.read_csv(f)
O = pd.read_csv("_pre_w6/hte_interactions.csv")
rows = []
for _, o in O.iterrows():
    a, b = o.contrast.split(" vs ")
    A = S[(S.y == o.y) & (S.subgroup == a)].set_index("cohort")
    B = S[(S.y == o.y) & (S.subgroup == b)].set_index("cohort")
    j = A.index.intersection(B.index)
    d = A.loc[j, "est"] - B.loc[j, "est"]
    s = np.sqrt(A.loc[j, "se"] ** 2 + B.loc[j, "se"] ** 2)
    h = hksj(d.values, s.values)
    k = h["k"]
    if k >= 3:
        lo, hi = h["hk_lo"], h["hk_hi"]
        p = 2 * stats.t.sf(abs(h["est"] / h["hk_se"]), k - 1)
    else:
        lo, hi = h["dl_lo"], h["dl_hi"]
        se = (hi - lo) / 3.92
        p = 2 * stats.norm.sf(abs(h["est"] / se))
    rows.append(dict(y=o.y, contrast=o.contrast, diff=h["est"], lo=lo, hi=hi, p=p, k=k))
R = pd.DataFrame(rows)
ps = R.p.sort_values()
n = len(ps)
holm = pd.Series(
    np.minimum(1, np.maximum.accumulate((n - np.arange(n)) * ps.values)), index=ps.index
)
R["p_holm"] = holm
print(R[["y", "contrast", "diff", "p", "p_holm"]].round(3).to_string())
print("min holm", R.p_holm.min().round(3))
if len(sys.argv) > 2:
    R.to_csv(sys.argv[2], index=False)
