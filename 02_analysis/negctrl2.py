"""Further negative-control outcomes: a new diagnosed chronic condition (count higher at t2 than at t0) and new difficulty with basic
activities of daily living (ADL; incident among those without at t0). Target trial and placebo designs as in negctrl.py.
"""

import os
import sys

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
os.environ["OUTS"] = "ncnew,adl"
import numpy as np
import pandas as pd
import newout as N

N.BIN.update({"ncnew", "adl"})
_oc = N.outcome_cols
N.outcome_cols = lambda x: {"ncnew": "ncond", "adl": "adl_any"}.get(x) or _oc(x)
_attach = N.attach


def attach(b, p, x, w0, wy, alive_col=None):
    if x != "ncnew":
        return _attach(b, p, x, w0, wy)
    ix = p.set_index(["id", "wave"])["ncond"]
    get = lambda w: (
        b.id.map(ix.xs(w, level="wave")) if w in set(p.wave) else pd.Series(np.nan, index=b.index)
    )
    b = b.copy()
    y2, y0 = get(wy).values, get(w0).values
    b["Y"] = (
        np.where(np.isnan(y2) | np.isnan(y0), np.nan, (y2 > y0).astype(float))
        if wy != w0
        else np.nan
    )
    b["base_y"] = 0.0
    b["base_y_lag"] = np.nan
    return b


N.attach = attach
if __name__ == "__main__":
    for c in sys.argv[1:]:
        p = N.panel(c)
        rows = N.tte(c, p)
        st = N.staggered(c, p)
        for x in N.OUTS:
            d = pd.concat([N.attach(g, p, x, t0, t2) for (t0, t2), g in st.groupby(["t0", "t2"])])
            d["id"] = c + "_" + d.id
            pl = d[d.grp.isin(["late", "never"])].assign(A=lambda z: (z.grp == "late").astype(int))
            r = N.run(pl, c, "placebo", x, lambda d: N.aipw(d, "Y", trim=0.025))
            if r:
                rows.append(r)
                print(
                    c, "placebo", x, f"{r['est']:+.3f} ({r['lo']:+.3f},{r['hi']:+.3f})", flush=True
                )
        pd.DataFrame(rows).to_csv(f"results_negctrl2_{c}.csv", index=False)
        print("saved", c, flush=True)
