"""Negative-control outcomes that internet use cannot plausibly change within 2-6 years: grip strength (z), any hospital stay since the last
interview (incident: among those without a stay reported at t0), new cancer diagnosis (incident). Same four designs and estimators as newout.py.
"""

import os
import sys

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
os.environ["OUTS"] = "grip,hosp,cancer"
import numpy as np
import pandas as pd
import newout as N

N.BIN.update({"hosp", "cancer"})
_panel = N.panel


def panel(c):
    p = _panel(c)
    base = p[(p.internet == 0) & (p.age >= 50) & p.wave.isin({s[0] for s in N.SCHED[c]})]
    mu, sd = base.grip.mean(), base.grip.std()
    p["z_grip"] = (p.grip - mu) / sd if sd and sd > 0 else np.nan
    return p


_oc = N.outcome_cols
N.outcome_cols = lambda x: {"grip": "z_grip", "hosp": "hosp", "cancer": "cancer"}.get(x) or _oc(x)
if __name__ == "__main__":
    for c in sys.argv[1:]:
        p = panel(c)
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
        pd.DataFrame(rows).to_csv(f"results_negctrl_{c}.csv", index=False)
        print("saved", c, flush=True)
