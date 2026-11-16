import pandas as pd
import numpy as np
from aipw import meta_dl

src = [
    ("Cross-sectional association", "results_ladder_cross.csv"),
    ("Target trial emulation", "results_main_hgb_trim0.025_lag_extra.csv"),
    ("Placebo: initiation 1 wave after outcome", "results_placebo.csv"),
    ("Placebo: initiation 2 waves after outcome", "results_placebo_lead2.csv"),
]
rows = []
for rung, f in src:
    r = pd.read_csv(f)
    for y in ["Y_recall", "Y_iadl", "Y_dep"]:
        g = r[r.y == y]
        for _, x in g.iterrows():
            rows.append(
                dict(
                    rung=rung,
                    y=y,
                    cohort=x.cohort,
                    est=x.est,
                    lo=x.lo,
                    hi=x.hi,
                    rr=x.get("rr", np.nan),
                    rr_lo=x.get("rr_lo", np.nan),
                    rr_hi=x.get("rr_hi", np.nan),
                    n=x.n,
                    n_A1=x.n_A1,
                )
            )
        if y == "Y_recall":
            m = meta_dl(g.est, g.se)
            rows.append(
                dict(
                    rung=rung,
                    y=y,
                    cohort="Pooled",
                    est=m["est"],
                    lo=m["lo"],
                    hi=m["hi"],
                    I2=m["I2"],
                    k=m["k"],
                )
            )
        else:
            m = meta_dl(np.log(g.rr), g.se_lrr)
            rows.append(
                dict(
                    rung=rung,
                    y=y,
                    cohort="Pooled",
                    rr=np.exp(m["est"]),
                    rr_lo=np.exp(m["lo"]),
                    rr_hi=np.exp(m["hi"]),
                    I2=m["I2"],
                    k=m["k"],
                )
            )
d = pd.DataFrame(rows)
d.to_csv("ladder_all.csv", index=False)
p = d[d.cohort == "Pooled"]
for _, x in p.iterrows():
    v = (
        f"{x.est:+.3f} ({x.lo:+.3f} to {x.hi:+.3f})"
        if x.y == "Y_recall"
        else f"{x.rr:.2f} ({x.rr_lo:.2f}-{x.rr_hi:.2f})"
    )
    print(f"{x.rung:45s} {x.y:9s} {v}  I2={x.I2:.2f} k={int(x.k)}")
