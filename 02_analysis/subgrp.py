"""Subgroup target-trial and placebo estimates for word recall from cross-fitted AIPW pseudo-outcomes (same estimator and covariates as newout.py);
selection-corrected contrast = target trial - placebo within each subgroup (independent samples assumed for the SE); pooled by HKSJ.
"""

import os
import sys

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
os.environ["OUTS"] = "recall"
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import newout as N
from aipw import aipw, cluster_var
from hte import subgroups

rows = []
for c in sys.argv[1:]:
    p = N.panel(c)
    t = N.main[N.main.cohort == c].copy()
    d = pd.concat(
        [N.attach(g, p, "recall", t0, t2) for (t0, t1, t2), g in t.groupby(["t0", "t1", "t2"])]
    )
    d["id"] = c + "_" + d.id
    d = d.assign(cohort=c)
    s = N.staggered(c, p)
    q = pd.concat([N.attach(g, p, "recall", t0, t2) for (t0, t2), g in s.groupby(["t0", "t2"])])
    q["id"] = c + "_" + q.id
    q = q[q.grp.isin(["late", "never"])].assign(A=lambda z: (z.grp == "late").astype(int), cohort=c)
    for design, dd in [("tte", d), ("placebo", q)]:
        res, P = aipw(dd, "Y", trim=0.025, return_pseudo=True)
        tt = P["t"].copy()
        phi = P["psi1"] - P["psi0"]
        phi = np.asarray(phi)
        print(c, design, round(res["est"], 3), flush=True)
        for name, mask in subgroups(tt).items():
            m = mask.fillna(False).values
            if m.sum() < 150 or tt.A.values[m].sum() < 25:
                continue
            est = phi[m].mean()
            se = np.sqrt(cluster_var(phi[m] - est, tt.id.values[m]))
            rows.append(
                dict(
                    cohort=c,
                    design=design,
                    subgroup=name,
                    est=est,
                    se=se,
                    n=int(m.sum()),
                    n_A1=int(tt.A.values[m].sum()),
                )
            )
        rows.append(
            dict(
                cohort=c,
                design=design,
                subgroup="Overall",
                est=phi.mean(),
                se=np.sqrt(cluster_var(phi - phi.mean(), tt.id.values)),
                n=len(tt),
                n_A1=int(tt.A.sum()),
            )
        )
    pd.DataFrame(rows).to_csv(f"results_subgrp_{c}.csv", index=False)
    print("saved", c, flush=True)
