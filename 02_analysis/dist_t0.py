"""Individual-level recall at t0 among target-trial eligibles, by later initiation (for Fig. 3i).
Recall is residualized on age, age^2, sex and trial within cohort and scaled by the cohort SD of recall at baseline.
"""

import pandas as pd
import numpy as np

t = pd.read_csv(
    "trials.csv.gz",
    usecols=["cohort", "id", "t0", "A", "age", "female", "recall", "trial"],
    low_memory=False,
)
t = t.dropna(subset=["A", "recall", "age", "female"])
out = []
for c, g in t.groupby("cohort"):
    X = np.column_stack(
        [np.ones(len(g)), g.age, g.age**2, g.female]
        + [(g.trial == k).astype(float) for k in sorted(g.trial.unique())[1:]]
    )
    beta, *_ = np.linalg.lstsq(X, g.recall.values, rcond=None)
    res = g.recall.values - X @ beta
    z = res / g.recall.std()
    out.append(pd.DataFrame({"cohort": c, "A": g.A.astype(int).values, "z": z}))
d = pd.concat(out)
d.to_csv("dist_t0.csv.gz", index=False)
print(d.groupby(["cohort", "A"]).z.agg(["count", "mean", "std"]).round(3))
