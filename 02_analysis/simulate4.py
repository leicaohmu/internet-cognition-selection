"""Round-4 simulation: as round 3 with a finer grid of true effects (0 to 0.20 SD by 0.025) for the compatibility curve (Fig. 8c).
Round-3 simulation: wider effect grid (0-0.20 SD), five waves (t-1..t3) so estimators adjust for recall at t0 and t-1
as in the analysis, and four fitting targets (target trial, placebo, pre-trend, within-adopter pre-trend). Two replicates.
"""

import numpy as np
import itertools
import pandas as pd

N = 150_000


def sim(beta, g, phi, rel, a, variant, seed):
    rng = np.random.default_rng(seed)
    su = 0.6
    U = rng.normal(0, np.sqrt(su), N)
    V = rng.normal(0, np.sqrt(1 - su), N)
    on = np.zeros(N, bool)
    start = np.full(N, 9)
    won = np.zeros(N)
    sd_e = np.sqrt((1 - rel) / rel)
    Y = []
    for t in range(5):  # 0=t-1, 1=t0, 2=t1, 3=t2, 4=t3
        if t > 0:
            V = phi * V + rng.normal(0, np.sqrt((1 - su) * (1 - phi**2)), N)
        trig = np.zeros(N)
        if variant == "trigger" and t > 0:
            ev = rng.random(N) < 0.10
            V = V - 0.3 * ev
            trig = ev * 1.0
        c = U + V + beta * (won if variant == "cumulative" else on)
        Y.append(c + rng.normal(0, sd_e, N))
        if t >= 2:
            p = 1 / (1 + np.exp(-(a + g * c + trig)))
            new = (start == 9) & (rng.random(N) < p)
            start[new] = t - 1  # start coded 1=t1,2=t2,3=t3
            on = (on & ~(rng.random(N) < 0.3)) | new if variant == "nonpersist" else on | new
        won = won + on
    Y = np.array(Y)
    Y = (Y - Y[1].mean()) / Y[1].std()

    def ols(mask, A, yi, adj):
        X = np.column_stack([np.ones(mask.sum()), A[mask]] + [Y[j][mask] for j in adj])
        return np.linalg.lstsq(X, Y[yi][mask], rcond=None)[0][1]

    s1 = (start == 1).astype(float)
    return dict(
        beta=beta,
        g=g,
        phi=phi,
        rel=rel,
        a=a,
        variant=variant,
        seed=seed,
        TTE=ols(np.ones(N, bool), s1, 3, [1, 0]),
        placebo=ols(start >= 3, (start == 3).astype(float), 3, [1, 0]),
        pretrend=ols(np.ones(N, bool), s1, 1, [0]),
        EL=ols(np.isin(start, [1, 3]), s1, 3, [1, 0]),
        EL_pre=ols(np.isin(start, [1, 3]), s1, 1, [0]),
    )


rows = []
for g, phi, rel, a in itertools.product(
    [0.25, 0.5, 1.0], [0.6, 0.8, 0.9], [0.6, 0.7, 0.8], [-3.4, -2.3]
):
    for beta in np.round(np.arange(0, 0.2001, 0.025), 3):
        rows.append(sim(beta, g, phi, rel, a, "base", 1))
for variant, g, phi in itertools.product(
    ["trigger", "nonpersist", "cumulative"], [0.25, 0.5, 1.0], [0.6, 0.8, 0.9]
):
    for beta in np.round(np.arange(0, 0.2001, 0.025), 3):
        rows.append(sim(beta, g, phi, 0.7, -2.3, variant, 1))
d = pd.DataFrame(rows)
# Monte Carlo replicate for the first 60 scenarios
d["fit2"] = d.TTE.between(0.073, 0.186) & d.placebo.between(0.076, 0.285)
d["fit4"] = d.fit2 & d.pretrend.between(0.083, 0.268) & d.EL_pre.between(-0.125, 0.304)
d.to_csv("simulation_grid4.csv", index=False)
print("scenarios", len(d), "fit2", d.fit2.sum(), "fit4", d.fit4.sum())
for f in ["fit2", "fit4"]:
    print(f, d[d[f]].groupby("beta").EL.agg(["size", "min", "median", "max"]).round(3).to_string())
print(
    "largest beta among fit4 with EL <= 0.082 (HK upper):",
    d[d.fit4 & (d.EL <= 0.082)].beta.max(),
    "| with EL<=0.055 (DL upper):",
    d[d.fit4 & (d.EL <= 0.055)].beta.max(),
)
