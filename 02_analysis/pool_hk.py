"""Re-pool key results with DerSimonian-Laird + Hartung-Knapp-Sidik-Jonkman (HKSJ) CIs and 95% prediction intervals.
HKSJ: q = sum w*(y-mu)^2/(k-1); se_hk = sqrt(q/sum w); CI uses t(k-1); variance floored at DL SE (modified HK).
Prediction interval: mu +/- t(k-2) * sqrt(tau2 + se^2) (only when k>=3).
"""

import numpy as np
import pandas as pd
from scipy import stats
from aipw import meta_dl


def hksj(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float)
    m = meta_dl(est, se)
    k = m["k"]
    w = 1 / (se**2 + m["tau2"])
    q = (w * (est - m["est"]) ** 2).sum() / (k - 1)
    se_hk = max(np.sqrt(q / w.sum()), m["se"])
    tq = stats.t.ppf(0.975, k - 1)
    out = dict(
        k=k,
        est=m["est"],
        dl_lo=m["lo"],
        dl_hi=m["hi"],
        hk_lo=m["est"] - tq * se_hk,
        hk_hi=m["est"] + tq * se_hk,
        hk_se=se_hk,
        tau2=m["tau2"],
        I2=m["I2"],
        hk90_lo=m["est"] - stats.t.ppf(0.95, k - 1) * se_hk,
        hk90_hi=m["est"] + stats.t.ppf(0.95, k - 1) * se_hk,
    )
    if k >= 3:
        tp = stats.t.ppf(0.975, k - 2)
        s = np.sqrt(m["tau2"] + m["se"] ** 2)
        out.update(pi_lo=m["est"] - tp * s, pi_hi=m["est"] + tp * s)
    return out


SOURCES = {
    "cross-sectional": ("results_ladder_cross.csv", None),
    "target trial": ("results_main_hgb_trim0.025_lag_extra.csv", None),
    "placebo lead 1": ("results_placebo.csv", None),
    "early vs late": ("results_within.csv", "early vs late"),
    "pre-trend": ("results_pretrend.csv", None),
}

if __name__ == "__main__":
    rows = []
    for lab, (f, ct) in SOURCES.items():
        d = pd.read_csv(f)
        if ct:
            d = d[d.contrast == ct]
        for y, g in d.groupby("y"):
            if len(g) < 2:
                continue
            r = hksj(g.est, g.se)
            r.update(analysis=lab, y=y, scale="RD/MD")
            rows.append(r)
            if "se_lrr" in g and g.se_lrr.notna().all() and (g.rr > 0).all():
                rl = hksj(np.log(g.rr), g.se_lrr)
                rr = dict(analysis=lab, y=y, scale="RR", k=rl["k"], I2=rl["I2"], tau2=rl["tau2"])
                for a in ["est", "dl_lo", "dl_hi", "hk_lo", "hk_hi", "pi_lo", "pi_hi"]:
                    if a in rl:
                        rr[a] = np.exp(rl[a])
                rows.append(rr)
    res = pd.DataFrame(rows)
    cols = [
        "analysis",
        "y",
        "scale",
        "k",
        "est",
        "dl_lo",
        "dl_hi",
        "hk_lo",
        "hk_hi",
        "hk90_lo",
        "hk90_hi",
        "pi_lo",
        "pi_hi",
        "I2",
    ]
    res = res[[c for c in cols if c in res]]
    res.to_csv("results_pool_hk.csv", index=False)
    pd.set_option("display.width", 250)
    print(
        res[res.y.isin(["Y_recall", "Y_iadl", "Y_dep", "Y_pre", "Y_iadl_pre", "Y_death"])]
        .round(3)
        .to_string(index=False)
    )
