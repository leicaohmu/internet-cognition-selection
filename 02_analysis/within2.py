"""Within-adopter diagnostics requested in round-2 review.
- persistence of use among early starters; sustained-use contrast
- life-event transitions by group in t0-t1 and t2-t3 windows; contrast excluding people with transitions
- within-adopter pre-trend (early vs late on recall/IADL at t0 adjusted for t-1)
- HRS without outcome waves 14-15 (web-mode recall)
- exclusions due to unknown status/death at t3
- missingness of outcome by group (for tipping-point)
"""

import os

os.environ.setdefault("USE_LAG", "1")
os.environ.setdefault("USE_EXTRA", "1")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from build_trials import load
from aipw import aipw
from ladder import aipw_cols
from within import SCHED4
import within

PRECOLS = [
    "age",
    "female",
    "edu",
    "partnered",
    "live_alone",
    "rural",
    "year",
    "dep_lag",
    "srh_poor_lag",
    "iadl_any_lag",
    "adl_any_lag",
    "recall_lag",
    "ncond_lag",
    "work_lag",
    "wealth_q_lag",
]
EV = ["work", "partnered", "ncond", "hosp", "sp_internet", "live_alone"]
OUT = [("Y_recall", lambda d: d), ("Y_iadl", lambda d: d[d.iadl_any == 0]), ("Y_dep", lambda d: d)]


def build(c):
    p = load(c)
    ix = p.set_index(["id", "wave"])
    t = within.build(c)  # early/middle/late/never with outcomes
    t["rawid"] = t.id.str.replace(c + "_", "", n=1)
    # attach event variables at t1, t2, t3 per trial
    parts = []
    for k, (t0, t1, t2, t3) in enumerate(SCHED4[c]):
        s = t[t.trial == k].copy()
        for w, suf in [(t1, "_e1"), (t2, "_e2"), (t3, "_e3")]:
            s = s.join(ix.xs(w, level="wave")[EV].add_suffix(suf), on="rawid")
        s["t2wave"] = t2
        parts.append(s)
    t = pd.concat(parts, ignore_index=True)

    def ev(a, b):  # any transition between two sets of measurements
        retire = (t["work" + a] == 1) & (t["work" + b] == 0)
        widow = (t["partnered" + a] == 1) & (t["partnered" + b] == 0)
        newcond = t["ncond" + b] > t["ncond" + a]
        hosp = t["hosp" + b] == 1
        spon = (t["sp_internet" + a] == 0) & (t["sp_internet" + b] == 1)
        alone = (t["live_alone" + a] == 0) & (t["live_alone" + b] == 1)
        return pd.DataFrame(
            dict(
                retire=retire,
                partner_loss=widow,
                new_condition=newcond,
                hospital=hosp,
                spouse_online=spon,
                alone=alone,
            )
        )

    for col in EV:
        t[col + "_e0"] = t[col]
    e01 = ev("_e0", "_e1")
    e23 = ev("_e2", "_e3")
    for k in e01:
        t["ev01_" + k] = e01[k]
        t["ev23_" + k] = e23[k]
    t["any01"] = e01[["retire", "partner_loss", "new_condition", "hospital"]].any(axis=1)
    t["any23"] = e23[["retire", "partner_loss", "new_condition", "hospital"]].any(axis=1)
    m = within.main[within.main.cohort == c]
    t["Y_pre"] = (t.recall - m.recall.mean()) / m.recall.std()
    t["Y_iadl_pre"] = t.iadl_any.where(t.iadl_any_lag == 0)
    return t


def t3_exclusions(c):
    p = load(c)
    ix = p.set_index(["id", "wave"])
    rows = []
    for t0, t1, t2, t3 in SCHED4[c]:
        b = p[
            (p.wave == t0)
            & (p.iwstat == 1)
            & (p.internet == 0)
            & (p.age >= 50)
            & (p.proxy.fillna(0) != 1)
        ]
        for w, suf, cols in [
            (t1, "_1", ["iwstat", "internet"]),
            (t2, "_2", ["iwstat", "internet"]),
            (t3, "_3", ["iwstat", "internet"]),
        ]:
            b = b.join(ix.xs(w, level="wave")[cols].add_suffix(suf), on="id")
        ok12 = (b.iwstat_1 == 1) & (b.iwstat_2 == 1) & b.internet_1.notna() & b.internet_2.notna()
        g = np.select(
            [b.internet_1 == 1, (b.internet_1 == 0) & (b.internet_2 == 1)],
            ["early", "middle"],
            "offline_t2",
        )
        lost = ok12 & ~((b.iwstat_3 == 1) & b.internet_3.notna())
        died = ok12 & b.iwstat_3.isin([5, 6])
        for gg in ["early", "middle", "offline_t2"]:
            s = ok12 & (g == gg)
            rows.append(
                dict(
                    cohort=c,
                    grp=gg,
                    n_known_t2=int(s.sum()),
                    lost_t3=int((lost & s).sum()),
                    died_t3=int((died & s).sum()),
                )
            )
    return pd.DataFrame(rows).groupby(["cohort", "grp"]).sum().reset_index()


if __name__ == "__main__":
    rows, desc, excl = [], [], []
    for c in SCHED4:
        t = build(c)
        excl.append(t3_exclusions(c))
        el = t[t.grp.isin(["early", "late"])].copy()
        el["A"] = (el.grp == "early").astype(int)
        # descriptives
        e = el[el.grp == "early"]
        d = dict(
            cohort=c,
            n_early=len(e),
            n_late=int((el.grp == "late").sum()),
            early_use_t2=e.internet_2.mean(),
            early_use_t3=e.internet_3.mean(),
            early_ever_before=e.ever_before.mean(),
            late_ever_before=el[el.grp == "late"].ever_before.mean(),
        )
        for g in ["early", "late"]:
            s = el[el.grp == g]
            for k in ["retire", "partner_loss", "new_condition", "hospital", "spouse_online"]:
                d[f"{g}_ev01_{k}"] = s["ev01_" + k].mean()
                d[f"{g}_ev23_{k}"] = s["ev23_" + k].mean()
            d[f"{g}_any01"] = s.any01.mean()
            d[f"{g}_any23"] = s.any23.mean()
            for y in ["Y_recall", "Y_iadl", "Y_dep"]:
                base = s if y != "Y_iadl" else s[s.iadl_any == 0]
                d[f"{g}_miss_{y}"] = base[y].isna().mean()
        desc.append(d)
        variants = {
            "primary": el,
            "sustained": el[(el.grp == "late") | ((el.internet_2 == 1) & (el.internet_3 == 1))],
            "no_transitions": el[~el.any01 & ~el.any23],
            "never_before": el[el.ever_before == 0],
        }
        if c == "HRS":
            variants["hrs_no_web_waves"] = el[el.t2wave < 14]
        for name, dd in variants.items():
            for y, f in OUT:
                x = f(dd)
                if x.A.sum() < 30 or (1 - x.A).sum() < 30 or x[y].notna().sum() < 150:
                    continue
                r = aipw(x, y, trim=0.025)
                r.update(variant=name)
                rows.append(r)
                print(
                    c,
                    name,
                    y,
                    f"{r['est']:+.4f} ({r['lo']:+.4f},{r['hi']:+.4f}) n={r['n']} A1={r['n_A1']}",
                    flush=True,
                )
        # within-adopter pre-trend
        for y in ["Y_pre", "Y_iadl_pre"]:
            x = el[el.recall_lag.notna()] if y == "Y_pre" else el[el.iadl_any_lag == 0]
            if x[y].notna().sum() < 150 or x.A.sum() < 30 or (1 - x.A).sum() < 30:
                continue
            r = aipw_cols(x, y, PRECOLS)
            r.update(variant="pretrend_within")
            rows.append(r)
            print(
                c,
                "pretrend_within",
                y,
                f"{r['est']:+.4f} ({r['lo']:+.4f},{r['hi']:+.4f}) n={r['n']}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv("results_within2.csv", index=False)
    pd.DataFrame(desc).to_csv("within2_descriptives.csv", index=False)
    pd.concat(excl).to_csv("within2_t3_exclusions.csv", index=False)
    print(pd.DataFrame(desc).round(3).T.to_string())
    print(pd.concat(excl).to_string())
