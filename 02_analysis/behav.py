"""v1.0: apply the same four designs to a second self-selected later-life behaviour: starting weekly vigorous physical
activity (all four cohorts). Exposure X replaces internet use in eligibility (X = 0 at t0), exposure (X at t1), placebo
(X first at t3) and early-vs-late grouping. Outcomes: recall, two-domain composite, fair/poor self-rated health.
Covariates as in the primary analysis plus internet use at t0; ever-X before t0 replaces ever-internet.
"""

import os
import sys

os.environ["USE_LAG"] = "1"
os.environ["USE_EXTRA"] = "1"
os.environ.setdefault("OUTS", "recall,gcog,srh")
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import newout
from build_trials import SCHED, COV

BEH = os.environ.get("BEH", "vig_weekly")
if "net0" not in COV:
    COV.append("net0")


def behaviour_panel(c):
    p = newout.panel(c)
    p["net0"] = p["internet"]
    p["internet"] = p[BEH].where(p[BEH].notna())
    p = p.sort_values(["id", "wave"])
    p["ever_before"] = p.groupby("id").internet.transform(lambda s: s.shift().fillna(0).cummax())
    return p


def trials(c, p):
    ix = p.set_index(["id", "wave"])
    out = []
    for k, (t0, t1, t2) in enumerate(SCHED[c]):
        b = p[
            (p.wave == t0)
            & (p.iwstat == 1)
            & (p.internet == 0)
            & (p.age >= 50)
            & (p.proxy.fillna(0) != 1)
        ].copy()
        s1 = ix.xs(t1, level="wave")[["iwstat", "internet"]].add_suffix("_1")
        b = b.join(s1, on="id")
        b = b[(b.iwstat_1 == 1) & b.internet_1.notna()]
        b["A"] = b.internet_1
        b["trial"] = k
        b["t0"], b["t1"], b["t2"] = t0, t1, t2
        out.append(b)
    t = pd.concat(out, ignore_index=True)
    t["cohort"] = c
    return t


if __name__ == "__main__":
    for c in sys.argv[1:] or ["HRS", "ELSA", "SHARE", "CHARLS"]:
        p = behaviour_panel(c)
        t = trials(c, p)
        print(c, "trials", len(t), "starters", int(t.A.sum()), f"{t.A.mean():.2f}", flush=True)
        newout.main = pd.concat([newout.main[newout.main.cohort != c], t], ignore_index=True)
        rows = newout.tte(c, p) + newout.falsif(c, p) + newout.pretrend(c, p)
        pd.DataFrame(rows).assign(behaviour=BEH).to_csv(f"results_behav_{BEH}_{c}.csv", index=False)
        print("saved", c, flush=True)
