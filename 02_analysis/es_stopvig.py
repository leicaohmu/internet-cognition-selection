"""Reverse event study for physical activity: outcomes around first report of no weekly vigorous activity among people active at their first observed wave."""

import os
import sys

os.environ["BEH"] = "stopvig"
os.environ.setdefault("ES_OUTS", "z_recall,z_gcog,srh")
import es

_panel = es.panel


def panel(c):
    p = _panel(c)
    p["stopvig"] = 1 - p["vig_weekly"]
    return p


es.panel = panel
if __name__ == "__main__":
    for c in sys.argv[1:]:
        es.run(c)
