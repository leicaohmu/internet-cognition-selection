"""Reverse event study: recall and composite around first reported NON-use among people online at their first observed internet wave.
Comparators: people still online (not yet stopped). Reuses es.py with exposure = 1 - internet."""

import os
import sys

os.environ["BEH"] = "stop"
os.environ.setdefault("ES_OUTS", "z_recall,z_gcog")
import es

_panel = es.panel


def panel(c):
    p = _panel(c)
    p["stop"] = 1 - p["internet"]
    return p


es.panel = panel
if __name__ == "__main__":
    for c in sys.argv[1:]:
        es.run(c)
