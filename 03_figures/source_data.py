"""Build Source Data workbook: one sheet per figure panel, values read from the plotted matplotlib artists
(so numbers match the figures exactly), plus Fig. 1 from its input CSVs."""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, PolyCollection, LineCollection
from matplotlib.patches import Rectangle
import nh3fig as N

OUT = sys.argv[1] if len(sys.argv) > 1 else "Source_Data.xlsx"
VER = sys.argv[2] if len(sys.argv) > 2 else ""

CMAP = {N.COL[c].lower(): f"{c}" for c in N.COL}
CMAP.update(
    {
        N.OBS: "Pooled / observed",
        N.OBS2: "Observed (2nd series)",
        N.OBS3: "Observed (3rd series)",
        N.EFF: "Prediction if going online improved outcomes",
        N.SELC: "Prediction if only selection / death benchmark",
        N.GREY: "Reference / comparison group",
        N.INK: "Black",
        N.INK2: "Dark grey",
        "#b9b7b0": "Light grey (harm direction)",
        "#d6d4cd": "Light grey (off)",
        "#ffffff": "White",
    }
)


def cname(c):
    try:
        h = to_hex(c).lower()
    except Exception:
        return ""
    return CMAP.get(h, h)


def isnum(s):
    return re.fullmatch(r"[−\-+]?\d*\.?\d+%?", s.strip().replace(",", "")) is not None


def tickmap(ax, axis):
    tk = ax.get_xticks() if axis == "x" else ax.get_yticks()
    lb = [t.get_text() for t in (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels())]
    m = [(p, l.replace("\n", " ")) for p, l in zip(tk, lb) if l and not isnum(l)]
    return m


def nearest(v, m, tol=0.5):
    if not m or v is None or not np.isfinite(v):
        return ""
    p, l = min(m, key=lambda t: abs(t[0] - v))
    return l if abs(p - v) <= tol else ""


def r(v):
    return None if v is None or not np.isfinite(v) else float(np.round(v, 5))


def extract(ax, title=""):
    rows = []
    xm, ym = tickmap(ax, "x"), tickmap(ax, "y")
    YT, XT = ax.get_yaxis_transform(), ax.get_xaxis_transform()
    if not ym:
        ym = [
            (t.get_position()[1], t.get_text().replace("\n", " "))
            for t in ax.texts
            if t.get_transform() in (YT,) and t.get_text().strip() and not isnum(t.get_text())
        ]
    if not xm:
        xm = [
            (t.get_position()[0], t.get_text().replace("\n", " "))
            for t in ax.texts
            if t.get_transform() in (XT,) and t.get_text().strip() and not isnum(t.get_text())
        ]
    (X0, X1), (Y0, Y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    xl = ax.get_xscale() == "log"
    data = ax.transData
    for ln in ax.lines:
        if ln.get_transform() is not data:
            continue  # reference lines in axes coordinates
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if len(x) == 0:
            continue
        col = cname(ln.get_color())
        mk = ln.get_marker() if ln.get_marker() not in ("None", None, "") else ""
        if mk == "" and ln.get_markerfacecolor() is not None and str(ln.get_mfc()) not in ("none",):
            pass
        if len(x) == 2 and mk == "" and y[0] == y[1] and x[0] != x[1]:
            rows.append(
                dict(
                    element="interval (horizontal)",
                    series=col,
                    y=r(y[0]),
                    x_lo=r(min(x)),
                    x_hi=r(max(x)),
                    y_category=nearest(y[0], ym),
                )
            )
        elif len(x) == 2 and mk == "" and x[0] == x[1] and y[0] != y[1]:
            rows.append(
                dict(
                    element="interval (vertical)",
                    series=col,
                    x=r(x[0]),
                    y_lo=r(min(y)),
                    y_hi=r(max(y)),
                    x_category=nearest(x[0], xm),
                )
            )
        else:
            kind = (
                "point"
                if ln.get_linestyle() in ("None", "", " ")
                else ("line" if mk == "" else "line+point")
            )
            if (
                mk
                and str(ln.get_mfc()).lower() in ("#ffffff", "white", "w")
                or (mk and cname(ln.get_mfc()) == "White")
            ):
                col = cname(ln.get_mec()) + " (open marker)"
            for xi, yi in zip(x, y):
                rows.append(
                    dict(
                        element=kind,
                        series=col,
                        marker=mk,
                        x=r(xi),
                        y=r(yi),
                        x_category=nearest(xi, xm),
                        y_category=nearest(yi, ym),
                    )
                )
    for co in ax.collections:
        if isinstance(co, PathCollection) and len(co.get_offsets()):
            fc = co.get_facecolors()
            ec = co.get_edgecolors()
            sz = co.get_sizes()
            for k, (xi, yi) in enumerate(np.asarray(co.get_offsets())):
                f = cname(fc[k % len(fc)]) if len(fc) else "none"
                e = cname(ec[k % len(ec)]) if len(ec) else ""
                rows.append(
                    dict(
                        element="scatter",
                        series=e if f == "White" else f,
                        filled="no" if f in ("White", "none") else "yes",
                        size=r(sz[k % len(sz)]) if len(sz) else None,
                        x=r(xi),
                        y=r(yi),
                        x_category=nearest(xi, xm),
                        y_category=nearest(yi, ym),
                    )
                )
        elif isinstance(co, PolyCollection) and co.get_transform() is data:
            for p in co.get_paths():
                v = p.vertices
                if len(v) < 3:
                    continue
                xs = np.unique(np.round(v[:, 0], 9))
                if len(xs) > 200:
                    xs = xs[np.linspace(0, len(xs) - 1, 200).astype(int)]
                for xi in xs:
                    yy = v[np.isclose(v[:, 0], xi), 1]
                    rows.append(
                        dict(
                            element="band (e.g. 95% CI)",
                            series=cname(co.get_facecolor()[0]) if len(co.get_facecolor()) else "",
                            x=r(xi),
                            y_lo=r(yy.min()),
                            y_hi=r(yy.max()),
                            x_category=nearest(xi, xm),
                        )
                    )
        elif isinstance(co, LineCollection):
            for s in co.get_segments():
                if len(s) == 2:
                    rows.append(
                        dict(
                            element="segment",
                            series=cname(co.get_color()[0]) if len(co.get_color()) else "",
                            x_lo=r(s[0, 0]),
                            y_lo=r(s[0, 1]),
                            x_hi=r(s[1, 0]),
                            y_hi=r(s[1, 1]),
                        )
                    )
    for p in ax.patches:
        if isinstance(p, Rectangle) and p.get_data_transform() is data:
            x0, y0, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
            if w == 0 and h == 0:
                continue
            rows.append(
                dict(
                    element="bar / cell",
                    series=cname(p.get_facecolor()),
                    x=r(x0),
                    y=r(y0),
                    width=r(w),
                    height=r(h),
                    x_category=nearest(x0 + w / 2, xm),
                    y_category=nearest(y0 + h / 2, ym),
                )
            )
    for t in ax.texts:
        s = t.get_text().strip()
        if not s or len(s) == 1 and s.isalpha():
            continue
        if t.get_fontsize() >= 10 or s == title.strip():
            continue
        tr = t.get_transform()
        dx = tr is data or tr is XT
        dy = tr is data or tr is YT
        x, y = t.get_position()
        rows.append(
            dict(
                element="annotation",
                text=s.replace("\n", " "),
                x=r(x) if dx else None,
                y=r(y) if dy else None,
                x_category=nearest(x, xm) if dx else "",
                y_category=nearest(y, ym) if dy else "",
            )
        )
    for d in rows:
        if not str(d.get("element", "")).startswith(("interval", "segment", "band")):
            continue
        cl = [
            k
            for k, lim in [("x_lo", X0), ("x_hi", X1), ("y_lo", Y0), ("y_hi", Y1)]
            if d.get(k) is not None and abs(d[k] - lim) < 1e-6 * max(1, abs(lim))
        ]
        if cl:
            d["note"] = (
                "touches the axis limit ("
                + ", ".join(cl)
                + "): may be truncated in the figure or be a reference line; full value in the analysis CSV"
            )
    return rows


def panels(fig):
    out = {}
    cur = None
    for ax in fig.axes:
        L = [
            t
            for t in ax.texts
            if t.get_fontsize() >= 10 and re.fullmatch(r"[a-z]", t.get_text().strip())
        ]
        tt = None
        if L:
            cur = L[0].get_text().strip()
            ti = [
                t
                for t in ax.texts
                if t is not L[0]
                and abs(t.get_position()[1] - L[0].get_position()[1]) < 1e-6
                and t.get_weight() == "bold"
            ]
            tt = ti[0].get_text() if ti else ""
        if cur is None:
            continue
        d = out.setdefault(cur, dict(title="", rows=[], axes=0))
        if tt is not None and not d["title"]:
            d["title"] = tt
        d["axes"] += 1
        rows = extract(ax, d["title"])
        for rr in rows:
            rr["sub-axis"] = d["axes"]
        d["rows"] += rows
        xlab, ylab = ax.get_xlabel().replace("\n", " "), ax.get_ylabel().replace("\n", " ")
        d.setdefault("labels", []).append((d["axes"], xlab, ylab, ax.get_xscale(), ax.get_yscale()))
    return out


CAP = {}


def _cap(fig, name):
    CAP[name] = fig
    print("captured", name)


N.save = _cap

SRC = {
    "Fig2": "results_ladder (cross-sectional vs target trial), results_meta_hgb_trim0.025*.csv, results_newout_all.csv, results_dementia_HRS.csv, hte_subgroups_pooled.csv, results_sens_overlap.csv",
    "Fig3": "results_traj (trajectories), results_newout_all.csv, placebo / pre-trend results, results_tte_minus_placebo.csv, dist_t0.csv.gz",
    "Fig4": "results_newout_all.csv, results_dementia_HRS.csv, results_dose_lag.csv, flows_t0.csv, results_sustained_traj*.csv, early-vs-late results",
    "Fig5": "results_corr_pooled.csv (selection index), results_es_all/pooled.csv, results_es_stop_all/pooled.csv, pre-trend sensitivity",
    "Fig6": "anatomy results (starting by recall quintile, predictors), results_stop_pred*.csv, results_robust_sel*.csv, gradient / meta-regression",
    "Fig7": "results_negctrl_all/pooled.csv, results_main/meta_hgb_trim0.025_lag_t1_extra.csv (death), results_subgrp_all/pooled.csv, results_newout_all.csv, results_dementia_HRS.csv",
    "Fig8": "simulation_grid4.csv, results_pool_hk.csv, results_tte_minus_placebo.csv, results_es_pooled.csv",
    "EDFig1": "results_ladder (cross-sectional vs target trial), results_subgrp_pooled.csv, hte_subgroups_pooled.csv, results_sens_overlap.csv, results_meta_hgb_trim0.025*.csv",
    "EDFig2": "results_traj (cohort trajectories), sensitivity results (results_sens*), results_es_all/pooled.csv (pre-trend sensitivity)",
    "EDFig3": "flows_t0.csv, early-vs-late results, results_newout_all.csv, results_dementia_HRS.csv, results_sustained_traj*.csv",
    "EDFig4": "results_behav_vig_all.csv, results_vig_pred_pooled.csv, results_es_vig_all/pooled.csv, results_es_stopvig_all/pooled.csv",
}
RUN = [
    ("Fig2", N.fig2, "NH3_Fig2"),
    ("Fig3", N.fig3, "NH3_Fig3"),
    ("Fig4", N.fig4, "NH3_Fig4"),
    ("Fig5", N.fig5, "NH3_Fig5"),
    ("Fig6", N.fig6, "NH2_Fig6"),
    ("Fig7", N.fig7n, "NH3_Fig7"),
    ("Fig8", N.fig8, "NH3_Fig8"),
    ("EDFig1", N.ed1, "NH3_EDFig1"),
    ("EDFig2", N.ed2, "NH3_EDFig2"),
    ("EDFig3", N.ed3, "NH3_EDFig3"),
    ("EDFig4", N.fig7, "NH3_EDFig4"),
]

COLS = [
    "sub-axis",
    "element",
    "series",
    "marker",
    "filled",
    "size",
    "x_category",
    "y_category",
    "x",
    "y",
    "x_lo",
    "x_hi",
    "y_lo",
    "y_hi",
    "width",
    "height",
    "text",
    "note",
]
sheets = []
for lab_, fn, nm in RUN:
    fn()
    fig = CAP[nm]
    for L, d in sorted(panels(fig).items()):
        df = pd.DataFrame(d["rows"]).reindex(columns=COLS).dropna(axis=1, how="all")
        hdr = [
            f'{"Extended Data Fig. " + lab_[5:] if lab_.startswith("ED") else "Fig. " + lab_[3:]}{L}: {d["title"]}',
            "Values are the plotted quantities, read from the figure objects. Estimates are point estimates; intervals are 95% CIs unless the legend states otherwise.",
            "Axis labels: "
            + " | ".join(
                f'[{k}] x = {a or "-"} ({xs}); y = {b or "-"} ({ys})'
                for k, a, b, xs, ys in d["labels"]
            ),
            "Derived from (03_表/分析结果csv): " + SRC[lab_],
        ]
        sheets.append((f"{lab_}{L}", hdr, df))
    plt.close(fig)


# exact tables for grid panels and the selection index
def sig_table(beh):
    pred = {"pla": (0, 1), "pre": (0, 1), "evl": (1, 0)}
    names = {
        "tte": "Target trial (start before outcome)",
        "pla": "Placebo (start after outcome)",
        "pre": "Before anyone starts (difference at t0)",
        "evl": "Early vs late (at t2)",
    }
    out = []
    for rw in N.signature_rows():
        if rw["beh"] != beh:
            continue
        T = rw["tte"]["est"]
        for k in ["tte", "pla", "pre", "evl"]:
            p = rw[k]
            e_, s_ = pred.get(k, (None, None))
            out.append(
                dict(
                    outcome=rw["label"],
                    unit=rw["unit"],
                    design=names[k],
                    estimate=None if p is None else p["est"],
                    ci_lo=None if p is None else p["lo"],
                    ci_hi=None if p is None else p["hi"],
                    k_cohorts=None if p is None else p["k"],
                    predicted_if_effect=None if e_ is None else e_ * T,
                    predicted_if_selection=None if s_ is None else s_ * T,
                )
            )
    return pd.DataFrame(out).round(5)


def selidx_table():
    out = []
    for rw in [x for x in N.signature_rows() if x["beh"] == "Internet"]:
        if not ((rw["tte"]["lo"] > 0) or (rw["tte"]["hi"] < 0)):
            out.append(
                dict(
                    outcome=rw["label"],
                    component="—",
                    median=None,
                    lo95=None,
                    hi95=None,
                    note="target trial CI includes 0; not computed",
                )
            )
            continue
        parts, comb = N._sel_index(rw)
        for k, v in list(parts.items()) + [("combined", comb)]:
            out.append(
                dict(
                    outcome=rw["label"],
                    component={
                        "pla": "placebo",
                        "pre": "before anyone starts",
                        "evl": "early vs late",
                    }.get(k, k),
                    median=v[0],
                    lo95=v[1],
                    hi95=v[2],
                )
            )
    return pd.DataFrame(out).round(5)


HA = "Pooled (Hartung-Knapp) estimates with 95% CIs; predicted columns are the values implied by each hypothesis, in units of the target-trial estimate."
ins = [
    (
        "Fig5a",
        [
            "Fig. 5a: Predicted versus observed",
            HA,
            "SD for cognitive outcomes; percentage points for binary outcomes.",
            "Derived from results_newout_all.csv, results_dementia_HRS.csv and core target-trial / placebo / pre-trend / early-vs-late results",
        ],
        sig_table("Internet"),
    ),
    (
        "EDFig4a",
        [
            "Extended Data Fig. 4a: Predicted versus observed, starting weekly vigorous activity",
            HA,
            "",
            "Derived from " + SRC["EDFig4"],
        ],
        sig_table("Physical activity"),
    ),
]
for nm, hdr, df in ins:
    pos = next(i for i, t in enumerate(sheets) if t[0].startswith(nm[:-1]))
    sheets.insert(pos, (nm, hdr, df))
i5b = next(i for i, t in enumerate(sheets) if t[0] == "Fig5b")
sheets.insert(
    i5b + 1,
    (
        "Fig5b_exact",
        [
            "Fig. 5b (exact): Selection index, untruncated",
            "Median and 2.5th/97.5th percentiles of 20,000 multivariate-normal draws (seed 1); 0 = effect prediction, 1 = selection prediction.",
            "",
            "Derived from results_corr_pooled.csv and Fig5a estimates",
        ],
        selidx_table(),
    ),
)

# Fig. 1: data panels from input CSVs
pr = pd.read_csv("prevalence.csv")
pr = pr[~((pr.cohort == "SHARE") & (pr.wave == 7))]
ini = pd.read_csv("initiation.csv")
sheets.insert(
    0,
    (
        "Fig1c",
        [
            "Fig. 1c: Non-users online by the next wave",
            "Share of non-users at one wave who report internet use at the next wave.",
            "",
            "Derived from initiation.csv",
        ],
        ini,
    ),
)
sheets.insert(
    0,
    (
        "Fig1b",
        [
            "Fig. 1b: Older adults online, by survey wave",
            "prev = weighted share using the internet; n = respondents at that wave.",
            "",
            "Derived from prevalence.csv",
        ],
        pr,
    ),
)

with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    idx = pd.DataFrame(
        [(s, h[0].split(": ", 1)[1] if ": " in h[0] else "", len(df)) for s, h, df in sheets],
        columns=["Sheet", "Panel title", "Rows"],
    )
    readme = pd.DataFrame(
        {
            "README": [
                f"Source Data {VER} — Internet use and cognition in later life",
                "One sheet per figure panel (FigXy). Extended Data Fig. n sheets are EDFigny.",
                "Columns: series = colour meaning (cohort name, pooled/observed, reference group, or prediction); element = point, line, interval (95% CI), band, bar/cell, scatter or annotation;",
                "x_category / y_category = the axis category the value belongs to (e.g. outcome, subgroup or cohort row); sub-axis numbers stacked axes within one panel.",
                "Cohorts: HRS (USA), ELSA (England), SHARE (Europe), CHARLS (China), LASI (India, cross-sectional only). Pooled estimates: Hartung-Knapp random effects.",
                "Full analysis outputs with standard errors and sample sizes are provided as CSV files alongside this workbook.",
                "",
            ]
        }
    )
    readme.to_excel(xw, "README", index=False)
    idx.to_excel(xw, "README", index=False, startrow=len(readme) + 2)
    for s, hdr, df in sheets:
        pd.DataFrame({hdr[0]: hdr[1:]}).to_excel(xw, s[:31], index=False)
        df.to_excel(xw, s[:31], index=False, startrow=5)
print(len(sheets), "sheets ->", OUT)
