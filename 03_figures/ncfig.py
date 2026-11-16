"""Nature Communications-style multi-panel main figures (v0.8). 180 mm width, Liberation Sans (Arial metrics), 6.5 pt text."""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from pool_hk import hksj

INK, INK2, MUTED, GRID, SURF, SHADE = (
    "#1a1a1a",
    "#4d4d4a",
    "#8a8a85",
    "#e6e5e0",
    "#ffffff",
    "#f4f3ef",
)
COL = {
    "HRS": "#2a78d6",
    "ELSA": "#eb6834",
    "SHARE": "#1baf7a",
    "CHARLS": "#eda100",
    "LASI": "#e87ba4",
}
MK = {"HRS": "o", "ELSA": "s", "SHARE": "^", "CHARLS": "D", "LASI": "v"}
NAMES = {
    "HRS": "HRS (USA)",
    "ELSA": "ELSA (England)",
    "SHARE": "SHARE (Europe)",
    "CHARLS": "CHARLS (China)",
    "LASI": "LASI (India)",
}
COH = ["HRS", "ELSA", "SHARE", "CHARLS"]
GCOL = {
    "early": "#5b2a86",
    "middle": "#6aa2e0",
    "late": "#9b1b30",
    "never": "#8a8a85",
    "start": "#5b2a86",
    "stay": "#8a8a85",
}
W = 7.09  # 180 mm
FS = 1.24  # global text scale (v1.3: larger type)
plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.size": 6.5 * FS,
        "axes.titlesize": 7 * FS,
        "axes.labelsize": 6.5 * FS,
        "xtick.labelsize": 6 * FS,
        "ytick.labelsize": 6.3 * FS,
        "legend.fontsize": 6 * FS,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
    }
)


def lab(ax, s, x=-0.02, y=1.02, title=None):
    ax.text(
        x,
        y,
        s,
        transform=ax.transAxes,
        fontsize=10.5,
        weight="bold",
        va="bottom",
        ha="right",
        color=INK,
    )
    if title:
        ax.text(
            x + 0.02,
            y,
            title,
            transform=ax.transAxes,
            fontsize=FS * 7,
            weight="bold",
            va="bottom",
            ha="left",
            color=INK,
        )


def clean(ax, grid="x"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if grid == "x":
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f"fig/{name}.png", bbox_inches="tight", facecolor=SURF, dpi=400)
    fig.savefig(f"fig/{name}.pdf", bbox_inches="tight", facecolor=SURF)
    print("saved", name)


def se_of(g):
    return g.se.values if "se" in g and g.se.notna().all() else ((g.hi - g.lo) / 3.92).values


def pool(g, scale=1.0):
    r = hksj(g.est.values, se_of(g))
    return {
        k: (v * scale if isinstance(v, float) and k not in ("I2", "tau2", "k") else v)
        for k, v in r.items()
    }


def forest(
    ax,
    rows,
    xlim,
    scale=1.0,
    ref=0.0,
    shade=(),
    band=None,
    fmt="{:.2f}",
    cohorts=COH + ["LASI"],
    vals=True,
    band_rows=None,
):
    """rows: list of (label, df with cohort/est/lo/hi/se). Cohort points offset; pooled diamond (DL) + HKSJ line."""
    n = len(rows)
    offs = np.linspace(-0.3, 0.06, 5)
    L, H = xlim
    if band and band_rows is None:
        ax.axvspan(-band, band, color="#ebe4f2", zorder=-3, lw=0)
    if band and band_rows is not None:
        for i in band_rows:
            ax.add_patch(
                Rectangle((-band, n - i - 0.5), 2 * band, 1, fc="#ebe4f2", ec="none", zorder=-1)
            )
    for i, (label, g) in enumerate(rows):
        yy = n - i
        if i in shade:
            ax.axhspan(yy - 0.5, yy + 0.5, color=SHADE, zorder=-2, lw=0)
        if g is None or len(g) == 0:
            continue
        for j, c in enumerate(cohorts):
            x = g[g.cohort == c]
            if not len(x):
                continue
            x = x.iloc[0]
            e, lo, hi = x.est * scale, x.lo * scale, x.hi * scale
            y0 = yy + offs[j]
            ax.plot(
                [max(lo, L), min(hi, H)],
                [y0] * 2,
                color=COL[c],
                lw=0.9,
                alpha=0.8,
                solid_capstyle="butt",
            )
            if hi > H:
                ax.plot(H, y0, marker=">", color=COL[c], ms=2.6, clip_on=False)
            if lo < L:
                ax.plot(L, y0, marker="<", color=COL[c], ms=2.6, clip_on=False)
            if L <= e <= H:
                ax.plot(e, y0, MK[c], color=COL[c], ms=3, mec=SURF, mew=0.4)
        gc = g[g.cohort.isin(COH + ["LASI"])]
        if len(gc) >= 2:
            p = pool(gc, scale)
            e = p["est"]
            ax.fill(
                [p["dl_lo"], e, p["dl_hi"], e],
                [yy + 0.3, yy + 0.4, yy + 0.3, yy + 0.2],
                color=INK,
                lw=0,
                zorder=4,
            )
            if len(gc) >= 3:
                ax.plot(
                    [max(p["hk_lo"], L), min(p["hk_hi"], H)],
                    [yy + 0.3] * 2,
                    color=INK,
                    lw=0.6,
                    zorder=3,
                )
                if p["hk_hi"] > H:
                    ax.plot(H, yy + 0.3, marker=">", color=INK, ms=2.4, clip_on=False)
                if p["hk_lo"] < L:
                    ax.plot(L, yy + 0.3, marker="<", color=INK, ms=2.4, clip_on=False)
            if vals:
                ax.text(
                    1.01,
                    yy + 0.3,
                    fmt.format(e).replace("-", "−"),
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=FS * 6,
                    weight="bold",
                    color=INK,
                )
    ax.axvline(ref, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=-1)
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(0.45, n + 0.55)
    ax.set_xlim(L, H)
    clean(ax)


def cohort_legend(fig, y=1.0, extra=(), cohorts=COH, x=0.5, ncol=None):
    h = [
        Line2D([], [], marker=MK[c], color=COL[c], ls="-", lw=0.9, ms=3.5, label=NAMES[c])
        for c in cohorts
    ]
    h.append(
        Line2D(
            [],
            [],
            marker="D",
            color=INK,
            ls="-",
            lw=0.6,
            ms=4,
            label="Pooled (diamond: 95% CI, DerSimonian–Laird; line: Hartung–Knapp)",
        )
    )
    h += list(extra)
    fig.legend(
        handles=h,
        loc="lower center",
        ncol=ncol or len(h),
        frameon=False,
        bbox_to_anchor=(x, y),
        fontsize=FS * 6,
        handlelength=1.6,
        columnspacing=1.1,
    )


# ------------------------------------------------------------------ data helpers
TTE = pd.read_csv("results_main_hgb_trim0.025_lag_extra.csv")
PLA = pd.read_csv("results_placebo.csv")
PL2 = pd.read_csv("results_placebo_lead2.csv")
PRE = pd.read_csv("results_pretrend.csv")
WIT = pd.read_csv("results_within.csv")
W2 = pd.read_csv("results_within2.csv")
W3 = pd.read_csv("results_within3.csv")
R3 = pd.read_csv("results_round3.csv")
LAD = pd.read_csv("ladder_all.csv")
SENS = pd.read_csv("results_sens_overlap.csv")
ATT = pd.read_csv("results_attrition.csv")
DIF = pd.read_csv("results_tte_minus_placebo.csv")
TRJ = pd.read_csv("results_trajectories.csv")


def sel(d, **kw):
    for k, v in kw.items():
        d = d[d[k] == v]
    return d


def ci(d):
    d = d.copy()
    if "lo" not in d or d.lo.isna().any():
        d["lo"] = d.est - 1.96 * d.se
        d["hi"] = d.est + 1.96 * d.se
    return d


def rd(d):  # risk difference in percentage points
    return ci(d)


# ================================================================== Figure 1
def fig1():
    fig = plt.figure(figsize=(W, 5.4))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.15, 1],
        hspace=0.32,
        wspace=0.62,
        left=0.07,
        right=0.98,
        top=0.95,
        bottom=0.08,
    )
    # a  design schematic
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.25, 5.05)
    lab(ax, "a", x=0.0, y=0.975, title="Four designs and what each predicts")
    Wv = ["t−1", "t0", "t1", "t2", "t3"]
    X = {w: 3.05 + 0.95 * i for i, w in enumerate(Wv)}
    for w in Wv:
        ax.text(X[w], 4.32, w, ha="center", fontsize=FS * 7, weight="bold", color=INK)
        ax.plot([X[w]] * 2, [0.35, 4.15], color=GRID, lw=0.6, zorder=0)
    ax.text(
        X["t0"], 4.55, "baseline: offline, aged ≥50", ha="center", fontsize=FS * 5.8, color=INK2
    )
    ax.text(X["t2"], 4.55, "outcome wave", ha="center", fontsize=FS * 5.8, color=INK2)
    ax.text(
        8.15,
        4.4,
        "If going online\nimproves memory",
        ha="center",
        fontsize=FS * 6.3,
        weight="bold",
        color="#5b2a86",
        va="center",
    )
    ax.text(
        9.65,
        4.4,
        "If adopters are\nonly selected",
        ha="center",
        fontsize=FS * 6.3,
        weight="bold",
        color="#9b1b30",
        va="center",
    )
    rows = [
        (
            "Target trial",
            "start at t1  vs  stay offline",
            [("t1", "s")],
            "t2",
            "difference",
            "difference",
        ),
        (
            "Placebo trial",
            "start at t3  vs  stay offline",
            [("t3", "s")],
            "t2",
            "none",
            "difference",
        ),
        (
            "Pre-trend",
            "start at t1  vs  not, before starting",
            [("t1", "s")],
            "t0",
            "none",
            "difference",
        ),
        (
            "Early vs late adopters",
            "start at t1  vs  start at t3",
            [("t1", "s"), ("t3", "l")],
            "t2",
            "difference",
            "none*",
        ),
    ]
    for i, (name, sub, starts, out, pe, ps) in enumerate(rows):
        y = 3.62 - i * 0.92
        ax.add_patch(
            FancyBboxPatch(
                (0.05, y - 0.36),
                10.4,
                0.72,
                boxstyle="round,pad=0.01,rounding_size=0.06",
                fc=SHADE if i % 2 == 0 else SURF,
                ec="none",
                zorder=-1,
            )
        )
        ax.text(0.15, y + 0.1, name, fontsize=FS * 7, weight="bold", color=INK, va="center")
        ax.text(0.15, y - 0.17, sub, fontsize=FS * 6, color=INK2, va="center")
        ax.plot([X["t−1"], X["t3"]], [y, y], color=MUTED, lw=0.7, zorder=1)
        for w, kind in starts:
            ax.plot(
                X[w], y, marker="^", ms=7, color="#5b2a86" if kind == "s" else "#9b1b30", zorder=3
            )
        ax.plot(X[out], y, marker="s", ms=6.5, mfc="white", mec=INK, mew=1, zorder=4)
        if name == "Pre-trend":
            ax.plot(X["t−1"], y, marker="s", ms=4.5, mfc="white", mec=MUTED, mew=0.8, zorder=4)
        for xx, txt, col in [(8.15, pe, "#5b2a86"), (9.65, ps, "#9b1b30")]:
            ax.text(
                xx,
                y,
                txt,
                ha="center",
                va="center",
                fontsize=FS * 6.6,
                color=col,
                weight="bold" if txt != "none" else "normal",
            )
    hh = [
        Line2D([], [], marker="^", color="#5b2a86", ls="", ms=6, label="starts using the internet"),
        Line2D(
            [], [], marker="^", color="#9b1b30", ls="", ms=6, label="comparison group starts later"
        ),
        Line2D([], [], marker="s", mfc="white", mec=INK, ls="", ms=5.5, label="outcome measured"),
        Line2D([], [], marker="s", mfc="white", mec=MUTED, ls="", ms=4, label="adjusted for"),
    ]
    ax.legend(
        handles=hh,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.07),
        ncol=4,
        frameon=False,
        fontsize=FS * 6,
    )
    ax.text(
        10.45,
        0.03,
        "*if early and late adopters are equally selected (assessed by within-adopter pre-trends)",
        ha="right",
        fontsize=FS * 5.6,
        color=INK2,
    )

    # b  internet use by calendar year
    ax = fig.add_subplot(gs[1, 0])
    lab(ax, "b", title="Internet use, adults aged ≥50")
    pr = pd.read_csv("prevalence.csv")
    pr = pr[~((pr.cohort == "SHARE") & (pr.wave == 7))]
    for c in COH + ["LASI"]:
        s = pr[pr.cohort == c].sort_values("year")
        ax.plot(
            s.year,
            100 * s.prev,
            marker=MK[c],
            color=COL[c],
            ms=2.8,
            lw=1,
            mec=SURF,
            mew=0.3,
            label=c,
        )
    ax.set_ylim(0, 100)
    ax.set_xlim(2001, 2023)
    ax.set_ylabel("Respondents using the internet (%)")
    ax.set_xticks([2002, 2008, 2014, 2020])
    clean(ax, "y")
    ax.legend(
        frameon=False,
        fontsize=FS * 5.4,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.13),
        ncol=2,
        handlelength=1.2,
        borderaxespad=0.1,
        columnspacing=0.8,
    )

    # c  initiation per wave by cohort and calendar year
    ax = fig.add_subplot(gs[1, 1])
    lab(ax, "c", title="Starting within one wave")
    ini = pd.read_csv("initiation.csv")
    for c in COH:
        s = ini[ini.cohort == c].sort_values("year")
        ax.plot(s.year, 100 * s.rate, marker=MK[c], color=COL[c], ms=2.8, lw=1, mec=SURF, mew=0.3)
        ax.plot(
            [],
            [],
            marker=MK[c],
            color=COL[c],
            ms=2.8,
            lw=1,
            label=f"{c}: {s.n.sum():,} person-trials",
        )
    ax.set_ylabel("Non-users online by next wave (%)")
    ax.set_xlabel("Baseline year (t0)")
    ax.set_xlim(2001, 2016.5)
    ax.set_ylim(0, 37)
    ax.set_xticks([2002, 2006, 2010, 2014])
    clean(ax, "y")
    ax.legend(
        frameon=False, fontsize=FS * 5.4, loc="upper left", handlelength=1.2, borderaxespad=0.1
    )

    # d  covariate balance
    ax = fig.add_subplot(gs[1, 2])
    lab(ax, "d", title="Covariate balance (target trials)")
    b = pd.read_csv("balance_hgb_trim0.025_lag_extra.csv")
    b = b[~b["var"].isin(["trial"])]
    order = b.groupby("var").smd_raw.apply(lambda s: s.abs().max()).sort_values().index[-12:]
    nice = {
        "age": "Age",
        "edu": "Education",
        "recall": "Word recall",
        "wealth_q": "Wealth",
        "work": "Working",
        "srh_poor": "Fair/poor health",
        "iadl_any": "IADL difficulty",
        "adl_any": "ADL difficulty",
        "dep": "Depression score",
        "ncond": "Chronic conditions",
        "partnered": "Partnered",
        "live_alone": "Lives alone",
        "female": "Female",
        "rural": "Rural",
        "drink": "Drinks",
        "smoke": "Smokes",
        "vig_weekly": "Vigorous activity",
        "ever_before": "Used before t0",
        "year": "Calendar year",
        "recall_lag": "Recall, t−1",
        "sp_internet": "Spouse online",
        "trial": "Trial",
        "grip": "Grip strength",
        "bmi": "BMI",
        "lonely": "Loneliness",
        "socwk": "Social activity",
        "kcnt": "Contact with children",
    }
    for i, v in enumerate(order):
        s = b[b["var"] == v]
        for _, r in s.iterrows():
            ax.plot(abs(r.smd_raw), i, "o", mfc="none", mec=COL[r.cohort], ms=3, mew=0.6)
            ax.plot(abs(r.smd_w), i, MK[r.cohort], color=COL[r.cohort], ms=3, mec=SURF, mew=0.3)
    ax.axvline(0.1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([nice.get(v, v) for v in order], fontsize=FS * 5.8)
    ax.set_xlabel("|Standardised mean difference|")
    ax.set_xlim(0, max(0.8, b.smd_raw.abs().max() + 0.05))
    clean(ax)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", mfc="none", mec=INK2, ls="", ms=3.5, label="unweighted"),
            Line2D([], [], marker="o", color=INK2, ls="", ms=3.5, label="weighted"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=FS * 5.6,
    )
    save(fig, "NC_Fig1")


# ================================================================== Figure 2
def fig2():
    fig = plt.figure(figsize=(W, 4.6))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1, 1.15],
        hspace=0.62,
        wspace=0.55,
        left=0.12,
        right=0.95,
        top=0.9,
        bottom=0.08,
    )
    rungs = [
        ("Cross-sectional association", "Cross-sectional\nassociation"),
        ("Target trial emulation", "Target trial\nemulation"),
    ]
    spec = [
        ("Y_recall", "a", "Word recall", "Difference (SD)", False, (-0.05, 0.65)),
        ("Y_iadl", "b", "IADL difficulty", "Risk ratio", True, (0.2, 1.3)),
        ("Y_dep", "c", "Elevated depressive symptoms", "Risk ratio", True, (0.4, 1.3)),
    ]
    for j, (y, L, t, xl, rr, xlim) in enumerate(spec):
        ax = fig.add_subplot(gs[0, j])
        lab(ax, L, title=t)
        rows = []
        for rg, rlab in rungs:
            g = LAD[(LAD.rung == rg) & (LAD.y == y) & (LAD.cohort != "Pooled")].copy()
            if rr:
                g["est"], g["lo"], g["hi"] = np.log(g.rr), np.log(g.rr_lo), np.log(g.rr_hi)
                g["se"] = (g.hi - g.lo) / 3.92
            else:
                g["se"] = (g.hi - g.lo) / 3.92
            rows.append((rlab, g))
        if rr:
            forest(ax, rows, (np.log(xlim[0]), np.log(xlim[1])), ref=0, fmt="{:.2f}", vals=False)
            for i, (rl, g) in enumerate(rows):
                p = pool(g[g.cohort.isin(COH + ["LASI"])])
                ax.text(
                    1.01,
                    len(rows) - i + 0.3,
                    f"{np.exp(p['est']):.2f}",
                    transform=ax.get_yaxis_transform(),
                    fontsize=FS * 6,
                    weight="bold",
                    va="center",
                )
            tk = [0.25, 0.5, 0.75, 1.0] if xlim[0] < 0.3 else [0.5, 0.75, 1.0]
            ax.set_xticks(np.log(tk))
            ax.set_xticklabels([f"{v:g}" for v in tk])
        else:
            forest(ax, rows, xlim)
        ax.set_xlabel(xl)
        if j:
            ax.set_yticklabels([])
    # d adjustment ladder
    models = [
        ("hgb_trim0.025", "Demographics + baseline health"),
        ("hgb_trim0.025_lag", "+ prior-wave health"),
        ("hgb_trim0.025_lag_extra", "+ social and physical measures (primary)"),
        ("hgb_trim0.025_lag_t1_extra", "+ health at t1 (conservative)"),
    ]
    shades = ["#a9cbf2", "#6da7ec", "#2a78d6", "#0f3f7c"]
    gb = gs[1, :].subgridspec(1, 2, width_ratios=[1, 2.3], wspace=0.75)
    ax = fig.add_subplot(gb[0])
    lab(ax, "d", title="Recall across adjustment sets")
    for k, (tag, ml) in enumerate(models):
        m = pd.read_csv(f"results_meta_{tag}.csv")
        r = m[(m.y == "Y_recall") & (m.scale == "RD/MD")].iloc[0]
        ax.plot([r.lo, r.hi], [3 - k] * 2, color=shades[k], lw=1.4)
        ax.plot(r.est, 3 - k, "o", color=shades[k], ms=4, mec=SURF, mew=0.5)
        ax.text(
            r.est,
            3 - k + 0.22,
            f"{r.est:.2f}",
            va="bottom",
            ha="center",
            fontsize=FS * 5.8,
            color=INK2,
        )
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["Set 1", "Set 2", "Set 3\n(primary)", "Set 4"])
    ax.set_xlim(0, 0.22)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("Difference (SD)")
    clean(ax)
    ax.set_ylim(-0.6, 3.6)
    # e  RR outcomes incl. negative control and mortality
    ax = fig.add_subplot(gb[1])
    lab(ax, "e", title="Binary outcomes, negative control and mortality benchmark")
    outs = [
        ("Y_iadl", "New IADL difficulty"),
        ("Y_dep", "Elevated depressive symptoms"),
        ("Y_cancer", "New cancer\n(negative control)"),
        ("Y_death", "All-cause death\n(benchmark)"),
    ]
    for j, (y, ol) in enumerate(outs):
        for k, (tag, ml) in enumerate(models):
            m = pd.read_csv(f"results_meta_{tag}.csv")
            r = m[(m.y == y) & (m.scale == "RR")].iloc[0]
            yy = (len(outs) - j) + 0.16 * (1.5 - k)
            ax.plot([r.lo, r.hi], [yy] * 2, color=shades[k], lw=1.2)
            ax.plot(
                r.est,
                yy,
                "o",
                color=shades[k],
                ms=3.4,
                mec=SURF,
                mew=0.4,
                label=ml if j == 0 else None,
            )
    ax.axhspan(0.5, 2.5, color=SHADE, zorder=-2, lw=0)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.6, 0.8, 1.0, 1.25])
    ax.set_xticklabels(["0.5", "0.6", "0.8", "1.0", "1.25"])
    ax.minorticks_off()
    ax.set_xlim(0.4, 1.35)
    ax.axvline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("Risk ratio, initiators vs non-initiators (log scale)")
    ax.set_yticks([4, 3, 2, 1])
    ax.set_yticklabels([o[1] for o in outs])
    clean(ax)
    ax.set_ylim(0.45, 4.55)
    ax.legend(
        frameon=False,
        fontsize=FS * 5.4,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        handlelength=1,
        title="Adjustment set",
        title_fontsize=FS * 5.6,
        alignment="left",
    )
    cohort_legend(fig, y=0.955, cohorts=COH + ["LASI"], ncol=6)
    save(fig, "NC_Fig2")


# ================================================================== Figure 3
def traj(ax, an, groups, labels, pos, starts, ylim, legend=False):
    t = TRJ[(TRJ.analysis == an) & (TRJ.cohort == "Pooled")]
    xs = {p: i for i, p in enumerate(pos)}
    for g in groups:
        s = t[t.grp == g].set_index("pos").reindex(pos)
        x = [xs[p] for p in pos]
        ax.fill_between(
            x, s.est - 1.96 * s.se, s.est + 1.96 * s.se, color=GCOL[g], alpha=0.18, lw=0
        )
        ax.plot(x, s.est, "-o", color=GCOL[g], ms=3, lw=1.2, mec=SURF, mew=0.4)
        if g in starts:
            p = starts[g]
            ax.plot(xs[p], s.est[p] + 0.06, marker="v", color=GCOL[g], ms=4.5)
        if legend:
            ax.plot([], [], "-o", color=GCOL[g], ms=3, lw=1.2, label=labels[g])
        else:
            ax.text(
                len(pos) - 1 + 0.12,
                s.est.iloc[-1],
                labels[g],
                color=GCOL[g],
                fontsize=FS * 6,
                va="center",
                weight="bold",
            )
    ax.set_xticks(range(len(pos)))
    ax.set_xticklabels(pos)
    ax.set_xlim(-0.2, len(pos) - 1 + (0.25 if legend else 0.9))
    ax.set_ylim(*ylim)
    if legend:
        ax.legend(
            frameon=False,
            fontsize=FS * 5.6,
            loc="lower left",
            handlelength=1.4,
            borderaxespad=0.1,
            title="first use",
            title_fontsize=FS * 5.6,
            alignment="left",
        )
    ax.set_ylabel("Word recall, age- and sex-\nstandardised mean (SD)")
    ax.set_xlabel("Survey wave")
    clean(ax, "y")


def fig3():
    fig = plt.figure(figsize=(W, 4.9))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1, 1],
        hspace=0.6,
        wspace=0.62,
        left=0.09,
        right=0.95,
        top=0.9,
        bottom=0.08,
    )
    ax = fig.add_subplot(gs[0, 0])
    lab(ax, "a", title="Recall before and after starting")
    traj(
        ax,
        "tte",
        ["start", "stay"],
        {"start": "initiators", "stay": "non-\ninitiators"},
        ["t−1", "t0", "t1", "t2"],
        {"start": "t1"},
        (-0.32, 0.5),
    )
    ax.axvspan(1.5, 2.5, color="#ebe4f2", zorder=-3, lw=0)
    ax.text(2, 0.47, "start", ha="center", fontsize=FS * 5.8, color="#5b2a86")
    t = TRJ[(TRJ.analysis == "tte") & (TRJ.cohort == "Pooled")].pivot(
        index="pos", columns="grp", values="est"
    )
    gap = (t.start - t.stay).reindex(["t−1", "t0", "t1", "t2"])
    for i, v in enumerate(gap):
        ax.text(i, -0.29, f"gap {v:.2f}", ha="center", fontsize=FS * 5.3, color=INK2)
    rows_r = [
        ("Target trial\n(start at t1)", ci(sel(TTE, y="Y_recall"))),
        ("Placebo\n(start at t3)", ci(sel(PLA, y="Y_recall"))),
        ("Placebo\n(start at t4)", ci(sel(PL2, y="Y_recall"))),
        ("Pre-trend\n(recall at t0)", ci(sel(PRE, y="Y_pre"))),
    ]
    ax = fig.add_subplot(gs[0, 1:])
    lab(ax, "b", title="Word recall: target trial, placebo trials and pre-trend")
    forest(ax, rows_r, (-0.15, 0.45), shade=(1, 2))
    ax.set_xlabel("Difference, initiators vs non-initiators (SD)")
    rows_i = [
        ("Target trial", ci(sel(TTE, y="Y_iadl"))),
        ("Placebo (t3)", ci(sel(PLA, y="Y_iadl"))),
        ("Placebo (t4)", ci(sel(PL2, y="Y_iadl"))),
        ("Pre-trend", ci(sel(PRE, y="Y_iadl_pre"))),
    ]
    ax = fig.add_subplot(gs[1, 0])
    lab(ax, "c", title="New IADL difficulty")
    forest(ax, rows_i, (-12, 6), scale=100, shade=(1, 2), fmt="{:.1f}")
    ax.set_xlabel("Risk difference (percentage points)")
    rows_d = [
        ("Target trial", ci(sel(TTE, y="Y_dep"))),
        ("Placebo (t3)", ci(sel(PLA, y="Y_dep"))),
        ("Placebo (t4)", ci(sel(PL2, y="Y_dep"))),
    ]
    ax = fig.add_subplot(gs[1, 1])
    lab(ax, "d", title="Elevated depressive symptoms")
    forest(ax, rows_d, (-12, 8), scale=100, shade=(1, 2), fmt="{:.1f}")
    ax.set_xlabel("Risk difference (percentage points)")
    # e target minus placebo
    ax = fig.add_subplot(gs[1, 2])
    lab(ax, "e", title="Target trial minus placebo")
    d1 = ci(sel(DIF, y="Y_recall"))
    kn = sel(R3, analysis="tte_known_t3", y="Y_recall").set_index("cohort")
    pl = sel(PLA, y="Y_recall").set_index("cohort")
    d2 = pd.DataFrame(
        {
            "cohort": kn.index,
            "est": kn.est - pl.est.reindex(kn.index),
            "se": np.sqrt(kn.se**2 + pl.se.reindex(kn.index) ** 2),
        }
    )
    d2 = ci(d2)
    forest(
        ax,
        [("Full target-trial\npopulation", d1), ("Alive, status\nknown at t3", d2)],
        (-0.25, 0.2),
        fmt="{:+.3f}",
    )
    ax.set_xlabel("Difference in recall (SD)")
    cohort_legend(fig, y=0.955)
    save(fig, "NC_Fig3")


# ================================================================== Figure 4
FIG4NAME = "NC_Fig4"
FIG5NAME = "NC_Fig5"


def fig4():
    fig = plt.figure(figsize=(W, 6.9))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.1, 1, 1],
        height_ratios=[0.8, 1.25],
        hspace=0.42,
        wspace=0.65,
        left=0.09,
        right=0.97,
        top=0.91,
        bottom=0.07,
    )
    ax = fig.add_subplot(gs[0, 0])
    lab(ax, "a", title="Recall by timing of first use")
    traj(
        ax,
        "timing",
        ["early", "middle", "late", "never"],
        {"early": "early (t1)", "middle": "middle (t2)", "late": "late (t3)", "never": "never"},
        ["t−1", "t0", "t1", "t2", "t3"],
        {"early": "t1", "middle": "t2", "late": "t3"},
        (-0.3, 0.62),
        legend=True,
    )
    ax.axvline(3, color=INK2, lw=0.5, ls=(0, (2, 2)))
    ax.text(3, 0.6, "outcome", ha="center", fontsize=FS * 5.6, color=INK2)
    W1 = lambda ct, y: ci(WIT[(WIT.contrast == ct) & (WIT.y == y)])
    ax = fig.add_subplot(gs[0, 1:])
    lab(ax, "b", title="Each adopter group compared with never-starters, and early with late")
    rows = [
        ("Early (t1) vs never", W1("early vs never", "Y_recall")),
        ("Middle (t2) vs never", W1("middle vs never", "Y_recall")),
        ("Late (t3) vs never\n= placebo", W1("late vs never", "Y_recall")),
        ("Early vs late", W1("early vs late", "Y_recall")),
    ]
    forest(ax, rows, (-0.15, 0.4), shade=(3,), band=0.10, fmt="{:.3f}", band_rows=(3,))
    ax.set_xlabel("Difference in word recall at t2 (SD)")
    gb = gs[1, :].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.5)
    gr = gb[1].subgridspec(2, 1, height_ratios=[0.8, 1.2], hspace=0.75)
    ax = fig.add_subplot(gr[0])
    lab(ax, "c", title="Early vs late: other outcomes")
    forest(
        ax,
        [("IADL", W1("early vs late", "Y_iadl")), ("Depr. sx", W1("early vs late", "Y_dep"))],
        (-8, 8),
        scale=100,
        band=2,
        fmt="{:+.1f}",
    )
    ax.set_xlabel("Risk difference (pp)")
    # d specification curve (pooled, HKSJ 90% and 95%)
    ax = fig.add_subplot(gb[0])
    lab(ax, "d", title="Early vs late in recall across specifications (pooled)")
    sp = spec_rows()
    n = len(sp)
    ax.axvspan(-0.10, 0.10, color="#ebe4f2", zorder=-3, lw=0)
    for i, (name, p, grp) in enumerate(sp):
        yy = n - i
        if grp:
            ax.axhspan(yy - 0.5, yy + 0.5, color=SHADE, zorder=-2, lw=0)
        lo95, hi95 = (p["hk_lo"], p["hk_hi"]) if p["k"] >= 3 else (p["dl_lo"], p["dl_hi"])
        lo90, hi90 = (
            (p["hk90_lo"], p["hk90_hi"])
            if p["k"] >= 3
            else (
                p["est"] - 1.645 * (p["dl_hi"] - p["dl_lo"]) / 3.92,
                p["est"] + 1.645 * (p["dl_hi"] - p["dl_lo"]) / 3.92,
            )
        )
        ax.plot([lo95, hi95], [yy] * 2, color=INK2, lw=0.7)
        ax.plot([lo90, hi90], [yy] * 2, color=INK, lw=2.0, solid_capstyle="butt")
        ax.plot(p["est"], yy, "o", color=SURF, mec=INK, ms=3.2, mew=0.8, zorder=5)
        ax.text(
            1.01,
            yy,
            f"{p['est']:+.3f}".replace("-", "−"),
            transform=ax.get_yaxis_transform(),
            fontsize=FS * 5.8,
            va="center",
            color=INK,
        )
        ax.text(
            0.99,
            yy,
            f"k={p['k']}",
            transform=ax.get_yaxis_transform(),
            fontsize=FS * 5,
            va="center",
            ha="right",
            color=MUTED,
        )
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([s[0] for s in sp], fontsize=FS * 5.8)
    ax.set_ylim(0.4, n + 0.6)
    ax.set_xlim(-0.22, 0.26)
    ax.set_xlabel("Difference in recall (SD); thick 90% CI, thin 95% CI")
    clean(ax)
    # e sustained users vs never users, before and after starting
    ax = fig.add_subplot(gr[1])
    lab(ax, "e", title="Sustained users vs never users")
    ST = pd.read_csv("results_sustained_traj.csv")
    SP = pd.read_csv("results_sustained_traj_pooled.csv")
    pos = ["t-1", "t0", "t1", "t2", "t3"]
    xs = np.arange(5)
    for j, c in enumerate(["HRS", "ELSA", "SHARE"]):
        v = [ST[(ST.cohort == c) & (ST.term == "traj:" + p)].est.iloc[0] for p in pos]
        ax.plot(xs + (j - 1) * 0.06, v, color=COL[c], lw=0.6, alpha=0.6, marker=MK[c], ms=2)
    pp = [SP[SP.term == "traj:" + p].iloc[0] for p in pos]
    ax.fill_between(xs, [r.lo for r in pp], [r.hi for r in pp], color=INK, alpha=0.1, lw=0)
    ax.plot(xs, [r.est for r in pp], color=INK, lw=1.1, marker="o", ms=2.8, mec=SURF, mew=0.4)
    ax.axvspan(1.5, 3.5, color="#ebe4f2", zorder=-3, lw=0)
    ax.text(2.5, 0.02, "online at\nt1 and t2", ha="center", fontsize=FS * 5.4, color="#5b2a86")
    ax.set_xticks(xs)
    ax.set_xticklabels(["t−1", "t0", "t1", "t2", "t3"])
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Recall difference (SD)")
    clean(ax, grid="y")
    ax.spines["left"].set_visible(True)
    cohort_legend(
        fig, y=0.955, extra=[Rectangle((0, 0), 1, 1, fc="#ebe4f2", label="equivalence margin")]
    )
    save(fig, FIG4NAME)


def spec_rows():
    out = []

    def add(name, g, grp=False):
        g = g[g.cohort.isin(COH)]
        out.append((name, hksj(g.est.values, se_of(ci(g))), grp))

    s = lambda m: SENS[
        (SENS.design == "early_vs_late") & (SENS.y == "Y_recall") & (SENS.method == m)
    ]
    add("Primary (overlap 0.025–0.975)", s("trim"))
    add("Untrimmed", s("untrim"))
    add("Overlap weights (ATO)", s("ato"))
    add("Baseline before 2010", s("era<2010"))
    add("Baseline from 2010", s("era>=2010"))
    add("Excluding waves from 2020", sel(R3, analysis="early_vs_late_precovid", y="Y_recall"))
    add("Without SHARE", sel(WIT, contrast="early vs late", y="Y_recall").query("cohort!='SHARE'"))
    add("Never used before t0", sel(W2, variant="never_before", y="Y_recall"))
    add("No life transitions", sel(W2, variant="no_transitions", y="Y_recall"))
    add("Early vs very late (start t4)", sel(R3, analysis="early_vs_verylate", y="Y_recall"))
    add("Before starting (t0, adj. t−1)", sel(W2, variant="pretrend_within", y="Y_pre"), True)
    add("Sustained early users vs late", sel(W2, variant="sustained", y="Y_recall"), True)
    add("Early users who stopped vs late", sel(W3, variant="stopped_by_t2", y="Y_recall"), True)
    return out


# ================================================================== Figure 5
def fig5():
    d = pd.read_csv("simulation_grid3.csv")
    d["fit"] = d.fit4
    VC = {"base": "#2a78d6", "trigger": "#eb6834", "nonpersist": "#1baf7a", "cumulative": "#e87ba4"}
    VL = {
        "base": "Selection on cognition",
        "trigger": "+ adverse-event trigger",
        "nonpersist": "+ 30% stop each wave",
        "cumulative": "+ effect grows with use",
    }
    fig = plt.figure(figsize=(W, 5.6))
    gs = fig.add_gridspec(
        2,
        3,
        hspace=0.55,
        wspace=0.62,
        left=0.08,
        right=0.97,
        top=0.92,
        bottom=0.08,
        width_ratios=[1, 1, 1.15],
    )
    ax = fig.add_subplot(gs[0, 0])
    lab(ax, "a", title=f"{len(d)} simulated scenarios")
    for v in VC:
        s = d[d.variant == v]
        ax.scatter(s.TTE, s.placebo, s=4, color=VC[v], alpha=0.55, lw=0, label=VL[v])
    ax.add_patch(
        Rectangle((0.089, 0.091), 0.099, 0.127, fill=False, ec=INK, lw=0.8, ls=(0, (3, 2)))
    )
    ax.plot(0.138, 0.155, "D", color=INK, ms=3.5)
    ax.text(0.03, 0.228, "observed 95% CIs", fontsize=FS * 5.5, color=INK)
    ax.set_xlabel("Simulated target-trial estimate (SD)")
    ax.set_ylabel("Simulated placebo estimate (SD)")
    clean(ax, "both")
    ax = fig.add_subplot(gs[0, 1])
    f = d[d.fit]
    lab(ax, "b", title=f"{len(f)} scenarios fit all four")
    rng = np.random.default_rng(1)
    for v in VC:
        s = f[f.variant == v]
        if len(s):
            ax.scatter(
                s.beta + rng.uniform(-0.006, 0.006, len(s)),
                s.EL,
                s=9,
                color=VC[v],
                lw=0.3,
                ec="white",
            )
    ax.axhspan(-0.059, 0.082, color=SHADE, zorder=-3)
    ax.axhline(0.012, color=INK, lw=0.8)
    ax.text(0.205, 0.0, "observed 0.012", fontsize=FS * 5.5, ha="right", va="top")
    ax.text(0.205, 0.086, "95% CI", fontsize=FS * 5.5, ha="right", color=INK2)
    ax.plot([-0.01, 0.21], [-0.01, 0.21], color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlim(-0.015, 0.215)
    ax.set_xlabel("True effect of starting (SD)")
    ax.set_ylabel("Simulated early − late (SD)")
    clean(ax, "both")
    h = [Line2D([], [], marker="o", color=VC[v], ls="", ms=3.5, label=VL[v]) for v in VC]
    ax.legend(
        handles=h,
        frameon=False,
        fontsize=FS * 5.4,
        loc="upper left",
        handletextpad=0.2,
        borderaxespad=0.1,
    )
    # c robustness of target-trial and placebo estimates
    ax = fig.add_subplot(gs[0, 2])
    lab(ax, "c", title="Target trial vs placebo by analysis")
    rows = []

    def pr(design, m):
        return SENS[(SENS.design == design) & (SENS.y == "Y_recall") & (SENS.method == m)]

    specs = [
        ("Primary", pr("main", "trim"), pr("placebo", "trim")),
        ("Untrimmed", pr("main", "untrim"), pr("placebo", "untrim")),
        ("Overlap weights", pr("main", "ato"), pr("placebo", "ato")),
        ("Before 2010", pr("main", "era<2010"), pr("placebo", "era<2010")),
        ("From 2010", pr("main", "era>=2010"), pr("placebo", "era>=2010")),
        (
            "Excl. 2020 waves",
            sel(R3, analysis="tte_precovid", y="Y_recall"),
            sel(R3, analysis="placebo_precovid", y="Y_recall"),
        ),
        (
            "Without SHARE",
            sel(TTE, y="Y_recall").query("cohort!='SHARE'"),
            sel(PLA, y="Y_recall").query("cohort!='SHARE'"),
        ),
        ("Alive at t3", sel(R3, analysis="tte_known_t3", y="Y_recall"), sel(PLA, y="Y_recall")),
        ("Loss-weighted (t1)", sel(ATT, analysis="ipcw_t1", y="Y_recall"), None),
    ]
    n = len(specs)
    for i, (nm, a, b) in enumerate(specs):
        yy = n - i
        for g, off, col, mk in [(a, 0.15, "#5b2a86", "o"), (b, -0.15, "#9b1b30", "s")]:
            if g is None:
                continue
            g = g[g.cohort.isin(COH)]
            p = hksj(g.est.values, se_of(ci(g)))
            lo, hi = (p["hk_lo"], p["hk_hi"]) if p["k"] >= 3 else (p["dl_lo"], p["dl_hi"])
            ax.plot([lo, hi], [yy + off] * 2, color=col, lw=0.9)
            ax.plot(p["est"], yy + off, mk, color=col, ms=3, mec=SURF, mew=0.3)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([s[0] for s in specs], fontsize=FS * 5.8)
    ax.set_ylim(-0.9, n + 0.6)
    ax.set_xlim(-0.05, 0.35)
    ax.set_xlabel("Difference in recall (SD), 95% CI")
    clean(ax)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color="#5b2a86", ms=3, label="target trial (start t1)"),
            Line2D([], [], marker="s", color="#9b1b30", ms=3, label="placebo (start t3)"),
        ],
        frameon=False,
        fontsize=FS * 5.5,
        loc="lower right",
        borderaxespad=0.1,
    )
    # d years of decline
    gbot = gs[1, :].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=0.75)
    ax = fig.add_subplot(gbot[0])
    lab(
        ax,
        "d",
        x=-0.005,
        title="Recall estimates as years of age-related decline (0.05 SD per year)",
    )
    hk = pd.read_csv("results_pool_hk.csv")

    def g(an, y="Y_recall"):
        r = hk[(hk.analysis == an) & (hk.y == y) & (hk.scale == "RD/MD")].iloc[0]
        return r.est, r.hk_lo, r.hk_hi

    items = [
        ("Cross-sectional association", *g("cross-sectional"), "#8a8a85"),
        ("Target trial emulation", *g("target trial"), "#5b2a86"),
        ("Placebo trial (start after outcome)", *g("placebo lead 1"), "#9b1b30"),
        ("Pre-trend (before anyone starts)", *g("pre-trend", "Y_pre"), "#9b1b30"),
        ("Early vs late adopters", *g("early vs late"), "#0f3f7c"),
    ]
    n = len(items)
    for i, (nm, e, lo, hi, col) in enumerate(items):
        yy = n - i
        ax.barh(yy, e / 0.05, height=0.55, color=col, alpha=0.85, lw=0)
        ax.plot([lo / 0.05, hi / 0.05], [yy] * 2, color=INK, lw=0.8)
        ax.text(
            max(hi / 0.05, e / 0.05) + 0.15,
            yy,
            f"{e:.3f} SD ≈ {e / 0.05:.1f} years",
            va="center",
            fontsize=FS * 5.8,
            color=INK,
        )
    ax.axvspan(0, 2.0, ymin=0, ymax=1, color="#ebe4f2", zorder=-3, lw=0)
    ax.text(
        1.95,
        0.12,
        "effects compatible with the data (≤0.10 SD)",
        ha="right",
        fontsize=FS * 5.6,
        color="#5b2a86",
        va="center",
    )
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([s[0] for s in items])
    ax.set_ylim(-0.15, n + 0.5)
    ax.set_xlim(-1.5, 12.5)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Years of decline among non-initiators (95% CI, Hartung–Knapp)")
    clean(ax)
    if FIG5NAME != "NC_Fig5":
        ax = fig.add_subplot(gbot[1])
        lab(ax, "e", x=-0.02, title="Placebo estimate relative to target trial")
        CAL = pd.read_csv("results_calibrated.csv")
        n = len(CAL)
        for i, r in CAL.reset_index(drop=True).iterrows():
            yy = n - i
            ax.plot([r.lo, r.hi], [yy] * 2, color=INK, lw=0.8)
            ax.plot(r.d, yy, "D", color=INK, ms=3)
        ax.axvline(1, color=INK2, lw=0.6, ls=(0, (3, 2)))
        ax.axvline(0, color=MUTED, lw=0.5)
        ax.text(1.02, 0.55, "placebo = target trial", fontsize=FS * 5.3, color=INK2, va="bottom")
        ax.set_yticks([n - i for i in range(n)])
        ax.set_yticklabels(CAL.label, fontsize=FS * 5.8)
        ax.set_ylim(0.4, n + 0.6)
        ax.set_xlim(-0.2, 2.4)
        ax.set_xlabel("Placebo ÷ target-trial estimate (95% CI)")
        clean(ax)
    save(fig, FIG5NAME)


if __name__ == "__main__":
    import sys

    for f in sys.argv[1:] or ["1", "2", "3", "4", "5"]:
        globals()["fig" + f]()


def supp_traj():
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.9), sharex="row")
    for j, c in enumerate(COH):
        for i, (an, groups, pos) in enumerate(
            [
                ("tte", ["start", "stay"], ["t−1", "t0", "t1", "t2"]),
                ("timing", ["early", "middle", "late", "never"], ["t−1", "t0", "t1", "t2", "t3"]),
            ]
        ):
            ax = axes[i, j]
            t = TRJ[(TRJ.analysis == an) & (TRJ.cohort == c)]
            for g in groups:
                s = t[t.grp == g].set_index("pos").reindex(pos)
                ax.fill_between(
                    range(len(pos)),
                    s.est - 1.96 * s.se,
                    s.est + 1.96 * s.se,
                    color=GCOL[g],
                    alpha=0.18,
                    lw=0,
                )
                ax.plot(
                    range(len(pos)),
                    s.est,
                    "-o",
                    color=GCOL[g],
                    ms=2.5,
                    lw=1,
                    label=g if j == 0 else None,
                )
            ax.set_xticks(range(len(pos)))
            ax.set_xticklabels(pos)
            clean(ax, "y")
            if i == 0:
                ax.set_title(NAMES[c], fontsize=FS * 7, weight="bold", color=COL[c])
            if j == 0:
                ax.set_ylabel("Recall (SD), age-sex\nstandardised")
                ax.legend(frameon=False, fontsize=FS * 5.4, loc="lower left")
    lab(axes[0, 0], "a", x=-0.25)
    lab(axes[1, 0], "b", x=-0.25)
    fig.tight_layout()
    save(fig, "NC_SuppFig_traj")
