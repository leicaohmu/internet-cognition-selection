from scipy import stats
from matplotlib.lines import Line2D

"""v1.2 figure redesign: more varied chart types and a common visual grammar.
Blue = what an effect of going online predicts; orange = what selection predicts."""
import shutil
import numpy as np
import pandas as pd
import matplotlib
import ncfig
from ncfig import *
import ncfig6 as F
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update(
    {
        "font.size": 6.8 * FS,
        "axes.titlesize": 7.2 * FS,
        "axes.labelsize": 6.8 * FS,
        "xtick.labelsize": 6.3 * FS,
        "ytick.labelsize": 6.5 * FS,
        "legend.fontsize": 6.2 * FS,
    }
)
EFF, SELC, GREY = "#5b2a86", "#9b1b30", "#8a8a85"
OBS, OBS2, OBS3 = (
    "#0f5c6e",
    "#2f8a9c",
    "#7ab5c2",
)  # observed data: initiators / earlier → later adopters
C4 = COH


def cohort_legend(fig, y=1.0, extra=(), cohorts=COH, x=0.5, ncol=None, pooled=None):
    """Cohort colours and markers; pooled estimates in teal (Figs 2-5) unless another colour is passed."""
    pc = OBS if pooled is None else pooled
    h = [
        Line2D([], [], marker=MK[c], color=COL[c], ls="-", lw=0.9, ms=3.5, label=NAMES[c])
        for c in cohorts
    ]
    h.append(
        Line2D([], [], marker="D", color=pc, ls="-", lw=1.4, ms=4, label="Pooled across cohorts")
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
        columnspacing=1.3,
    )


def pooled(g, scale=1.0):
    g = g[g.cohort.isin(COH)]
    se = g.se.values if "se" in g and g.se.notna().all() else ((g.hi - g.lo) / 3.92).values
    if len(g) == 1:
        return dict(
            est=g.est.iloc[0] * scale,
            lo=(g.est.iloc[0] - 1.96 * se[0]) * scale,
            hi=(g.est.iloc[0] + 1.96 * se[0]) * scale,
            k=1,
        )
    r = hksj(g.est.values, se)
    lo, hi = (r["hk_lo"], r["hk_hi"]) if r["k"] >= 3 else (r["dl_lo"], r["dl_hi"])
    return dict(est=r["est"] * scale, lo=lo * scale, hi=hi * scale, k=r["k"])


def panel_title(ax, L, t, x=-0.02):
    lab(ax, L, x=x, title=t)


# ================================================================== Fig. 2  shrinkage dumbbells + adjustment ladder + attenuation matrix
def _xs_tte(y, rr):
    xs = LAD[(LAD.rung == "Cross-sectional association") & (LAD.y == y)].set_index("cohort")
    tt = LAD[(LAD.rung == "Target trial emulation") & (LAD.y == y)].set_index("cohort")
    lg = (
        (lambda z: z.assign(est=np.log(z.rr), lo=np.log(z.rr_lo), hi=np.log(z.rr_hi)))
        if rr
        else (lambda z: z)
    )
    return xs, tt, pooled(lg(xs.reset_index())), pooled(lg(tt.reset_index()))


def fig2():
    fig = plt.figure(figsize=(W, 6.6))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.05, 1.25], hspace=0.40, left=0.09, right=0.98, top=0.885, bottom=0.07
    )
    g1 = gs[0].subgridspec(1, 3, wspace=0.12)
    spec = [
        ("Y_recall", "a", "Word recall", "Difference (SD)", False, (-0.02, 0.62)),
        ("Y_iadl", "b", "New IADL difficulty", "Risk ratio (log scale)", True, (0.25, 1.25)),
        ("Y_dep", "c", "Elevated depressive symptoms", "Risk ratio (log scale)", True, (0.45, 1.2)),
    ]
    order = ["LASI", "CHARLS", "SHARE", "ELSA", "HRS"]
    rows = order + ["Pooled"]
    for j, (y, L, t, xl, rr, xlim) in enumerate(spec):
        ax = fig.add_subplot(g1[j])
        panel_title(ax, L, t)
        xs, tt, pa, pb = _xs_tte(y, rr)
        val = (lambda d, c: d.loc[c, "rr"]) if rr else (lambda d, c: d.loc[c, "est"])
        for i, c in enumerate(rows):
            col = OBS if c == "Pooled" else COL[c]
            if c == "Pooled":
                x0, x1 = (np.exp(pa["est"]), np.exp(pb["est"])) if rr else (pa["est"], pb["est"])
                ax.axhspan(i - 0.45, i + 0.45, color=SHADE, zorder=-3, lw=0)
                # cross-sectional and target-trial CIs for the pooled row
                for p_, yo in [(pa, 0.22), (pb, -0.22)]:
                    lo, hi = (np.exp(p_["lo"]), np.exp(p_["hi"])) if rr else (p_["lo"], p_["hi"])
                    ax.plot([lo, hi], [i + yo] * 2, color=OBS, lw=0.7, alpha=0.6)
            else:
                if c not in xs.index:
                    continue
                x0 = val(xs, c)
                x1 = val(tt, c) if c in tt.index else None
                lo, hi = (
                    (xs.loc[c, "rr_lo"], xs.loc[c, "rr_hi"])
                    if rr
                    else (xs.loc[c, "lo"], xs.loc[c, "hi"])
                )
                ax.plot([lo, hi], [i + 0.2] * 2, color=col, lw=0.6, alpha=0.45)
                if x1 is not None:
                    lo, hi = (
                        (tt.loc[c, "rr_lo"], tt.loc[c, "rr_hi"])
                        if rr
                        else (tt.loc[c, "lo"], tt.loc[c, "hi"])
                    )
                    ax.plot([lo, hi], [i - 0.2] * 2, color=col, lw=0.6, alpha=0.45)
            ax.plot(x0, i, "o", mfc=SURF, mec=col, ms=5.2, mew=1.1, zorder=4)
            if x1 is not None:
                ax.annotate(
                    "",
                    xy=(x1, i),
                    xytext=(x0, i),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=col,
                        lw=1.0,
                        shrinkA=3.5,
                        shrinkB=2.5,
                        mutation_scale=7,
                    ),
                )
                ax.plot(x1, i, "o", color=col, ms=5.2, mec=SURF, mew=0.5, zorder=5)
                shr = (1 - (np.log(x1) / np.log(x0) if rr else x1 / x0)) * 100
                ax.text(
                    1.0,
                    i,
                    f"−{shr:.0f}%",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=FS * 5.6,
                    color=col,
                    weight="bold" if c == "Pooled" else "normal",
                )
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(
            [NAMES.get(c, c).split(" (")[0] if c != "Pooled" else "Pooled" for c in rows]
            if j == 0
            else []
        )
        if rr:
            ax.set_xscale("log")
            tk = [0.25, 0.5, 0.75, 1.0] if xlim[0] < 0.3 else [0.5, 0.75, 1.0]
            ax.set_xticks(tk)
            ax.set_xticklabels([f"{v:g}" for v in tk])
            ax.minorticks_off()
            ax.axvline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        else:
            ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel(xl)
        clean(ax)
    h = [
        Line2D(
            [],
            [],
            marker="o",
            mfc=SURF,
            mec=INK,
            ls="",
            ms=5,
            label="Cross-sectional association (users vs non-users)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color=INK,
            ls="",
            ms=5,
            label="Target trial emulation (starting vs not starting)",
        ),
        Line2D([], [], color=INK, lw=0, label="−%: share of the association removed"),
    ]
    fig.legend(
        handles=h,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=3,
        frameon=False,
        fontsize=FS * 6.0,
        handletextpad=0.3,
        columnspacing=1.4,
    )
    # d, e adjustment ladders; f attenuation matrix
    models = [
        "hgb_trim0.025",
        "hgb_trim0.025_lag",
        "hgb_trim0.025_lag_extra",
        "hgb_trim0.025_lag_t1_extra",
    ]
    mlab = ["Base-\nline", "+ prior\nhealth", "+ social,\nphysical", "+ t1\nhealth"]
    M = [pd.read_csv(f"results_meta_{t}.csv") for t in models]
    gb = gs[1].subgridspec(1, 3, width_ratios=[1.0, 1.6, 0.95], wspace=0.85)
    ax = fig.add_subplot(gb[0])
    panel_title(ax, "d", "Recall by adjustment set")
    r = [m[(m.y == "Y_recall") & (m.scale == "RD/MD")].iloc[0] for m in M]
    x = np.arange(4)
    xs, _, pa, _ = _xs_tte("Y_recall", False)
    ax.axhspan(pa["lo"], pa["hi"], color=GREY, alpha=0.18, lw=0)
    ax.axhline(pa["est"], color=GREY, lw=0.9, ls=(0, (3, 2)))
    ax.text(
        1.5,
        pa["est"] + 0.012,
        f'cross-sectional {pa["est"]:.2f}',
        va="bottom",
        ha="center",
        fontsize=FS * 5.4,
        color=INK2,
    )
    ax.fill_between(x, [q.lo for q in r], [q.hi for q in r], color=OBS, alpha=0.15, lw=0)
    ax.plot(x, [q.est for q in r], color=OBS, marker="o", ms=4.5, lw=1.4, mec=SURF, mew=0.5)
    for i, q in enumerate(r):
        ax.text(
            i,
            q.hi + 0.012,
            f"{q.est:.2f}",
            ha="center",
            fontsize=FS * 5.6,
            color=OBS,
            weight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(mlab, fontsize=FS * 4.7)
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(0, 0.46)
    ax.set_ylabel("Difference in recall (SD)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    ax = fig.add_subplot(gb[1])
    panel_title(ax, "e", "Binary outcomes by adjustment set")
    LBL = []
    outs = [
        ("Y_iadl", "IADL difficulty", OBS),
        ("Y_dep", "Depressive sx", OBS3),
        ("Y_cancer", "Cancer (neg. control)", GREY),
        ("Y_death", "Death (benchmark)", SELC),
    ]
    for k, (y, nm, col) in enumerate(outs):
        r = [m[(m.y == y) & (m.scale == "RR")].iloc[0] for m in M]
        xx = x + (k - 1.5) * 0.07
        for xi, q in zip(xx, r):
            ax.plot([xi, xi], [q.lo, q.hi], color=col, lw=0.9, alpha=0.7)
        ax.plot(xx, [q.est for q in r], color=col, marker="o", ms=4, lw=1.4, mec=SURF, mew=0.4)
        LBL.append((r[-1].est, f"{nm} {r[-1].est:.2f}", col))
    LBL.sort(key=lambda t: t[0])
    pos = []
    for v, txt, col in LBL:
        ly = np.log(v)
        if pos and ly - pos[-1] < 0.12:
            ly = pos[-1] + 0.12
        pos.append(ly)
        ax.text(3.25, np.exp(ly), txt, va="center", fontsize=FS * 5.5, color=col, weight="bold")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.6, 0.8, 1.0, 1.25])
    ax.set_yticklabels(["0.5", "0.6", "0.8", "1.0", "1.25"])
    ax.minorticks_off()
    ax.plot([-0.25, 3.15], [1, 1], color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(-0.25, 5.9)
    ax.set_xticks(x)
    ax.set_xticklabels(mlab, fontsize=FS * 4.7)
    ax.set_ylabel("Risk ratio (log scale)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    forest_all(fig, gb[2], "f")
    cohort_legend(fig, y=0.935, cohorts=COH + ["LASI"], ncol=6)
    save(fig, "NH3_Fig2")


def _forest_rows(ax, rows, scale, xlim, ref=0, unit_fmt="{:+.2f}", ref_band=None):
    """rows: list of (label, df with cohort/est/lo/hi[/se]). Pooled point + 95% CI (black), cohort dots."""
    n = len(rows)
    if ref_band is not None:
        ax.axvspan(*ref_band, color=GREY, alpha=0.12, lw=0)
    for i, (lab, g) in enumerate(rows):
        yy = n - 1 - i
        g = ci(g)
        for c in COH:
            q = g[g.cohort == c]
            if len(q):
                ax.plot(
                    q.est.iloc[0] * scale,
                    yy + 0.24,
                    MK[c],
                    color=COL[c],
                    ms=3.0,
                    mec=SURF,
                    mew=0.3,
                    alpha=0.9,
                )
        p = pooled(g, scale)
        ax.plot(
            [max(p["lo"], xlim[0]), min(p["hi"], xlim[1])],
            [yy] * 2,
            color=OBS,
            lw=1.3,
            solid_capstyle="round",
        )
        ax.plot(p["est"], yy, "o", color=OBS, ms=4.6, mec=SURF, mew=0.6, zorder=5)
        ax.text(
            1.02,
            yy,
            unit_fmt.format(p["est"]).replace("-", "−"),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=FS * 5.6,
            color=OBS,
        )
    ax.axvline(ref, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(*xlim)
    clean(ax)


def forest_all(fig, cell, L):
    NO = pd.read_csv("results_newout_all.csv")
    DM = pd.read_csv("results_dementia_HRS.csv")
    gg = cell.subgridspec(2, 1, hspace=0.75)
    ax = fig.add_subplot(gg[0])
    panel_title(ax, L, "Target trial, all outcomes")
    nt = lambda o: NO[(NO.outcome == o) & (NO.design == "tte")]
    _forest_rows(
        ax,
        [
            ("Word recall", sel(TTE, y="Y_recall")),
            ("Serial 7s", nt("ser7")),
            ("Verbal fluency", nt("verbf")),
            ("Composite", nt("gcog")),
            ("Life satisfaction", nt("lsat")),
        ],
        1,
        (-0.12, 0.42),
    )
    ax.set_xlabel("Difference (SD)")
    ax = fig.add_subplot(gg[1])
    _forest_rows(
        ax,
        [
            ("Impairment (HRS)", nt("cimp")),
            ("Dementia (HRS)", DM[(DM.outcome == "dem") & (DM.design == "tte")]),
            ("IADL difficulty", sel(TTE, y="Y_iadl")),
            ("Depressive sx", sel(TTE, y="Y_dep")),
            ("Fair/poor health", nt("srh")),
        ],
        100,
        (-8, 4),
        unit_fmt="{:+.1f}",
    )
    ax.set_xlabel("Risk difference (pp)")


def ed_attn(ax, L):
    panel_title(ax, L, "Association removed by design")
    cols = [("Y_recall", "Recall", False), ("Y_iadl", "IADL", True), ("Y_dep", "Depr.", True)]
    rws = ["HRS", "ELSA", "SHARE", "CHARLS", "Pooled"]
    cmap = LinearSegmentedColormap.from_list("att", ["#eef4f5", "#8fbdc7", "#0f5c6e"])
    for j, (y, nm, rr) in enumerate(cols):
        xs, tt, pa, pb = _xs_tte(y, rr)
        for i, c in enumerate(rws):
            if c == "Pooled":
                v = 1 - pb["est"] / pa["est"]
            else:
                if c not in tt.index:
                    continue
                v = 1 - (
                    np.log(tt.loc[c, "rr"]) / np.log(xs.loc[c, "rr"])
                    if rr
                    else tt.loc[c, "est"] / xs.loc[c, "est"]
                )
            yy = len(rws) - 1 - i
            ax.add_patch(Rectangle((j, yy), 0.95, 0.9, color=cmap(np.clip(v, 0, 1)), lw=0))
            ax.text(
                j + 0.475,
                yy + 0.45,
                f"{v * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=FS * 6.0,
                color="white" if v > 0.6 else INK,
                weight="bold" if c == "Pooled" else "normal",
            )
    ax.plot([0, 3], [0.95] * 2, color=INK, lw=0.6)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(rws))
    ax.set_xticks([j + 0.475 for j in range(3)])
    ax.set_xticklabels([c[1] for c in cols])
    ax.set_yticks([len(rws) - 1 - i + 0.45 for i in range(len(rws))])
    ax.set_yticklabels(rws)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_xlabel(
        "% of cross-sectional association\nnot reproduced by the target trial", fontsize=FS * 5.4
    )


def ed_subgroups(ax, L):
    panel_title(ax, L, "22 subgroups, three outcomes")
    hp = pd.read_csv("hte_subgroups_pooled.csv")
    order = [
        ["Age 50-64", "Age 65-74", "Age 75+"],
        ["Men", "Women"],
        ["Education low", "Education mid", "Education high"],
        ["Wealth Q1-2", "Wealth Q3-5"],
        ["Lives alone", "Lives with others"],
        ["Rural", "Urban"],
        ["Good+ health", "Fair/poor health"],
        ["No elevated symptoms", "Elevated depressive symptoms"],
        ["Spouse online", "Spouse offline/none"],
        ["Never used before", "Used before (returner)"],
    ]
    nice = {
        "Age 50-64": "Age 50–64",
        "Age 65-74": "Age 65–74",
        "Age 75+": "Age ≥75",
        "Education low": "Low education",
        "Education mid": "Middle education",
        "Education high": "High education",
        "Wealth Q1-2": "Wealth Q1–2",
        "Wealth Q3-5": "Wealth Q3–5",
        "Good+ health": "Good health",
        "No elevated symptoms": "No depressive sx",
        "Elevated depressive symptoms": "Depressive sx",
        "Used before (returner)": "Used before",
    }
    outs = [
        ("Y_recall", "Recall", 1, 1, 0.20),
        ("Y_iadl", "IADL", 100, -1, 4.0),
        ("Y_dep", "Depr.", 100, -1, 4.0),
    ]
    ov_rows = {y: pooled(ci(sel(TTE, y=y)), sc) for y, _, sc, _, _ in outs}
    y = 0
    ticks, labs = [], []

    def dot(x, yy, est, lo, hi, sgn, vmax):
        b = sgn * est
        sig = (lo > 0) or (hi < 0)
        r = 6 + 75 * np.clip(abs(b) / vmax, 0, 1)
        col = INK if b > 0 else "#b9b7b0"
        ax.scatter(
            x, yy, s=r, facecolor=col if sig else SURF, edgecolor=col, linewidth=0.9, zorder=3
        )

    for j, (yv, nm, sc, sgn, vmax) in enumerate(outs):
        p = ov_rows[yv]
        dot(j, 0, p["est"], p["lo"], p["hi"], sgn, vmax * sc / sc if sc == 1 else vmax)
    ticks.append(0)
    labs.append("Overall")
    y = 1.4
    for items in order:
        for it in items:
            for j, (yv, nm, sc, sgn, vmax) in enumerate(outs):
                r = hp[(hp.y == yv) & (hp.subgroup == it)].iloc[0]
                dot(j, -y, r.est * sc, r.lo * sc, r.hi * sc, sgn, vmax)
            ticks.append(-y)
            labs.append(nice.get(it, it))
            y += 1
        y += 0.35
    ax.axhline(-0.7, color=OBS, lw=0.6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=FS * 5.2)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([o[1] for o in outs])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-y + 0.1, 0.8)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(
        0.5,
        -0.075,
        "area, size of benefit; filled,\n95% CI excludes 0",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FS * 5.0,
        color=INK2,
    )


def ed_spec_tte(fig, cell, L):
    ax_i = cell.subgridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.08)
    ax = fig.add_subplot(ax_i[0])
    panel_title(ax, L, "Specification curve, recall")
    SN = pd.read_csv("results_sens_overlap.csv")
    SN = SN[(SN.design == "main") & (SN.y == "Y_recall")]
    R3_ = R3[R3.y == "Y_recall"] if "y" in R3 else R3
    met = (
        lambda t: pd.read_csv(f"results_meta_{t}.csv")
        .query("y == 'Y_recall' and scale == 'RD/MD'")
        .iloc[0]
    )
    specs = []

    def add(lbl, flags, g=None, row=None):
        if g is not None:
            p = pooled(ci(g))
            specs.append((lbl, flags, p["est"], p["lo"], p["hi"]))
        else:
            specs.append((lbl, flags, row.est, row.lo, row.hi))

    add("Primary", set(), SN[SN.method == "trim"])
    add("Untrimmed", {"No trimming"}, SN[SN.method == "untrim"])
    add("Overlap", {"Overlap weights"}, SN[SN.method == "ato"])
    add("<2010", {"Baseline before 2010"}, SN[SN.method == "era<2010"])
    add(">=2010", {"Baseline from 2010"}, SN[SN.method == "era>=2010"])
    add("pre2020", {"Excluding 2020 waves"}, R3_[R3_.analysis == "tte_precovid"])
    add("alive", {"Alive at t3 only"}, R3_[R3_.analysis == "tte_known_t3"])
    add("ipcw", {"Weighted for loss"}, ATT[(ATT.analysis == "ipcw_t1") & (ATT.y == "Y_recall")])
    add("adj0", {"Baseline health only"}, row=met("hgb_trim0.025"))
    add("adj1", {"+ prior-wave health"}, row=met("hgb_trim0.025_lag"))
    add("adj3", {"+ health at t1"}, row=met("hgb_trim0.025_lag_t1_extra"))
    specs.sort(key=lambda t: t[2])
    xs = np.arange(len(specs))
    for x, (lbl, fl, e, lo, hi) in zip(xs, specs):
        col = INK if lbl != "Primary" else "#1a1a1a"
        ax.plot(
            [x, x],
            [max(lo, 0), min(hi, 0.32)],
            color=MUTED if lbl != "Primary" else INK,
            lw=1.4,
            solid_capstyle="round",
        )
        ax.plot(
            x, e, "o", color=OBS, ms=4.4 if lbl != "Primary" else 5.6, mec=SURF, mew=0.6, zorder=4
        )
        if hi > 0.32:
            ax.plot(x, 0.32, marker="^", color=MUTED, ms=3, clip_on=False)
    pr = [s for s in specs if s[0] == "Primary"][0]
    ax.axhline(pr[2], color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlim(-0.6, len(specs) - 0.4)
    ax.set_ylim(0, 0.32)
    ax.set_xticks([])
    ax.set_ylabel("Recall (SD)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    ax2 = fig.add_subplot(ax_i[1], sharex=ax)
    dims = [
        "No trimming",
        "Overlap weights",
        "Baseline before 2010",
        "Baseline from 2010",
        "Excluding 2020 waves",
        "Alive at t3 only",
        "Weighted for loss",
        "Baseline health only",
        "+ prior-wave health",
        "+ health at t1",
    ]
    for k, d in enumerate(dims):
        yy = len(dims) - 1 - k
        ax2.axhline(yy, color=GRID, lw=0.5, zorder=0)
        for x, (lbl, fl, *_r) in zip(xs, specs):
            on = d in fl
            ax2.plot(x, yy, "o", ms=3.6 if on else 2.0, color=OBS if on else "#d6d4cd", zorder=2)
    px = [x for x, s in zip(xs, specs) if s[0] == "Primary"][0]
    ax2.axvspan(px - 0.4, px + 0.4, color=GREY, alpha=0.18, lw=0)
    ax2.text(
        px, len(dims) - 0.3, "primary", ha="center", va="bottom", fontsize=FS * 5.0, color=INK2
    )
    ax2.set_yticks(range(len(dims)))
    ax2.set_yticklabels(dims[::-1], fontsize=FS * 5.0)
    ax2.set_ylim(-0.6, len(dims) + 0.2)
    ax2.set_xticks([])
    ax2.tick_params(axis="y", length=0)
    for s_ in ax2.spines.values():
        s_.set_visible(False)


# ================================================================== Fig. 3  trajectories + design signature
def _traj_small(ax, c, ylim, show_y=False):
    t = (
        TRJ[(TRJ.analysis == "tte") & (TRJ.cohort == c)]
        .pivot(index="pos", columns="grp", values=["est", "se"])
        .reindex(["t−1", "t0", "t1", "t2"])
    )
    x = np.arange(4)
    ax.axvspan(1.5, 2.5, color="#efeee9", zorder=-3, lw=0)
    for g, col, lw in [("stay", GREY, 1.1), ("start", OBS, 1.4)]:
        e, s = t[("est", g)].values, t[("se", g)].values
        ax.fill_between(x, e - 1.96 * s, e + 1.96 * s, color=col, alpha=0.15, lw=0)
        ax.plot(x, e, "-o", color=col, lw=lw, ms=2.6, mec=SURF, mew=0.3)
    gap0 = t[("est", "start")].iloc[1] - t[("est", "stay")].iloc[1]
    ax.text(
        0.03,
        0.97,
        c,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS * 6.0,
        weight="bold",
        color=COL[c],
    )
    ax.text(
        0.97,
        0.97,
        f"gap at t0 {gap0:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FS * 5.2,
        color=INK2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["t−1", "t0", "t1", "t2"], fontsize=FS * 5.2)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.3, 3.3)
    ax.set_yticks([0, 0.4, 0.8])
    if not show_y:
        ax.set_yticklabels([])
    clean(ax, "y")
    ax.tick_params(axis="y", labelsize=FS * 5.2)


def fig3():
    NO = pd.read_csv("results_newout_all.csv")
    GCOL["start"] = OBS
    GCOL["stay"] = GREY
    fig = plt.figure(figsize=(W, 7.1))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.92, 0.6, 0.95],
        hspace=0.42,
        left=0.085,
        right=0.985,
        top=0.94,
        bottom=0.07,
    )
    r1 = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.25)
    # a pooled trajectories
    ax = fig.add_subplot(r1[0])
    panel_title(ax, "a", "Recall before and after starting, pooled")
    traj(
        ax,
        "tte",
        ["start", "stay"],
        {"start": "initiators", "stay": "non-\ninitiators"},
        ["t−1", "t0", "t1", "t2"],
        {"start": "t1"},
        (-0.15, 0.45),
    )
    ax.axvspan(1.5, 2.5, color="#efeee9", zorder=-3, lw=0)
    ax.text(2, 0.432, "first use reported", ha="center", fontsize=FS * 5.4, color=INK2)
    ax.set_ylabel("Word recall, age- and sex-standardized (SD)")
    t = TRJ[(TRJ.analysis == "tte") & (TRJ.cohort == "Pooled")].pivot(
        index="pos", columns="grp", values="est"
    )
    gap = (t.start - t.stay).reindex(["t−1", "t0", "t1", "t2"])
    for i, v in enumerate(gap):
        a_, b_ = t.start.reindex(gap.index).iloc[i], t.stay.reindex(gap.index).iloc[i]
        ax.annotate(
            "",
            xy=(i, a_ - 0.02),
            xytext=(i, b_ + 0.02),
            arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.6, mutation_scale=5),
        )
        ax.text(i + 0.07, (a_ + b_) / 2, f"{v:.2f}", fontsize=FS * 5.6, color=INK2, va="center")
    from scipy.stats import gaussian_kde

    ax = fig.add_subplot(r1[1])
    panel_title(ax, "b", "Recall at t0, before anyone starts")
    D = pd.read_csv("dist_t0.csv.gz")
    xs = np.linspace(-3, 3, 300)
    for k, c in enumerate(["HRS", "ELSA", "SHARE", "CHARLS"]):
        base = (3 - k) * 1.0
        for a, col in [(0, GREY), (1, OBS)]:
            z = D[(D.cohort == c) & (D.A == a)].z.values
            if len(z) > 4000:
                z = np.random.default_rng(1).choice(z, 4000, replace=False)
            dens = gaussian_kde(z, bw_method=0.25)(xs)
            dens = dens / dens.max() * 0.85
            if a == 0:
                ax.fill_between(xs, base, base + dens, color=col, alpha=0.30, lw=0)
            ax.plot(xs, base + dens, color=col, lw=1.2 if a else 0.7)
        m0, m1 = D[(D.cohort == c) & (D.A == 0)].z.mean(), D[(D.cohort == c) & (D.A == 1)].z.mean()
        ax.plot([m0, m0], [base, base + 0.88], color=GREY, lw=0.8, ls=(0, (2, 2)))
        ax.plot([m1, m1], [base, base + 0.88], color=OBS, lw=0.9)
        ax.axhline(base, color=GRID, lw=0.5, zorder=-2)
        ax.text(-2.95, base + 0.3, c, fontsize=FS * 5.8, weight="bold", color=COL[c], va="bottom")
        ax.text(
            2.95,
            base + 0.3,
            f"Δ {m1 - m0:+.2f}".replace("-", "−"),
            fontsize=FS * 5.4,
            color=OBS,
            ha="right",
            va="bottom",
        )
    ax.set_yticks([])
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.05, 4.5)
    ax.set_xlabel("Recall at t0, adjusted for age, sex and trial (SD)")
    for s_ in ["left", "right", "top"]:
        ax.spines[s_].set_visible(False)
    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, fc=GREY, alpha=0.3, ec="none", label="stay offline"),
            Line2D([], [], color=OBS, lw=1.2, label="will start at t1"),
        ],
        frameon=False,
        fontsize=FS * 5.2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        handlelength=1.4,
        borderaxespad=0,
    )
    # f four designs x four outcomes (neutral; designs by marker)
    r2 = gs[1].subgridspec(1, 4, wspace=0.16)
    designs = [
        "Target trial (start t1)",
        "Placebo (start t3)",
        "Placebo (start t4)",
        "Before starting (t0)",
    ]
    dsty = [
        ("o", OBS, 5.2, OBS),
        ("s", INK2, 4.6, SURF),
        ("D", INK2, 4.2, SURF),
        ("^", INK2, 5.0, SURF),
    ]
    cols = [
        (
            "Word recall (SD)",
            1,
            (0, 0.32),
            [
                sel(TTE, y="Y_recall"),
                sel(PLA, y="Y_recall"),
                sel(PL2, y="Y_recall"),
                sel(PRE, y="Y_pre"),
            ],
        ),
        (
            "Composite (SD)",
            1,
            (0, 0.32),
            [NO[(NO.outcome == "gcog") & (NO.design == d)] for d in ["tte", "placebo"]]
            + [None, NO[(NO.outcome == "gcog") & (NO.design == "pretrend")]],
        ),
        (
            "New IADL difficulty (pp)",
            100,
            (-10, 6),
            [
                sel(TTE, y="Y_iadl"),
                sel(PLA, y="Y_iadl"),
                sel(PL2, y="Y_iadl"),
                sel(PRE, y="Y_iadl_pre"),
            ],
        ),
        (
            "Depressive symptoms (pp)",
            100,
            (-10, 6),
            [sel(TTE, y="Y_dep"), sel(PLA, y="Y_dep"), sel(PL2, y="Y_dep"), None],
        ),
    ]
    for j, (nm, sc, lim, gl) in enumerate(cols):
        ax = fig.add_subplot(r2[j])
        if j == 0:
            panel_title(ax, "c", "Four designs, four outcomes")
        for i, g in enumerate(gl):
            yy = 3 - i
            mk, lc, ms, fc = dsty[i]
            ax.axhline(yy - 0.5, color=GRID, lw=0.5, zorder=-2)
            if g is None or not len(g):
                ax.text(
                    np.mean(lim), yy, "–", ha="center", va="center", fontsize=FS * 6, color=MUTED
                )
                continue
            gg = ci(g)
            for c in COH:
                q = gg[gg.cohort == c]
                if len(q):
                    ax.plot(
                        q.est.iloc[0] * sc,
                        yy + 0.27,
                        MK[c],
                        color=COL[c],
                        ms=2.6,
                        mec=SURF,
                        mew=0.3,
                    )
            p = pooled(gg, sc)
            ax.plot(
                [max(p["lo"], lim[0]), min(p["hi"], lim[1])],
                [yy] * 2,
                color=lc,
                lw=1.3 if i == 0 else 1.0,
                solid_capstyle="round",
            )
            ax.plot(p["est"], yy, mk, color=lc, mfc=fc if i else OBS, ms=ms, mew=1.1, zorder=5)
            ax.text(
                min(p["hi"], lim[1]) + (lim[1] - lim[0]) * 0.025,
                yy,
                (f"{p['est']:.2f}" if sc == 1 else f"{p['est']:.1f}").replace("-", "−"),
                va="center",
                ha="left",
                fontsize=FS * 5.0,
                color=lc,
                clip_on=False,
            )
            if p["lo"] < lim[0]:
                ax.plot(lim[0], yy, "<", color=lc, ms=3, clip_on=False)
            if p["hi"] > lim[1]:
                ax.plot(lim[1], yy, ">", color=lc, ms=3, clip_on=False)
        ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(*lim)
        ax.set_ylim(-0.5, 3.6)
        ax.set_xlabel(nm, fontsize=FS * 5.8)
        ax.set_yticks(range(4))
        ax.set_yticklabels(designs[::-1] if j == 0 else [])
        clean(ax)
        ax.grid(False)
    # g (vertical) target − placebo, h scatter, i distributions
    r3 = gs[2].subgridspec(1, 2, width_ratios=[0.62, 1.0], wspace=0.3)
    ax = fig.add_subplot(r3[0])
    panel_title(ax, "d", "Target trial − placebo")
    d1 = ci(sel(DIF, y="Y_recall"))
    ax.axhspan(-0.1, 0.1, color="#efeee9", zorder=-3, lw=0)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    for i, c in enumerate(C4):
        r = d1[d1.cohort == c]
        if len(r):
            ax.errorbar(
                i,
                r.est.iloc[0],
                yerr=[[r.est.iloc[0] - r.lo.iloc[0]], [r.hi.iloc[0] - r.est.iloc[0]]],
                fmt=MK[c],
                color=COL[c],
                ms=4.4,
                elinewidth=1.0,
                capsize=0,
            )
    p = pooled(d1)
    ax.fill([4 - 0.25, 4, 4 + 0.25, 4], [p["est"], p["hi"], p["est"], p["lo"]], color=OBS)
    ax.text(
        4,
        p["hi"] + 0.02,
        f"{p['est']:+.3f}".replace("-", "−"),
        ha="center",
        va="bottom",
        fontsize=FS * 5.6,
        weight="bold",
    )
    ax.set_xticks(range(5))
    ax.set_xticklabels(
        ["HRS", "ELSA", "SHARE", "CHARLS", "Pooled"],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=FS * 5.6,
    )
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([-0.2, 0, 0.2])
    ax.set_ylabel("Recall difference (SD)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    ax.text(
        0.03,
        0.97,
        "band ±0.10 SD",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS * 5.0,
        color=INK2,
    )
    ax = fig.add_subplot(r3[1])
    panel_title(ax, "e", "Target trial vs placebo")
    pairs = []
    for o, g_t, g_p in [("recall", sel(TTE, y="Y_recall"), sel(PLA, y="Y_recall"))] + [
        (
            o,
            NO[(NO.outcome == o) & (NO.design == "tte")],
            NO[(NO.outcome == o) & (NO.design == "placebo")],
        )
        for o in ["ser7", "verbf", "gcog"]
    ]:
        g_t, g_p = ci(g_t).set_index("cohort"), ci(g_p).set_index("cohort")
        for c in C4:
            if c in g_t.index and c in g_p.index:
                pairs.append((c, o, g_t.loc[c], g_p.loc[c]))
    ax.plot([-0.1, 0.4], [-0.1, 0.4], color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for c, o, a_, b_ in pairs:
        ax.plot([a_.lo, a_.hi], [b_.est] * 2, color="#d6d4cd", lw=0.6, zorder=2)
        ax.plot([a_.est] * 2, [b_.lo, b_.hi], color="#d6d4cd", lw=0.6, zorder=2)
    for c, o, a_, b_ in pairs:
        ax.plot(a_.est, b_.est, MK[c], color=COL[c], ms=4.6, mec=SURF, mew=0.5, zorder=4)
    xa = np.array([q[2].est for q in pairs])
    yb = np.array([q[3].est for q in pairs])
    ax.text(
        0.03,
        0.97,
        f"{len(pairs)} cohort × test pairs\nmedian ratio {np.median(yb / xa):.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=FS * 5.3,
        color=INK2,
    )
    ax.text(
        0.33, 0.355, "y = x", rotation=45, fontsize=FS * 5.3, color=INK2, ha="center", va="bottom"
    )
    ax.set_xlim(-0.1, 0.4)
    ax.set_ylim(-0.1, 0.4)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("Target trial (SD)")
    ax.set_ylabel("Placebo (SD)")
    clean(ax, None)
    ax.grid(color=GRID, lw=0.5)
    cohort_legend(fig, y=0.968)
    save(fig, "NH3_Fig3")


def ed_cohort_traj(fig, cells, letters):
    GCOL["start"] = OBS
    GCOL["stay"] = GREY
    for k, (c, L) in enumerate(zip(["HRS", "ELSA", "SHARE", "CHARLS"], letters)):
        ax = fig.add_subplot(cells[k])
        _traj_small(ax, c, (-0.3, 0.95), show_y=(k == 0))
        lab(ax, L, x=-0.02, y=1.02)
        if k == 0:
            ax.set_ylabel("Recall (SD)", fontsize=FS * 5.6, labelpad=2)


# ================================================================== Fig. 4  timing of adoption
def _pool90(g, sc=1.0):
    g = ci(g)
    g = g[g.cohort.isin(COH)]
    se = g.se.values if "se" in g and g.se.notna().all() else ((g.hi - g.lo) / 3.92).values
    if len(g) == 1:
        e = g.est.iloc[0]
        s = se[0]
        return dict(
            est=e * sc,
            lo=(e - 1.96 * s) * sc,
            hi=(e + 1.96 * s) * sc,
            lo90=(e - 1.645 * s) * sc,
            hi90=(e + 1.645 * s) * sc,
            k=1,
        )
    r = hksj(g.est.values, se)
    if r["k"] >= 3:
        lo, hi, l9, h9 = r["hk_lo"], r["hk_hi"], r["hk90_lo"], r["hk90_hi"]
    else:
        lo, hi = r["dl_lo"], r["dl_hi"]
        s = (hi - lo) / 3.92
        l9, h9 = r["est"] - 1.645 * s, r["est"] + 1.645 * s
    return dict(est=r["est"] * sc, lo=lo * sc, hi=hi * sc, lo90=l9 * sc, hi90=h9 * sc, k=r["k"])


def _equiv(ax, rows, margin, xlim, unit_fmt):
    n = len(rows)
    ax.axvspan(-margin, margin, color="#efeee9", zorder=-3, lw=0)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    for i, (lab_, g, sc) in enumerate(rows):
        yy = n - 1 - i
        if g is None or not len(g):
            continue
        p = _pool90(g, sc)
        ax.plot([max(p["lo"], xlim[0]), min(p["hi"], xlim[1])], [yy] * 2, color=MUTED, lw=0.9)
        ax.plot(
            [max(p["lo90"], xlim[0]), min(p["hi90"], xlim[1])],
            [yy] * 2,
            color=OBS,
            lw=2.4,
            solid_capstyle="butt",
        )
        eq = (p["lo90"] > -margin) and (p["hi90"] < margin)
        ax.plot(p["est"], yy, "o", color=OBS if eq else SURF, mec=OBS, ms=4.6, mew=1.0, zorder=5)
        if p["hi"] > xlim[1]:
            ax.plot(xlim[1], yy, ">", color=MUTED, ms=3, clip_on=False)
        if p["lo"] < xlim[0]:
            ax.plot(xlim[0], yy, "<", color=MUTED, ms=3, clip_on=False)
        ax.text(
            1.03,
            yy,
            unit_fmt.format(p["est"]).replace("-", "−"),
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=FS * 5.4,
            color=OBS,
        )
        ax.text(
            1.03,
            yy - 0.34,
            "equivalent" if eq else "inconclusive",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=FS * 4.8,
            color=INK2 if eq else MUTED,
            style="italic",
        )
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(*xlim)
    clean(ax)
    ax.grid(False)


def _fig4_setup():
    NO = pd.read_csv("results_newout_all.csv")
    DM = pd.read_csv("results_dementia_HRS.csv")
    W1 = lambda ct, y: WIT[(WIT.contrast == ct) & (WIT.y == y)]
    SEQ = {"early": OBS, "middle": OBS2, "late": OBS3, "never": "#bdbbb4"}
    DOA = pd.read_csv("results_dose_lag.csv")
    DOA["dose"] = DOA.dose.astype(str)
    DO = DOA[DOA.adj == "full"]
    return NO, DM, W1, SEQ, DOA, DO


def fig4():
    NO, DM, W1, SEQ, DOA, DO = _fig4_setup()
    fig = plt.figure(figsize=(W, 5.9))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1, 1],
        width_ratios=[1.05, 0.8, 1.05],
        hspace=0.5,
        wspace=0.95,
        left=0.085,
        right=0.93,
        top=0.935,
        bottom=0.08,
    )
    G = {"a": gs[0, 0], "b": gs[0, 1], "c": gs[:, 2], "d": gs[1, 0], "e": gs[1, 1]}
    # a trajectories by timing (sequential greys)
    ax = fig.add_subplot(G["a"])
    panel_title(ax, "a", "Recall by timing of first use")
    t = TRJ[(TRJ.analysis == "timing") & (TRJ.cohort == "Pooled")]
    pos = ["t−1", "t0", "t1", "t2", "t3"]
    X = {p: i for i, p in enumerate(pos)}
    starts = {"early": "t1", "middle": "t2", "late": "t3"}
    for g in ["never", "late", "middle", "early"]:
        s = t[t.grp == g].set_index("pos").reindex(pos)
        ax.fill_between(
            range(5), s.est - 1.96 * s.se, s.est + 1.96 * s.se, color=SEQ[g], alpha=0.15, lw=0
        )
        ax.plot(
            range(5),
            s.est,
            "-o",
            color=SEQ[g],
            lw=1.4 if g != "never" else 1.1,
            ls="-" if g != "never" else (0, (3, 2)),
            ms=3,
            mec=SURF,
            mew=0.4,
        )
        if g in starts:
            p = starts[g]
            ax.plot(X[p], s.est[p] + 0.07, "v", color=SEQ[g], ms=4.5)
        lab_ = {
            "early": "early (t1)",
            "middle": "middle (t2)",
            "late": "late (t3)",
            "never": "never",
        }[g]
        ax.text(
            4.15,
            s.est.iloc[-1] + {"early": 0.05, "middle": 0.0, "late": -0.05, "never": 0}[g],
            lab_,
            va="center",
            fontsize=FS * 5.4,
            color=SEQ[g] if g != "never" else MUTED,
            weight="bold",
        )
    ax.axvline(3, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.text(3, 0.66, "outcome", ha="center", fontsize=FS * 5.2, color=INK2)
    ax.set_xticks(range(5))
    ax.set_xticklabels(pos)
    ax.set_xlim(-0.2, 5.4)
    ax.set_ylim(-0.3, 0.7)
    ax.set_ylabel("Word recall (SD)")
    ax.set_xlabel("Survey wave")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    # b observed vs expected under an effect
    ax = fig.add_subplot(G["b"])
    panel_title(ax, "b", "Gain over never-starters")
    grp = [
        ("Early\n(t1)", "early vs never"),
        ("Middle\n(t2)", "middle vs never"),
        ("Late\n(t3)", "late vs never"),
    ]
    obs = [pooled(ci(W1(ct, "Y_recall"))) for _, ct in grp]
    late = obs[2]["est"]
    exp_ = [late + 0.10, late + 0.05, late]
    ax.plot(range(3), exp_, color=EFF, lw=1.0, ls=(0, (3, 2)), marker="_", ms=10, mew=1.2)
    ax.text(
        2.55,
        0.395,
        "dashed: illustrative\npattern if starting\nhelped",
        ha="right",
        va="top",
        fontsize=FS * 4.8,
        color=EFF,
    )
    for i, p in enumerate(obs):
        ax.plot([i, i], [p["lo"], p["hi"]], color=OBS, lw=1.3)
        ax.plot(i, p["est"], "o", color=OBS, ms=5.2, mec=SURF, mew=0.6, zorder=5)
        ax.text(
            i - 0.12,
            p["est"] - 0.018,
            f"{p['est']:.2f}",
            va="center",
            ha="right",
            fontsize=FS * 5.4,
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in grp], fontsize=FS * 5.6)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(0, 0.4)
    ax.set_ylabel("Recall vs never (SD)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    # d standardized equivalence plot, e cohort x outcome matrix
    ne = lambda o: NO[(NO.outcome == o) & (NO.design == "early_vs_late")]
    OUTS4 = [
        (
            "Cognition",
            [
                ("Word recall", W1("early vs late", "Y_recall"), 1, 0.10, 1, "{:+.2f} SD"),
                ("Serial 7s", ne("ser7"), 1, 0.10, 1, "{:+.2f} SD"),
                ("Verbal fluency", ne("verbf"), 1, 0.10, 1, "{:+.2f} SD"),
                ("Composite", ne("gcog"), 1, 0.10, 1, "{:+.2f} SD"),
                ("Life satisfaction", ne("lsat"), 1, 0.10, 1, "{:+.2f} SD"),
            ],
        ),
        (
            "Function and health",
            [
                ("IADL difficulty", W1("early vs late", "Y_iadl"), 100, 2.0, -1, "{:+.1f} pp"),
                ("Depressive sx", W1("early vs late", "Y_dep"), 100, 2.0, -1, "{:+.1f} pp"),
                ("Fair/poor health", ne("srh"), 100, 2.0, -1, "{:+.1f} pp"),
                ("Impairment (HRS)", ne("cimp"), 100, 2.0, -1, "{:+.1f} pp"),
                (
                    "Dementia (HRS)",
                    DM[(DM.outcome == "dem") & (DM.design == "early_vs_late")],
                    100,
                    2.0,
                    -1,
                    "{:+.1f} pp",
                ),
            ],
        ),
    ]
    ax = fig.add_subplot(G["c"])
    panel_title(ax, "c", "Early vs late at t2, against the margin")
    lim = (-2.2, 2.2)
    ax.axvspan(-1, 1, color="#efeee9", zorder=-3, lw=0)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    y = 0
    ticks, labs = [], []
    for gname, items in OUTS4:
        ax.text(
            lim[0] - 0.05,
            -y + 0.05,
            gname,
            transform=ax.transData,
            ha="right",
            va="center",
            fontsize=FS * 5.4,
            weight="bold",
            color=OBS,
        )
        y += 0.9
        for nm, g, sc, mg, sgn, fmt in items:
            p = _pool90(g, sc)
            f_ = lambda v: v / mg
            yy = -y
            ax.plot(
                [max(f_(p["lo"]), lim[0]), min(f_(p["hi"]), lim[1])], [yy] * 2, color=MUTED, lw=0.9
            )
            ax.plot(
                [max(f_(p["lo90"]), lim[0]), min(f_(p["hi90"]), lim[1])],
                [yy] * 2,
                color=OBS,
                lw=2.4,
                solid_capstyle="butt",
            )
            eq = (p["lo90"] > -mg) and (p["hi90"] < mg)
            ax.plot(
                f_(p["est"]), yy, "o", color=OBS if eq else SURF, mec=OBS, ms=4.6, mew=1.0, zorder=5
            )
            if f_(p["hi"]) > lim[1]:
                ax.plot(lim[1], yy, ">", color=MUTED, ms=3, clip_on=False)
            if f_(p["lo"]) < lim[0]:
                ax.plot(lim[0], yy, "<", color=MUTED, ms=3, clip_on=False)
            ax.text(
                lim[1] + 0.08,
                yy,
                fmt.format(p["est"]).replace("-", "−"),
                va="center",
                fontsize=FS * 5.2,
                color=OBS,
            )
            ticks.append(yy)
            labs.append(nm)
            y += 1
        y += 0.35
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=FS * 5.6)
    ax.set_ylim(-y + 0.3, 0.3)
    ax.set_xlim(*lim)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(["−2", "−1", "0", "+1", "+2"])
    ax.set_xlabel("Early − late, in units of the margin (±0.10 SD; ±2 points)")
    clean(ax)
    ax.grid(False)
    # i dose gradient before any use
    ax = fig.add_subplot(G["d"])
    panel_title(ax, "d", "Ahead before starting")
    lv = DO[(DO.outcome == "level") & (DO.dose != "slope")]
    for j, c in enumerate(C4):
        q = lv[lv.cohort == c]
        xx = q.dose.astype(int).values + (j - 1.5) * 0.09
        ax.plot(np.r_[0 + (j - 1.5) * 0.09, xx], np.r_[0, q.est], color=COL[c], lw=0.6, alpha=0.6)
        ax.errorbar(
            xx,
            q.est,
            yerr=1.96 * q.se,
            fmt=MK[c],
            color=COL[c],
            ms=2.4,
            lw=0.6,
            capsize=0,
            alpha=0.85,
        )
    lb_ = DOA[(DOA.adj == "basic") & (DOA.outcome == "level")]
    PB = [0] + [_pool90(lb_[lb_.dose == str(d_)])["est"] for d_ in (1, 2)]
    ax.plot(range(3), PB, "--o", color="#8a8880", lw=1.1, ms=3.4, mfc=SURF, mew=0.9, zorder=4)
    P = [dict(est=0, lo=0, hi=0)] + [_pool90(lv[lv.dose == str(d_)]) for d_ in (1, 2)]
    for d_ in (1, 2):
        ax.plot([d_, d_], [P[d_]["lo"], P[d_]["hi"]], color=OBS, lw=1.0, zorder=5)
    ax.plot(range(3), [p["est"] for p in P], "-", color=OBS, lw=1.8, zorder=5)
    ax.plot(range(3), [p["est"] for p in P], "D", color=OBS, ms=4.2, mec=SURF, mew=0.6, zorder=6)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=OBS,
                lw=1.8,
                marker="D",
                mfc=OBS,
                mec=SURF,
                ms=4,
                label="fully adjusted (pooled)",
            ),
            Line2D(
                [],
                [],
                color="#8a8880",
                lw=1.1,
                ls="--",
                marker="o",
                mfc=SURF,
                ms=3.4,
                label="age, sex only (pooled)",
            ),
        ],
        loc="upper left",
        frameon=False,
        fontsize=FS * 4.8,
        handlelength=1.8,
        borderaxespad=0.2,
    )
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(3))
    ax.set_xlim(-0.3, 2.35)
    ax.set_ylim(-0.08, 0.95)
    ax.set_xlabel("Waves online at t1–t2")
    ax.set_ylabel("Recall at t0 vs 0 waves (SD)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    # j change per wave before, while and after the dose is accrued (dose = waves online at t1-t2)
    ax = fig.add_subplot(G["e"])
    panel_title(ax, "e", "Change, by period")
    ax.axhspan(-0.1, 0.1, color="#efeee9", zorder=-3, lw=0)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    per = [
        ("pre", "#8a8880", SURF, -0.24, "before (t−1 to t0)"),
        ("during", OBS3, OBS3, 0.0, "while online (t0 to t2)"),
        ("after", OBS, OBS, 0.24, "after (t2 to t3)"),
    ]
    for v, col, fc, off, lab_ in per:
        for d_ in (1, 2):
            p = _pool90(DO[(DO.outcome == v) & (DO.dose == str(d_)) & (DO.se > 0)])
            ax.plot([d_ + off] * 2, [p["lo"], p["hi"]], color=col, lw=0.9)
            ax.plot(
                [d_ + off] * 2, [p["lo90"], p["hi90"]], color=col, lw=2.2, solid_capstyle="butt"
            )
            ax.plot(d_ + off, p["est"], "o", color=fc, mec=col, ms=4.2, mew=1.0, zorder=5)
    ax.legend(
        handles=[
            Line2D([], [], color=c_, marker="o", mfc=f_, mec=c_, lw=2, ms=4, label=l_)
            for _, c_, f_, _, l_ in per
        ],
        loc="upper left",
        frameon=False,
        fontsize=FS * 4.6,
        handlelength=1.2,
        borderaxespad=0.1,
        labelspacing=0.25,
    )
    ax.set_xticks([1, 2])
    ax.set_xlim(0.55, 2.45)
    ax.set_ylim(-0.16, 0.26)
    ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1, 0.15])
    ax.set_xlabel("Waves online at t1–t2")
    ax.set_ylabel("vs 0 waves, SD per wave")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    cohort_legend(fig, y=0.965)
    save(fig, "NH3_Fig4")


def ed3():
    NO, DM, W1, SEQ, DOA, DO = _fig4_setup()
    fig = plt.figure(figsize=(W, 6.6))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.05, 0.95], hspace=0.5, left=0.12, right=0.985, top=0.94, bottom=0.07
    )
    r1 = gs[0].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.75)
    r2 = gs[1].subgridspec(1, 3, width_ratios=[1.25, 1.0, 0.65], wspace=0.6)
    G = {"a": r1[0], "b": r1[1], "c": r2[0], "d": r2[1], "e": r2[2]}
    # c alluvial: adoption and persistence among people who went online by t3
    ax = fig.add_subplot(G["a"])
    panel_title(ax, "a", "Adopters: going online and staying")
    FL = pd.read_csv("flows_t0.csv")
    N0 = FL.n.sum()

    def cls(r):
        if r.s1 == "online":
            return "early"
        if r.s2 == "online":
            return "middle"
        if r.s3 == "online":
            return "late"
        return None

    FL["cls"] = FL.apply(cls, axis=1)
    FL = FL[FL.cls.notna()].copy()
    corder = {"early": 0, "middle": 1, "late": 2}
    sorder = {"online": 0, "offline": 1, "unknown": 2}
    ccol = {"early": SEQ["early"], "middle": SEQ["middle"], "late": SEQ["late"]}
    N = FL.n.sum()
    gap = 0.05
    colw = 0.07
    FL["k"] = FL.cls.map(corder) * 100 + FL.s1.map(sorder) * 10 + FL.s2.map(sorder)
    FL = FL.sort_values("k").reset_index(drop=True)
    cols = ["s1", "s2", "s3"]
    pos = {}
    for j, cname in enumerate(cols):
        y = 1.0
        for st in ["online", "offline", "unknown"]:
            sub = FL[FL[cname] == st]
            if not len(sub):
                continue
            top = y
            for i, r in sub.iterrows():
                h = r.n / N
                pos[(j, i)] = (y, y - h)
                y -= h
            ax.add_patch(
                Rectangle((j - colw / 2, y), colw, top - y, fc=SURF, ec=INK, lw=0.7, zorder=3)
            )
            lab_ = {"online": "online", "offline": "offline", "unknown": "not seen"}[st]
            ax.text(
                j + colw / 2 + 0.03,
                (top + y) / 2,
                f"{lab_}\n{100 * (top - y):.0f}%",
                ha="left",
                va="center",
                fontsize=FS * 4.6,
                color=OBS,
                zorder=4,
                bbox=dict(fc=SURF, ec="none", pad=0.4, alpha=0.95),
            )
            y -= gap
    tt = np.linspace(0, 1, 30)
    sm_ = 3 * tt**2 - 2 * tt**3
    for i, r in FL.iterrows():
        for j in range(2):
            (a0, a1), (b0, b1) = pos[(j, i)], pos[(j + 1, i)]
            x = j + colw / 2 + (1 - colw) * tt
            ax.fill_between(
                x,
                a1 + (b1 - a1) * sm_,
                a0 + (b0 - a0) * sm_,
                color=ccol[r.cls],
                alpha=0.75,
                lw=0,
                zorder=1,
            )
    ax.set_xlim(-0.15, 2.55)
    ax.set_ylim(-2 * gap - 0.03, 1.02)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["t1", "t2", "t3"])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    [s_.set_visible(False) for s_ in ax.spines.values()]
    h = [
        Rectangle((0, 0), 1, 1, fc=ccol[k], alpha=0.75, ec="none", label=f"{k} (t{corder[k] + 1})")
        for k in ["early", "middle", "late"]
    ]
    lg = ax.legend(
        handles=h,
        frameon=False,
        fontsize=FS * 4.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=3,
        handlelength=0.9,
        columnspacing=0.6,
        title=f"{N:,} of {N0:,} people offline at t0 went online by t3",
        title_fontsize=FS * 4.8,
    )
    lg._legend_box.align = "center"
    # d standardized equivalence plot, e cohort x outcome matrix
    ne = lambda o: NO[(NO.outcome == o) & (NO.design == "early_vs_late")]
    OUTS4 = [
        (
            "Cognition",
            [
                ("Word recall", W1("early vs late", "Y_recall"), 1, 0.10, 1, "{:+.2f} SD"),
                ("Serial 7s", ne("ser7"), 1, 0.10, 1, "{:+.2f} SD"),
                ("Verbal fluency", ne("verbf"), 1, 0.10, 1, "{:+.2f} SD"),
                ("Composite", ne("gcog"), 1, 0.10, 1, "{:+.2f} SD"),
                ("Life satisfaction", ne("lsat"), 1, 0.10, 1, "{:+.2f} SD"),
            ],
        ),
        (
            "Function and health",
            [
                ("IADL difficulty", W1("early vs late", "Y_iadl"), 100, 2.0, -1, "{:+.1f} pp"),
                ("Depressive sx", W1("early vs late", "Y_dep"), 100, 2.0, -1, "{:+.1f} pp"),
                ("Fair/poor health", ne("srh"), 100, 2.0, -1, "{:+.1f} pp"),
                ("Impairment (HRS)", ne("cimp"), 100, 2.0, -1, "{:+.1f} pp"),
                (
                    "Dementia (HRS)",
                    DM[(DM.outcome == "dem") & (DM.design == "early_vs_late")],
                    100,
                    2.0,
                    -1,
                    "{:+.1f} pp",
                ),
            ],
        ),
    ]
    # e cohort x outcome matrix
    ax = fig.add_subplot(G["b"])
    panel_title(ax, "b", "Early vs late by cohort")
    y = 0
    ticks = []
    ylabs = []
    for gname, items in OUTS4:
        y += 0.9
        for nm, g, sc, mg, sgn, fmt in items:
            gg = ci(g)
            for j, c in enumerate(C4):
                q = gg[gg.cohort == c]
                if not len(q):
                    ax.plot(j, -y, "o", ms=2, color="#d6d4cd")
                    continue
                q = q.iloc[0]
                v = sgn * q.est * sc / mg
                sig = (q.lo > 0) or (q.hi < 0)
                col = OBS if v > 0 else "#a3a19b"
                ax.scatter(
                    j,
                    -y,
                    s=8 + 70 * min(abs(v), 2) / 2,
                    facecolor=col if sig else SURF,
                    edgecolor=col,
                    linewidth=0.9,
                    zorder=3,
                )
            ticks.append(-y)
            ylabs.append(nm)
            y += 1
        y += 0.35
    ax.set_yticks(ticks)
    ax.set_yticklabels(ylabs, fontsize=FS * 5.4)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(C4, fontsize=FS * 5.2, rotation=40, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    for j, c in enumerate(C4):
        ax.get_xticklabels()[j].set_color(COL[c])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-y + 0.3, 0.3)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for t_ in ticks:
        ax.axhline(t_, color=GRID, lw=0.4, zorder=0)
    ax.text(
        0.5,
        -0.17,
        "area, |difference| / margin\nteal, early better; grey, late\nfilled, 95% CI excludes 0",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FS * 4.8,
        color=INK2,
    )
    # f specification curve, g sustained trajectory, h sustained vs stopped, j change per wave
    sp = G["c"].subgridspec(2, 1, height_ratios=[1.0, 1.1], hspace=0.08)
    ax = fig.add_subplot(sp[0])
    panel_title(ax, "c", "Specification curve, early vs late")
    s = lambda m: SENS[
        (SENS.design == "early_vs_late") & (SENS.y == "Y_recall") & (SENS.method == m)
    ]
    specs = [
        ("Primary", set(), s("trim")),
        ("u", {"No trimming"}, s("untrim")),
        ("o", {"Overlap weights"}, s("ato")),
        ("e1", {"Baseline before 2010"}, s("era<2010")),
        ("e2", {"Baseline from 2010"}, s("era>=2010")),
        ("c", {"Excluding 2020 waves"}, sel(R3, analysis="early_vs_late_precovid", y="Y_recall")),
        (
            "s",
            {"Without SHARE"},
            sel(WIT, contrast="early vs late", y="Y_recall").query("cohort!='SHARE'"),
        ),
        ("n", {"Never used before t0"}, sel(W2, variant="never_before", y="Y_recall")),
        ("t", {"No life transitions"}, sel(W2, variant="no_transitions", y="Y_recall")),
        ("v", {"Late = start at t4"}, sel(R3, analysis="early_vs_verylate", y="Y_recall")),
    ]
    specs = [(a, b, _pool90(g)) for a, b, g in specs]
    specs.sort(key=lambda z: z[2]["est"])
    ax.axhspan(-0.1, 0.1, color="#efeee9", zorder=-3, lw=0)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    for x, (lb, fl, p) in enumerate(specs):
        ax.plot([x, x], [p["lo"], p["hi"]], color=MUTED, lw=0.9)
        ax.plot([x, x], [p["lo90"], p["hi90"]], color=OBS, lw=2.2, solid_capstyle="butt")
        ax.plot(
            x,
            p["est"],
            "o",
            color=OBS if lb != "Primary" else SURF,
            mec=OBS,
            ms=4.2 if lb != "Primary" else 5.4,
            mew=1.0,
            zorder=5,
        )
    ax.set_xlim(-0.6, len(specs) - 0.4)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xticks([])
    ax.set_yticks([-0.1, 0, 0.1])
    ax.set_ylabel("Recall (SD)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    ax2 = fig.add_subplot(sp[1], sharex=ax)
    dims = [
        "No trimming",
        "Overlap weights",
        "Baseline before 2010",
        "Baseline from 2010",
        "Excluding 2020 waves",
        "Without SHARE",
        "Never used before t0",
        "No life transitions",
        "Late = start at t4",
    ]
    for k_, d in enumerate(dims):
        yy = len(dims) - 1 - k_
        ax2.axhline(yy, color=GRID, lw=0.5, zorder=0)
        for x, (lb, fl, p) in enumerate(specs):
            ax2.plot(x, yy, "o", ms=3.4 if d in fl else 1.8, color=OBS if d in fl else "#d6d4cd")
    px = [x for x, z in enumerate(specs) if z[0] == "Primary"][0]
    ax2.axvspan(px - 0.4, px + 0.4, color=GREY, alpha=0.18, lw=0)
    ax2.text(
        px, len(dims) - 0.4, "primary", ha="center", va="bottom", fontsize=FS * 4.8, color=INK2
    )
    ax2.set_yticks(range(len(dims)))
    ax2.set_yticklabels(dims[::-1], fontsize=FS * 4.9)
    ax2.set_ylim(-0.6, len(dims) + 0.2)
    ax2.set_xticks([])
    ax2.tick_params(axis="y", length=0)
    for s_ in ax2.spines.values():
        s_.set_visible(False)
    # g sustained users vs never users
    ax = fig.add_subplot(G["d"])
    panel_title(ax, "d", "Sustained vs never users")
    ST = pd.read_csv("results_sustained_traj.csv")
    SP = pd.read_csv("results_sustained_traj_pooled.csv")
    pos2 = ["t-1", "t0", "t1", "t2", "t3"]
    xs = np.arange(5)
    ax.axvspan(1.5, 3.5, color="#f4f3ef", zorder=-3, lw=0)
    ax.text(2.5, 0.03, "online at\nt1 and t2", ha="center", fontsize=FS * 5.0, color=INK2)
    for j, c in enumerate(["HRS", "ELSA", "SHARE"]):
        v = [ST[(ST.cohort == c) & (ST.term == "traj:" + p_)].est.iloc[0] for p_ in pos2]
        ax.plot(xs + (j - 1) * 0.06, v, color=COL[c], lw=0.7, alpha=0.7, marker=MK[c], ms=2.2)
    pp = [SP[SP.term == "traj:" + p_].iloc[0] for p_ in pos2]
    ax.fill_between(xs, [r.lo for r in pp], [r.hi for r in pp], color=OBS, alpha=0.13, lw=0)
    ax.plot(xs, [r.est for r in pp], color=OBS, lw=1.4, marker="o", ms=3, mec=SURF, mew=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(["t−1", "t0", "t1", "t2", "t3"])
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Recall difference (SD)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    # h sustained vs stopped (each vs late adopters)
    ax = fig.add_subplot(G["e"])
    panel_title(ax, "e", "Kept vs stopped")
    rows = [
        ("Kept\nusing", _pool90(sel(W2, variant="sustained", y="Y_recall"))),
        ("Stopped\nby t2", _pool90(sel(W3, variant="stopped_by_t2", y="Y_recall"))),
    ]
    ax.axhspan(-0.1, 0.1, color="#efeee9", zorder=-3, lw=0)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    for i, (nm, p) in enumerate(rows):
        ax.plot([i, i], [p["lo"], p["hi"]], color=MUTED, lw=0.9)
        ax.plot([i, i], [p["lo90"], p["hi90"]], color=OBS, lw=2.2, solid_capstyle="butt")
        ax.plot(i, p["est"], "o", color=OBS, ms=4.6, mec=SURF, mew=0.5, zorder=5)
    ax.plot(range(2), [r[1]["est"] for r in rows], color=OBS, lw=0.8)
    for i, (nm, p) in enumerate(rows):
        ax.text(
            i + 0.12, p["est"], f"{p['est']:+.3f}".replace("-", "−"), va="center", fontsize=FS * 5.2
        )
    ax.set_xticks(range(2))
    ax.set_xticklabels([r[0] for r in rows], fontsize=FS * 5.4)
    ax.set_xlim(-0.5, 1.7)
    ax.set_ylim(-0.2, 0.22)
    ax.set_ylabel("vs late adopters (SD)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    cohort_legend(fig, y=0.965)
    save(fig, "NH3_EDFig3")


# ================================================================== Fig. 5  signature matrix + event study
def signature_rows():
    NO = pd.read_csv("results_newout_all.csv")
    DM = pd.read_csv("results_dementia_HRS.csv")
    BV = pd.read_csv("results_behav_vig_all.csv")
    W_ = WIT[(WIT.contrast == "early vs late")]
    rows = []

    def r(label, beh, unit, sc, tte, pla, pre, evl):
        rows.append(
            dict(
                label=label,
                beh=beh,
                unit=unit,
                **{
                    k: pooled(ci(v), sc) if v is not None and len(v) else None
                    for k, v in [("tte", tte), ("pla", pla), ("pre", pre), ("evl", evl)]
                },
            )
        )

    r(
        "Word recall",
        "Internet",
        "SD",
        1,
        sel(TTE, y="Y_recall"),
        sel(PLA, y="Y_recall"),
        sel(PRE, y="Y_pre"),
        sel(W_, y="Y_recall"),
    )
    for o, nm in [
        ("ser7", "Serial 7s"),
        ("verbf", "Verbal fluency"),
        ("gcog", "Two-domain composite"),
    ]:
        r(
            nm,
            "Internet",
            "SD",
            1,
            *[
                NO[(NO.outcome == o) & (NO.design == d)]
                for d in ["tte", "placebo", "pretrend", "early_vs_late"]
            ],
        )
    r(
        "Cognitive impairment (HRS)",
        "Internet",
        "pp",
        100,
        *[
            NO[(NO.outcome == "cimp") & (NO.design == d)]
            for d in ["tte", "placebo", "pretrend", "early_vs_late"]
        ],
    )
    r(
        "Dementia (HRS)",
        "Internet",
        "pp",
        100,
        *[
            DM[(DM.outcome == "dem") & (DM.design == d)]
            for d in ["tte", "placebo", "pretrend", "early_vs_late"]
        ],
    )
    r(
        "IADL difficulty",
        "Internet",
        "pp",
        100,
        sel(TTE, y="Y_iadl"),
        sel(PLA, y="Y_iadl"),
        sel(PRE, y="Y_iadl_pre"),
        sel(W_, y="Y_iadl"),
    )
    r(
        "Fair/poor self-rated health",
        "Internet",
        "pp",
        100,
        *[
            NO[(NO.outcome == "srh") & (NO.design == d)]
            for d in ["tte", "placebo", "pretrend", "early_vs_late"]
        ],
    )
    r(
        "Word recall",
        "Physical activity",
        "SD",
        1,
        *[
            BV[(BV.outcome == "recall") & (BV.design == d)]
            for d in ["tte", "placebo", "pretrend", "early_vs_late"]
        ],
    )
    r(
        "Two-domain composite",
        "Physical activity",
        "SD",
        1,
        *[
            BV[(BV.outcome == "gcog") & (BV.design == d)]
            for d in ["tte", "placebo", "pretrend", "early_vs_late"]
        ],
    )
    r(
        "Fair/poor self-rated health",
        "Physical activity",
        "pp",
        100,
        *[
            BV[(BV.outcome == "srh") & (BV.design == d)]
            for d in ["tte", "placebo", "pretrend", "early_vs_late"]
        ],
    )
    return rows


def signature(ax):
    rows = signature_rows()
    cols = [
        ("tte", "Target trial\n(start before\noutcome)"),
        ("pla", "Placebo\n(start after\noutcome)"),
        ("pre", "Before anyone\nstarts\n(t0)"),
        ("evl", "Early vs late\nadopters"),
    ]
    cmap = LinearSegmentedColormap.from_list("sig", ["#f4f3ef", "#e3b0bb", "#9b1b30"])
    exp = [
        ("If going online had an effect", [1, 0, 0, 1], EFF),
        ("If adopters are only selected", [1, 1, 1, 0], SELC),
    ]
    n = len(rows) + len(exp) + 1
    y = n
    for nm, v, col in exp:
        y -= 1
        for j, val in enumerate(v):
            ax.add_patch(
                Rectangle(
                    (j, y),
                    0.94,
                    0.86,
                    color=col if val else "#f4f3ef",
                    alpha=0.85 if val else 1,
                    lw=0,
                )
            )
            ax.text(
                j + 0.47,
                y + 0.43,
                "same as\ntarget trial" if val and j else ("effect" if val else "none"),
                ha="center",
                va="center",
                fontsize=FS * 5.4,
                color="white" if val else INK2,
            )
        ax.text(
            -0.08,
            y + 0.43,
            nm,
            ha="right",
            va="center",
            fontsize=FS * 6.2,
            color=col,
            weight="bold",
        )
    y -= 1
    ax.plot([-3.3, 4], [y + 0.93] * 2, color=GREY, lw=0.5, clip_on=False)
    last = None
    for r in rows:
        y -= 1
        if r["beh"] != last:
            ax.text(
                -3.3,
                y + 0.43,
                r["beh"],
                ha="left",
                va="center",
                fontsize=FS * 6.0,
                color=INK2,
                style="italic",
            )
            last = r["beh"]
        t = r["tte"]["est"]
        for j, (k, _) in enumerate(cols):
            p = r[k]
            if p is None:
                ax.add_patch(Rectangle((j, y), 0.94, 0.86, color="#ffffff", ec=GRID, lw=0.5))
                ax.text(j + 0.47, y + 0.43, "–", ha="center", va="center", color=MUTED)
                continue
            ratio = np.clip(p["est"] / t, -0.2, 1.4)
            ax.add_patch(Rectangle((j, y), 0.94, 0.86, color=cmap(max(ratio, 0) / 1.4), lw=0))
            sig = (p["lo"] > 0) or (p["hi"] < 0)
            u = "" if r["unit"] == "SD" else ""
            txt = f"{p['est']:+.2f}" if r["unit"] == "SD" else f"{p['est']:+.1f}"
            ax.text(
                j + 0.47,
                y + 0.5,
                txt.replace("-", "−"),
                ha="center",
                va="center",
                fontsize=FS * 6.0,
                weight="bold" if sig else "normal",
                color="white" if ratio > 0.85 else INK,
            )
            ax.text(
                j + 0.47,
                y + 0.2,
                (
                    f"{p['lo']:.2f} to {p['hi']:.2f}".replace("-", "−")
                    if r["unit"] == "SD"
                    else f"{p['lo']:.1f} to {p['hi']:.1f}".replace("-", "−")
                ),
                ha="center",
                va="center",
                fontsize=FS * 4.4,
                color="white" if ratio > 0.85 else INK2,
            )
        ax.text(
            -0.08,
            y + 0.43,
            f"{r['label']} ({r['unit']})",
            ha="right",
            va="center",
            fontsize=FS * 6.2,
        )
    for j, (_, nm) in enumerate(cols):
        ax.text(j + 0.47, n + 0.1, nm, ha="center", va="bottom", fontsize=FS * 6.0, weight="bold")
    ax.set_xlim(-3.3, 4.0)
    ax.set_ylim(y - 0.2, n + 2.1)
    ax.axis("off")
    cax = ax.inset_axes([0.6, -0.06, 0.38, 0.025])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1.4)),
        cax=cax,
        orientation="horizontal",
    )
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["0", "0.5", "1 = same as target trial"])
    cb.ax.tick_params(labelsize=FS * 5.2, length=1.5)
    cb.set_label("Estimate relative to target trial (colour)", fontsize=FS * 5.4)


def _sel_index(r, n=20000, seed=1):
    """Share of the way from the 'effect' prediction (0) to the 'selection' prediction (1), per design and combined:
    placebo/T, pre/T, 1 - (early-late)/T; multivariate normal draws from the pooled estimates with the between-design correlations
    estimated from per-person influence functions (composite: its own; other outcomes: those of word recall).
    """
    ks = ["tte", "pla", "pre", "evl"]
    P = pd.read_csv("results_corr_pooled.csv")
    y = "gcog" if "composite" in r["label"] else "recall"
    M = np.eye(4)
    for _, q in P[P.y == y].iterrows():
        i, j = ks.index(q.a), ks.index(q.b)
        M[i, j] = M[j, i] = q.rho
    use = [k for k in ks if r[k] is not None]
    idx = [ks.index(k) for k in use]
    se = lambda p: (p["hi"] - p["lo"]) / 3.92
    rng = np.random.default_rng(seed)
    mu = np.array([r[k]["est"] for k in use])
    sd = np.array([se(r[k]) for k in use])
    X = rng.multivariate_normal(mu, M[np.ix_(idx, idx)] * np.outer(sd, sd), n)
    T = X[:, 0]
    parts = {
        k: (X[:, i] / T if k != "evl" else 1 - X[:, i] / T) for i, k in enumerate(use) if k != "tte"
    }
    V = np.column_stack(list(parts.values()))
    w = 1 / np.var(V, axis=0)
    comb = (V * w).sum(1) / w.sum()
    q = lambda v: (np.median(v), *np.percentile(v, [2.5, 97.5]))
    return {k: q(v) for k, v in parts.items()}, q(comb)


def fig5():
    fig = plt.figure(figsize=(W, 7.15))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.5, 0.95], hspace=0.27, left=0.02, right=0.99, top=0.885, bottom=0.065
    )
    top = gs[0].subgridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.05)
    rows = [r for r in signature_rows() if r["beh"] == "Internet"]
    cols = [
        ("tte", "Target trial", "start before outcome"),
        ("pla", "Placebo", "start after outcome"),
        ("pre", "Before anyone starts", "difference at t0"),
        ("evl", "Early vs late", "adopters, at t2"),
    ]
    pred = {
        "pla": (0, 1),
        "pre": (0, 1),
        "evl": (1, 0),
    }  # (effect, selection) in units of the target-trial estimate
    A = top[0].subgridspec(len(rows), 5, width_ratios=[1.02, 1, 1, 1, 1], wspace=0.1, hspace=0.18)
    for i, r in enumerate(rows):
        T = r["tte"]["est"]
        vals = [r[k] for k, *_ in cols if r[k] is not None]
        lo_ = min([p["lo"] for p in vals] + [0, T])
        hi_ = max([p["hi"] for p in vals] + [0, T])
        pad = 0.12 * (hi_ - lo_)
        lim = (lo_ - pad, hi_ + pad)
        al = fig.add_subplot(A[i, 0])
        al.axis("off")
        al.text(
            1.0, 0.5, r["label"], ha="right", va="center", fontsize=FS * 5.6, transform=al.transAxes
        )
        al.text(
            1.0,
            0.08,
            "SD" if r["unit"] == "SD" else "percentage points",
            ha="right",
            va="center",
            fontsize=FS * 4.6,
            color=MUTED,
            transform=al.transAxes,
        )
        for j, (k, nm, sub) in enumerate(cols):
            ax = fig.add_subplot(A[i, j + 1])
            p = r[k]
            ax.axvline(0, color="#cfcdc6", lw=0.7, zorder=0)
            ax.set_xlim(*lim)
            ax.set_ylim(-0.6, 0.75)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_facecolor("#f7f6f2")
            if i == 0:
                ax.text(
                    0.5,
                    1.5,
                    nm,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=FS * 5.8,
                    weight="bold",
                )
                ax.text(
                    0.5,
                    1.12,
                    sub,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=FS * 4.8,
                    color=INK2,
                )
            if p is None:
                ax.text(
                    0.5, 0.5, "–", ha="center", va="center", color=MUTED, transform=ax.transAxes
                )
                continue
            if k == "evl" and r["unit"] == "SD":
                ax.axvspan(-0.10, 0.10, color="#ecebe5", zorder=0, lw=0)
            if k in pred:
                e_, s_v = pred[k]
                ax.plot(
                    [e_ * T] * 2, [-0.45, 0.3], color=EFF, lw=2.0, solid_capstyle="butt", zorder=2
                )
                ax.plot(
                    [s_v * T] * 2, [-0.45, 0.3], color=SELC, lw=2.0, solid_capstyle="butt", zorder=2
                )
                xs_ = [e_ * T, s_v * T, p["est"] if p is not None else 0]
            else:
                xs_ = [p["est"]]
            sig = (p["lo"] > 0) or (p["hi"] < 0)
            ax.plot([p["lo"], p["hi"]], [0, 0], color=OBS, lw=1.3, zorder=3)
            ax.plot(
                p["est"], 0, "o", color=OBS if sig else SURF, mec=OBS, ms=4.4, mew=1.0, zorder=4
            )
            txt = (f"{p['est']:+.2f}" if r["unit"] == "SD" else f"{p['est']:+.1f}").replace(
                "-", "−"
            )
            fr = [(x_ - lim[0]) / (lim[1] - lim[0]) for x_ in xs_]
            left_ok = min(abs(f_ - 0.13) for f_ in fr)
            right_ok = min(abs(f_ - 0.87) for f_ in fr)
            ax.text(
                0.03 if left_ok >= right_ok else 0.97,
                0.97,
                txt,
                transform=ax.transAxes,
                ha="left" if left_ok >= right_ok else "right",
                va="top",
                fontsize=FS * 4.7,
                color=OBS,
            )
    # legend for a
    fig.text(0.012, 0.945, "a", fontsize=FS * 8.5, weight="bold", va="bottom")
    fig.text(
        0.035, 0.945, "Predicted versus observed", fontsize=FS * 7.2, weight="bold", va="bottom"
    )
    la = fig.add_axes([0.03, 0.922, 0.62, 0.022])
    la.axis("off")
    la.legend(
        handles=[
            Line2D([], [], color=EFF, lw=2, label="if going online improved outcomes"),
            Line2D([], [], color=SELC, lw=2, label="if adopters were only selected"),
            Line2D([], [], color=OBS, lw=1.3, marker="o", ms=4, label="observed (95% CI)"),
        ],
        loc="center left",
        ncol=3,
        frameon=False,
        fontsize=FS * 5.0,
        handlelength=1.4,
        columnspacing=1.0,
    )
    # b selection index
    ax = fig.add_subplot(top[1])
    lab(ax, "b", x=-0.02, y=1.115, title="Selection index")
    sub = [r for r in rows if (r["tte"]["lo"] > 0) or (r["tte"]["hi"] < 0)]
    ax.axvline(0, color=EFF, lw=1.0)
    ax.axvline(1, color=SELC, lw=1.0)
    ax.text(
        0,
        1.005,
        "effect",
        transform=ax.get_xaxis_transform(),
        color=EFF,
        ha="center",
        va="bottom",
        fontsize=FS * 5.2,
        weight="bold",
    )
    ax.text(
        1,
        1.005,
        "selection",
        transform=ax.get_xaxis_transform(),
        color=SELC,
        ha="center",
        va="bottom",
        fontsize=FS * 5.2,
        weight="bold",
    )
    mk = {
        "pla": ("s", "placebo"),
        "pre": ("^", "before anyone starts"),
        "evl": ("v", "early vs late"),
    }
    labs = []
    for i, r in enumerate(rows):
        yy = len(rows) - 1 - i
        labs.append(r["label"])
        if r not in sub:
            ax.text(
                0.5,
                yy,
                "target trial CI includes 0",
                ha="center",
                va="center",
                fontsize=FS * 4.6,
                color=MUTED,
                style="italic",
            )
            continue
        parts, comb = _sel_index(r)
        ax.plot([max(comb[1], -0.32), min(comb[2], 1.62)], [yy, yy], color=OBS, lw=1.6, zorder=4)
        ax.plot(comb[0], yy, "D", color=OBS, ms=4.6, mec=SURF, mew=0.6, zorder=5)
        ax.text(1.66, yy, f"{comb[0]:.2f}", va="center", ha="left", fontsize=FS * 5.0, color=OBS)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlim(-0.32, 1.78)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.tick_params(axis="x", labelsize=FS * 5.2)
    ax.set_xlabel("Position between the two predictions", fontsize=FS * 5.4)
    clean(ax)
    ax.grid(False)
    # c-e event studies
    ES = pd.read_csv("results_es_all.csv")
    ESP = pd.read_csv("results_es_pooled.csv")
    bot = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 1.0], wspace=0.42)

    def es(
        ax,
        y,
        sc,
        ylim,
        L,
        t,
        yl,
        ES=ES,
        ESP=ESP,
        xl="Waves since first use",
        shade="online",
        extra=None,
    ):
        panel_title(ax, L, t)
        p = ESP[ESP.y == y].copy()
        ev = p[p.e.str.lstrip("-").str.isdigit() & (p.k >= 3)].copy()
        ev["e"] = ev.e.astype(int)
        ev = pd.concat([ev, pd.DataFrame([dict(e=-1, est=0, lo=0, hi=0)])]).sort_values("e")
        ax.axvspan(-0.5, 3.4, color="#f4f3ef", zorder=-3, lw=0)
        for c in COH:
            x = ES[
                (ES.y == y) & (ES.cohort == c) & ES.e.astype(str).str.lstrip("-").str.isdigit()
            ].copy()
            x["e"] = x.e.astype(int)
            x = pd.concat([x, pd.DataFrame([dict(e=-1, est=0)])]).sort_values("e")
            ax.plot(
                x.e + (COH.index(c) - 1.5) * 0.06,
                np.clip(x.est * sc, *ylim),
                color=COL[c],
                lw=0.6,
                alpha=0.6,
                marker=MK[c],
                ms=1.9,
            )
        sl = p[p.e == "slope_pre"].est.iloc[0]
        xs = np.array([-4, 3])
        ax.plot(xs, sl * (xs + 1) * sc, color=SELC, lw=0.9, ls=(0, (3, 2)))
        ax.fill_between(
            ev.e,
            np.clip(ev.lo * sc, *ylim),
            np.clip(ev.hi * sc, *ylim),
            color=OBS,
            alpha=0.14,
            lw=0,
        )
        ax.plot(ev.e, ev.est * sc, color=OBS, lw=1.5, marker="o", ms=3, mec=SURF, mew=0.4, zorder=5)
        ax.axhline(0, color=GRID, lw=0.6, zorder=-1)
        pa = p[p.e == "post_avg_adj"].iloc[0]
        ps = p[p.e == "slope_pre"].iloc[0]
        f = lambda v: (f"{v * sc:+.2f}" if sc == 1 else f"{v * sc:+.1f}").replace("-", "−")
        ax.text(
            0.03,
            0.97,
            f"before: {f(ps.est)}/wave\nafter vs trend: {f(pa.est)}",
            transform=ax.transAxes,
            va="top",
            fontsize=FS * 4.8,
            color=INK2,
        )
        ax.text(
            3.3,
            ylim[0] + 0.04 * (ylim[1] - ylim[0]),
            shade,
            ha="right",
            va="bottom",
            fontsize=FS * 4.6,
            color=MUTED,
        )
        if extra is not None:
            q = ESP[ESP.y == extra].copy()
            q = q[q.e.str.lstrip("-").str.isdigit() & (q.k >= 3)].copy()
            q["e"] = q.e.astype(int)
            q = pd.concat([q, pd.DataFrame([dict(e=-1, est=0)])]).sort_values("e")
            ax.plot(
                q.e,
                q.est * sc,
                color=OBS3,
                lw=1.3,
                ls=(0, (4, 1.5)),
                marker="s",
                ms=2.6,
                mec=SURF,
                mew=0.4,
                zorder=4,
            )
            ax.text(
                q.e.iloc[0] - 0.1,
                q.est.iloc[0] * sc + 0.05,
                "composite",
                fontsize=FS * 4.6,
                color=OBS3,
                ha="left",
                va="bottom",
                weight="bold",
            )
        ax.set_xticks(range(-4, 4))
        ax.set_xlim(-4.4, 3.4)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        clean(ax, "y")
        ax.spines["left"].set_visible(True)

    es(
        fig.add_subplot(bot[0]),
        "z_recall",
        1,
        (-0.22, 0.3),
        "c",
        "Recall around starting",
        "Difference (SD)",
        extra="z_gcog",
    )
    es(
        fig.add_subplot(bot[1]),
        "z_recall",
        1,
        (-0.22, 0.3),
        "d",
        "Recall around stopping",
        "Difference (SD)",
        ES=pd.read_csv("results_es_stop_all.csv"),
        ESP=pd.read_csv("results_es_stop_pooled.csv").astype({"e": str}),
        xl="Waves since first non-use",
        shade="stopped",
    )
    # f cohort pre-trend slope and post deviation
    ax = fig.add_subplot(bot[2])
    panel_title(ax, "e", "Cohort pre-trends")
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    grp = [("slope_pre", "before: slope per wave"), ("post_avg_adj", "after: deviation from trend")]
    yy = 0
    ticks, tl = [], []
    for gk, gname in grp:
        for c in COH + ["Pooled"]:
            for off, y, m_, fc in [(0.17, "z_recall", None, True), (-0.17, "z_gcog", "s", False)]:
                if c == "Pooled":
                    q = ESP[(ESP.y == y) & (ESP.e == gk)]
                    if not len(q):
                        continue
                    q = q.iloc[0]
                    e_, lo, hi, col, mm = q.est, q.lo, q.hi, OBS, "D"
                else:
                    q = ES[(ES.y == y) & (ES.cohort == c) & (ES.e == gk)]
                    if not len(q):
                        continue
                    q = q.iloc[0]
                    e_, lo, hi, col, mm = (
                        q.est,
                        q.est - 1.96 * q.se,
                        q.est + 1.96 * q.se,
                        COL[c],
                        MK[c],
                    )
                ax.plot([lo, hi], [-yy + off] * 2, color=col, lw=0.9)
                ax.plot(
                    e_, -yy + off, mm, color=col if fc else SURF, mec=col, ms=3.4, mew=0.9, zorder=4
                )
            ticks.append(-yy)
            tl.append(c)
            yy += 1
        ax.text(
            -0.21,
            -yy + 5.55,
            gname,
            fontsize=FS * 4.9,
            color=INK2,
            weight="bold",
            ha="left",
            va="center",
        )
        yy += 0.9
    ax.set_yticks(ticks)
    ax.set_yticklabels(tl, fontsize=FS * 5.0)
    for t_, c in zip(ax.get_yticklabels(), tl):
        t_.set_color(COL.get(c, OBS))
    ax.set_xlim(-0.21, 0.2)
    ax.set_ylim(-yy + 0.6, 1.0)
    ax.set_xlabel("SD (filled, recall; open, composite)", fontsize=FS * 5.0)
    clean(ax)
    ax.grid(False)
    cohort_legend(fig, y=0.97)
    save(fig, "NH3_Fig5")


def ed_pretrend_sens(ax, L):
    panel_title(ax, L, "Pre-trend sensitivity")
    ES = pd.read_csv("results_es_all.csv")
    ESP = pd.read_csv("results_es_pooled.csv")
    Eall = ES[ES.e.astype(str).isin(["0", "1", "2"])].copy()
    Eall["e"] = Eall.e.astype(int)
    Ms = np.linspace(0, 0.03, 61)
    TT_ = sel(TTE, y="Y_recall")
    tt = pooled(ci(TT_))["est"]
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.axhline(tt, color=SELC, lw=0.8, ls=(0, (1, 1.5)))
    ax.text(
        0.0005,
        tt + 0.008,
        f"target-trial estimate, recall ({tt:.2f})",
        fontsize=FS * 4.6,
        color=SELC,
        va="bottom",
    )
    for y, col, al, nm in [("z_recall", OBS, 0.18, "recall"), ("z_gcog", OBS3, 0.30, "composite")]:
        w = Eall[Eall.y == y].groupby("e").n_treated.sum()
        h = w.index.values + 1
        fct = (w.values * (h * (h + 1) / 2 + 1.5 * h)).sum() / w.sum()
        q = ESP[(ESP.y == y) & (ESP.e == "post_avg_adj")].iloc[0]
        ls_ = "-" if y == "z_recall" else (0, (4, 1.6))
        ax.plot(Ms, q.hi + fct * Ms, color=col, lw=1.4, ls=ls_)
        ax.plot(Ms, q.lo - fct * Ms, color=col, lw=1.4, ls=ls_)
        ax.text(
            0.0305,
            (q.hi + fct * 0.03) if y == "z_recall" else (q.lo - fct * 0.03),
            nm,
            fontsize=FS * 4.8,
            color=col,
            va="center",
            ha="left",
            weight="bold",
        )
        mx = (tt - q.hi) / fct
        ax.plot(mx, tt, "o", color=col, ms=3.2, mec=SURF, mew=0.5, zorder=6)
        print("breakdown M", y, round(mx, 4), "factor", round(fct, 2))
        pre = ESP[(ESP.y == y) & ESP.e.isin(["-4", "-3", "-2"])].copy()
        pre["e"] = pre.e.astype(int)
        pre = pre.sort_values("e")
        v = np.r_[pre.est.values, 0]
        d2 = np.abs(np.diff(v, 2)).max()
        if y == "z_recall":
            ax.axvline(d2, color=INK2, lw=0.6, ls=(0, (2, 2)))
            print("d2", y, round(d2, 4))
            ax.text(
                d2 - 0.0006,
                0.33,
                "largest bend\nbefore adoption",
                ha="right",
                va="top",
                fontsize=FS * 4.5,
                color=INK2,
            )
    ax.set_xlim(0, 0.039)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xticks([0, 0.01, 0.02, 0.03])
    ax.set_xlabel("Allowed bend, M (SD per wave)")
    ax.set_ylabel("Post-adoption effect (SD)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)


# ================================================================== Fig. 7  internet vs physical activity, side by side
def _es_panel(ax, ES, ESP, y, sc, ylim, L, t, yl, xl, shade, extra=None, note=True):
    panel_title(ax, L, t)
    ESP = ESP.astype({"e": str})
    ES = ES.copy()
    p = ESP[ESP.y == y].copy()
    ev = p[p.e.str.lstrip("-").str.isdigit() & (p.k >= 3)].copy()
    ev["e"] = ev.e.astype(int)
    ev = pd.concat([ev, pd.DataFrame([dict(e=-1, est=0, lo=0, hi=0)])]).sort_values("e")
    ax.axvspan(-0.5, 3.4, color="#f4f3ef", zorder=-3, lw=0)
    for c in COH:
        x = ES[
            (ES.y == y) & (ES.cohort == c) & ES.e.astype(str).str.lstrip("-").str.isdigit()
        ].copy()
        if not len(x):
            continue
        x["e"] = x.e.astype(int)
        x = pd.concat([x, pd.DataFrame([dict(e=-1, est=0)])]).sort_values("e")
        ax.plot(
            x.e + (COH.index(c) - 1.5) * 0.06,
            np.clip(x.est * sc, *ylim),
            color=COL[c],
            lw=0.6,
            alpha=0.6,
            marker=MK[c],
            ms=1.9,
        )
    q = p[p.e == "slope_pre"]
    if len(q):
        sl = q.est.iloc[0]
        xs = np.array([-4, 3])
        ax.plot(xs, sl * (xs + 1) * sc, color=SELC, lw=0.9, ls=(0, (3, 2)))
    ax.fill_between(
        ev.e, np.clip(ev.lo * sc, *ylim), np.clip(ev.hi * sc, *ylim), color=OBS, alpha=0.14, lw=0
    )
    ax.plot(ev.e, ev.est * sc, color=OBS, lw=1.5, marker="o", ms=3, mec=SURF, mew=0.4, zorder=5)
    if extra is not None:
        q2 = ESP[ESP.y == extra].copy()
        q2 = q2[q2.e.str.lstrip("-").str.isdigit() & (q2.k >= 3)].copy()
        q2["e"] = q2.e.astype(int)
        q2 = pd.concat([q2, pd.DataFrame([dict(e=-1, est=0)])]).sort_values("e")
        ax.plot(
            q2.e,
            q2.est * sc,
            color=OBS3,
            lw=1.3,
            ls=(0, (4, 1.5)),
            marker="s",
            ms=2.6,
            mec=SURF,
            mew=0.4,
            zorder=4,
        )
    ax.axhline(0, color=GRID, lw=0.6, zorder=-1)
    if note:
        f = lambda v: (f"{v * sc:+.2f}" if sc == 1 else f"{v * sc:+.1f}").replace("-", "−")
        ps = p[p.e == "slope_pre"]
        pa = p[p.e == "post_avg_adj"]
        ax.text(
            0.03,
            0.97,
            f"before: {f(ps.est.iloc[0])}/wave\nafter vs trend: {f(pa.est.iloc[0])}",
            transform=ax.transAxes,
            va="top",
            fontsize=FS * 4.8,
            color=INK2,
        )
    ax.text(
        3.3,
        ylim[0] + 0.04 * (ylim[1] - ylim[0]),
        shade,
        ha="right",
        va="bottom",
        fontsize=FS * 4.6,
        color=MUTED,
    )
    ax.set_xticks(range(-4, 4))
    ax.set_xlim(-4.4, 3.4)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    clean(ax, "y")
    ax.spines["left"].set_visible(True)


def fig7():
    fig = plt.figure(figsize=(W, 6.3))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.0], hspace=0.42, left=0.02, right=0.985, top=0.855, bottom=0.075
    )
    top = gs[0].subgridspec(1, 3, width_ratios=[1.95, 0.5, 0.82], wspace=0.0)
    rows = [r for r in signature_rows() if r["beh"] == "Physical activity"]
    cols = [
        ("tte", "Target trial", "start before outcome"),
        ("pla", "Placebo", "start after outcome"),
        ("pre", "Before anyone starts", "difference at t0"),
        ("evl", "Early vs late", "starters, at t2"),
    ]
    pred = {"pla": (0, 1), "pre": (0, 1), "evl": (1, 0)}
    A = top[0].subgridspec(len(rows), 5, width_ratios=[0.9, 1, 1, 1, 1], wspace=0.1, hspace=0.22)
    for i, r in enumerate(rows):
        T = r["tte"]["est"]
        vals = [r[k] for k, *_ in cols if r[k] is not None]
        lo_ = min([p["lo"] for p in vals] + [0, T])
        hi_ = max([p["hi"] for p in vals] + [0, T])
        pad = 0.12 * (hi_ - lo_)
        lim = (lo_ - pad, hi_ + pad)
        al = fig.add_subplot(A[i, 0])
        al.axis("off")
        al.text(
            1.0,
            0.55,
            r["label"].replace("Fair/poor self-rated health", "Fair/poor health"),
            ha="right",
            va="center",
            fontsize=FS * 5.6,
            transform=al.transAxes,
        )
        al.text(
            1.0,
            0.2,
            "SD" if r["unit"] == "SD" else "percentage points",
            ha="right",
            va="center",
            fontsize=FS * 4.6,
            color=MUTED,
            transform=al.transAxes,
        )
        for j, (k, nm, sub) in enumerate(cols):
            ax = fig.add_subplot(A[i, j + 1])
            p = r[k]
            ax.axvline(0, color="#cfcdc6", lw=0.7, zorder=0)
            ax.set_xlim(*lim)
            ax.set_ylim(-0.6, 0.75)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_facecolor("#f7f6f2")
            if i == 0:
                ax.text(
                    0.5,
                    1.42,
                    nm,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=FS * 5.8,
                    weight="bold",
                )
                ax.text(
                    0.5,
                    1.08,
                    sub,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=FS * 4.8,
                    color=INK2,
                )
            if k == "evl" and r["unit"] == "SD":
                ax.axvspan(-0.10, 0.10, color="#ecebe5", zorder=0, lw=0)
            if k in pred:
                e_, s_v = pred[k]
                ax.plot(
                    [e_ * T] * 2, [-0.45, 0.3], color=EFF, lw=2.0, solid_capstyle="butt", zorder=2
                )
                ax.plot(
                    [s_v * T] * 2, [-0.45, 0.3], color=SELC, lw=2.0, solid_capstyle="butt", zorder=2
                )
                xs_ = [e_ * T, s_v * T, p["est"]]
            else:
                xs_ = [p["est"]]
            sig = (p["lo"] > 0) or (p["hi"] < 0)
            ax.plot([p["lo"], p["hi"]], [0, 0], color=OBS, lw=1.3, zorder=3)
            ax.plot(
                p["est"], 0, "o", color=OBS if sig else SURF, mec=OBS, ms=4.4, mew=1.0, zorder=4
            )
            txt = (f"{p['est']:+.2f}" if r["unit"] == "SD" else f"{p['est']:+.1f}").replace(
                "-", "−"
            )
            fr = [(x_ - lim[0]) / (lim[1] - lim[0]) for x_ in xs_]
            lok = min(abs(f_ - 0.13) for f_ in fr)
            rok = min(abs(f_ - 0.87) for f_ in fr)
            ax.text(
                0.03 if lok >= rok else 0.97,
                0.97,
                txt,
                transform=ax.transAxes,
                ha="left" if lok >= rok else "right",
                va="top",
                fontsize=FS * 4.7,
                color=OBS,
            )
    fig.text(0.012, 0.935, "a", fontsize=FS * 8.5, weight="bold", va="bottom")
    fig.text(
        0.035,
        0.935,
        "Predicted versus observed: starting weekly vigorous activity",
        fontsize=FS * 7.0,
        weight="bold",
        va="bottom",
    )
    la = fig.add_axes([0.03, 0.905, 0.6, 0.025])
    la.axis("off")
    la.legend(
        handles=[
            Line2D([], [], color=EFF, lw=2, label="if activity improved outcomes"),
            Line2D([], [], color=SELC, lw=2, label="if starters were only selected"),
            Line2D([], [], color=OBS, lw=1.3, marker="o", ms=4, label="observed (95% CI)"),
        ],
        loc="center left",
        ncol=3,
        frameon=False,
        fontsize=FS * 5.0,
        handlelength=1.4,
        columnspacing=1.0,
    )
    # b predictors: internet vs activity
    ax = fig.add_subplot(top[2])
    lab(ax, "b", x=-0.5, y=1.12, title="Predictors of starting")
    AP = NH.AP
    VP = pd.read_csv("results_vig_pred_pooled.csv")
    terms = [
        ("lvl", "Memory, two-wave mean"),
        ("drz", "Recent memory change"),
        ("edu", "Education"),
        ("wealth_q", "Wealth"),
        ("ncond", "Chronic conditions"),
        ("srh_poor", "Fair/poor health"),
        ("iadl_any", "Any IADL difficulty"),
        ("age", "Age (per 10 years)"),
    ]
    n = len(terms)
    ax.axvline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    for i, (t, nm) in enumerate(terms):
        yy = n - i
        if t in ("lvl", "drz"):
            ax.axhspan(yy - 0.45, yy + 0.45, color="#eef4f5", zorder=-3, lw=0)
        for off, src, mk_, col in [(0.17, "int", "D", OBS), (-0.17, "vig", "s", "#8a8880")]:
            if src == "int":
                q = AP[(AP.part == "predictor") & (AP.term == t)].iloc[0]
            else:
                q = VP[VP.term == t].iloc[0]
            e, lo, hi = np.exp(q.est), np.exp(q.lo), np.exp(q.hi)
            sig = lo > 1 or hi < 1
            ax.plot([lo, hi], [yy + off] * 2, color=col, lw=1.2)
            ax.plot(
                e, yy + off, mk_, color=col if sig else SURF, mec=col, ms=3.8, mew=0.9, zorder=4
            )
    ax.set_xscale("log")
    ax.set_xlim(0.28, 3.0)
    ax.set_xticks([0.5, 1, 2])
    ax.set_xticklabels(["0.5", "1", "2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([t[1] for t in terms], fontsize=FS * 5.2)
    ax.set_ylim(0.4, n + 0.6)
    clean(ax)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Odds ratio (log scale)")
    ax.legend(
        handles=[
            Line2D([], [], color=OBS, marker="D", ms=3.8, lw=1.2, label="going online"),
            Line2D([], [], color="#8a8880", marker="s", ms=3.8, lw=1.2, label="vigorous activity"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.3, 1.0),
        ncol=2,
        frameon=False,
        fontsize=FS * 4.8,
        handlelength=1.4,
    )
    # c-f event studies around starting and stopping activity
    bot = gs[1].subgridspec(1, 4, wspace=0.5)
    ES1, EP1 = pd.read_csv("results_es_vig_all.csv"), pd.read_csv("results_es_vig_pooled.csv")
    ES2, EP2 = pd.read_csv("results_es_stopvig_all.csv"), pd.read_csv(
        "results_es_stopvig_pooled.csv"
    )
    _es_panel(
        fig.add_subplot(bot[0]),
        ES1,
        EP1,
        "z_recall",
        1,
        (-0.16, 0.18),
        "c",
        "Recall around starting",
        "Difference (SD)",
        "Waves since starting",
        "active",
        extra="z_gcog",
    )
    _es_panel(
        fig.add_subplot(bot[1]),
        ES1,
        EP1,
        "srh",
        100,
        (-9, 9),
        "d",
        "Health around starting",
        "Difference (pp)",
        "Waves since starting",
        "active",
    )
    _es_panel(
        fig.add_subplot(bot[2]),
        ES2,
        EP2,
        "z_recall",
        1,
        (-0.16, 0.18),
        "e",
        "Recall around stopping",
        "Difference (SD)",
        "Waves since stopping",
        "stopped",
        extra="z_gcog",
    )
    _es_panel(
        fig.add_subplot(bot[3]),
        ES2,
        EP2,
        "srh",
        100,
        (-9, 9),
        "f",
        "Health around stopping",
        "Difference (pp)",
        "Waves since stopping",
        "stopped",
    )
    cohort_legend(fig, y=0.965, cohorts=["HRS", "ELSA", "SHARE"])
    save(fig, "NH3_EDFig4")


# ================================================================== Supplementary Figs 4-5 (cohort forests, unified style)
def _forest2(ax, rows, xlim, scale=1.0, shade=(), band=None, band_rows=(), fmt="{:.2f}"):
    n = len(rows)
    offs = np.linspace(-0.3, 0.06, 4)
    L, H = xlim
    for i in band_rows:
        ax.add_patch(
            Rectangle((-band, n - i - 0.5), 2 * band, 1, fc="#e3eef0", ec="none", zorder=-1)
        )
    for i, (label, g) in enumerate(rows):
        yy = n - i
        if i in shade:
            ax.axhspan(yy - 0.5, yy + 0.5, color="#f7f6f2", zorder=-2, lw=0)
        if g is None or len(g) == 0:
            continue
        for j, c in enumerate(COH):
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
        gc = g[g.cohort.isin(COH)]
        if len(gc) >= 1:
            p = pooled(ci(gc), scale) if len(gc) >= 2 else None
            if p is None:
                continue
            ax.plot([max(p["lo"], L), min(p["hi"], H)], [yy + 0.3] * 2, color=OBS, lw=1.4, zorder=3)
            ax.plot(p["est"], yy + 0.3, "D", color=OBS, ms=4.2, mec=SURF, mew=0.5, zorder=4)
            ax.text(
                1.01,
                yy + 0.3,
                fmt.format(p["est"]).replace("-", "−"),
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=FS * 5.6,
                weight="bold",
                color=OBS,
            )
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=-1)
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS * 5.6)
    ax.set_ylim(0.45, n + 0.55)
    ax.set_xlim(L, H)
    clean(ax)


def _rows(src, outs):
    DES = [
        ("tte", "Target trial"),
        ("placebo", "Placebo (later starters)"),
        ("pretrend", "Before anyone starts (t0)"),
        ("early_vs_late", "Early vs late"),
    ]
    rows, heads = [], []
    for o, name, s_ in outs:
        heads.append(len(rows))
        rows.append((name, None))
        for d, dn in DES:
            rows.append((dn, s_[(s_.outcome == o) & (s_.design == d)]))
    return rows, heads


def _bold(ax, heads):
    for k, t in enumerate(ax.get_yticklabels()):
        if k in heads:
            t.set_fontweight("bold")


def supp4():
    NO = pd.read_csv("results_newout_all.csv")
    DM = pd.read_csv("results_dementia_HRS.csv")
    fig = plt.figure(figsize=(W, 6.6))
    gs = fig.add_gridspec(1, 2, wspace=0.75, left=0.17, right=0.95, top=0.92, bottom=0.07)
    ax = fig.add_subplot(gs[0])
    panel_title(ax, "a", "Cognition and life satisfaction (SD)", x=-0.45)
    r, h = _rows(
        NO,
        [
            ("ser7", "Serial 7s", NO),
            ("verbf", "Verbal fluency", NO),
            ("gcog", "Two-domain composite", NO),
            ("lsat", "Life satisfaction", NO),
        ],
    )
    _forest2(
        ax,
        r,
        (-0.3, 0.45),
        shade=list(range(0, 5)) + list(range(10, 15)),
        band=0.10,
        band_rows=[4, 9, 14, 19],
    )
    _bold(ax, h)
    ax.set_xlabel("Difference (SD)")
    ax = fig.add_subplot(gs[1])
    panel_title(ax, "b", "Impairment, dementia and health (pp)", x=-0.45)
    r, h = _rows(
        NO,
        [
            ("cimp", "Cognitive impairment (HRS)", NO),
            ("dem", "Dementia (HRS)", DM),
            ("srh", "Fair/poor self-rated health", NO),
        ],
    )
    _forest2(
        ax,
        r,
        (-12, 8),
        scale=100,
        fmt="{:.1f}",
        shade=list(range(0, 5)) + list(range(10, 15)),
        band=2,
        band_rows=[4, 9, 14],
    )
    _bold(ax, h)
    ax.set_xlabel("Risk difference (percentage points)")
    cohort_legend(fig, y=0.955)
    save(fig, "SuppFig4_outcomes")


def supp5():
    BV = pd.read_csv("results_behav_vig_all.csv")
    fig = plt.figure(figsize=(W, 4.4))
    gs = fig.add_gridspec(1, 2, wspace=0.75, left=0.17, right=0.95, top=0.89, bottom=0.1)
    ax = fig.add_subplot(gs[0])
    panel_title(ax, "a", "Cognition (SD)", x=-0.45)
    r, h = _rows(BV, [("recall", "Word recall", BV), ("gcog", "Two-domain composite", BV)])
    _forest2(ax, r, (-0.25, 0.3), shade=range(0, 5), band=0.10, band_rows=[4, 9])
    _bold(ax, h)
    ax.set_xlabel("Difference (SD)")
    ax = fig.add_subplot(gs[1])
    panel_title(ax, "b", "Fair/poor self-rated health (pp)", x=-0.45)
    r, h = _rows(BV, [("srh", "Fair/poor self-rated health", BV)])
    _forest2(ax, r, (-10, 6), scale=100, fmt="{:.1f}", band=2, band_rows=[4])
    _bold(ax, h)
    ax.set_xlabel("Risk difference (percentage points)")
    cohort_legend(fig, y=0.93)
    save(fig, "SuppFig5_activity")


# ================================================================== new Fig. 8  how large an effect remains compatible
def fig8():
    d = pd.read_csv("simulation_grid4.csv")
    VM = {
        "base": ("o", "#4d4b46"),
        "trigger": ("^", "#8a8880"),
        "nonpersist": ("s", "#b3b1aa"),
        "cumulative": ("D", OBS3),
    }
    VL = {
        "base": "selection on cognition",
        "trigger": "+ adverse-event trigger",
        "nonpersist": "+ 30% stop each wave",
        "cumulative": "+ effect grows with use",
    }
    fig = plt.figure(figsize=(W, 6.3))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1, 1], hspace=0.45, left=0.075, right=0.985, top=0.93, bottom=0.075
    )
    r1 = gs[0].subgridspec(1, 3, wspace=0.45)
    TT = pooled(ci(sel(TTE, y="Y_recall")))
    PL = pooled(ci(sel(PLA, y="Y_recall")))
    # a scenarios
    ax = fig.add_subplot(r1[0])
    panel_title(ax, "a", f"{len(d)} simulated scenarios")
    for v, (mk, col) in VM.items():
        s_ = d[d.variant == v]
        ax.scatter(
            s_.TTE, s_.placebo, s=5 if v == "base" else 7, marker=mk, color=col, alpha=0.55, lw=0
        )
    ax.add_patch(
        Rectangle(
            (TT["lo"], PL["lo"]),
            TT["hi"] - TT["lo"],
            PL["hi"] - PL["lo"],
            fill=False,
            ec=OBS,
            lw=1.0,
            ls=(0, (3, 2)),
            zorder=4,
        )
    )
    ax.plot(TT["est"], PL["est"], "D", color=OBS, ms=5, mec=SURF, mew=0.6, zorder=5)
    ax.text(
        TT["hi"] + 0.006,
        PL["lo"],
        "observed\n(95% CIs)",
        fontsize=FS * 4.8,
        color=OBS,
        va="bottom",
        ha="left",
        bbox=dict(fc=SURF, ec="none", pad=0.3, alpha=0.85),
        zorder=6,
    )
    ax.set_xlabel("Simulated target trial (SD)")
    ax.set_ylabel("Simulated placebo (SD)")
    clean(ax, "both")
    ax.legend(
        handles=[
            Line2D([], [], marker=mk, color=col, ls="", ms=3.5, label=VL[v])
            for v, (mk, col) in VM.items()
        ],
        frameon=False,
        fontsize=FS * 4.5,
        loc="upper left",
        handletextpad=0.1,
        borderaxespad=0.1,
        labelspacing=0.25,
    )
    # b early-late vs true effect among fitting scenarios
    ax = fig.add_subplot(r1[1])
    f = d[d.fit4]
    panel_title(ax, "b", f"{len(f)} scenarios fit all four")
    EL = _pool90(WIT[(WIT.contrast == "early vs late") & (WIT.y == "Y_recall")])
    ax.axhspan(EL["lo"], EL["hi"], color="#e3eef0", zorder=-3, lw=0)
    ax.axhline(EL["est"], color=OBS, lw=1.0)
    ax.text(
        0.205,
        EL["est"] - 0.006,
        f"observed {EL['est']:.3f}",
        fontsize=FS * 4.8,
        ha="right",
        va="top",
        color=OBS,
    )
    ax.text(
        0.205, EL["lo"] + 0.004, "95% CI", fontsize=FS * 4.8, ha="right", va="bottom", color=OBS
    )
    rng = np.random.default_rng(1)
    for v, (mk, col) in VM.items():
        s_ = f[f.variant == v]
        if len(s_):
            ax.scatter(
                s_.beta + rng.uniform(-0.005, 0.005, len(s_)),
                s_.EL,
                s=9,
                marker=mk,
                color=col,
                lw=0.3,
                ec="white",
                zorder=3,
            )
    ax.plot([-0.01, 0.21], [-0.01, 0.21], color=EFF, lw=0.8, ls=(0, (3, 2)))
    ax.text(
        0.205,
        0.188,
        "early − late\n= true effect",
        fontsize=FS * 4.5,
        color=EFF,
        ha="right",
        va="top",
    )
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlim(-0.012, 0.212)
    ax.set_ylim(-0.07, 0.2)
    ax.set_xlabel("True effect of starting (SD)")
    ax.set_ylabel("Simulated early − late (SD)")
    clean(ax, "both")
    # c compatibility curve
    ax = fig.add_subplot(r1[2])
    panel_title(ax, "c", "Compatibility with the data")
    g = d.groupby("beta")
    fit = g.fit4.mean()
    comp95 = g.apply(lambda z: (z.fit4 & z.EL.between(EL["lo"], EL["hi"])).mean())
    comp90 = g.apply(lambda z: (z.fit4 & z.EL.between(EL["lo90"], EL["hi90"])).mean())
    ax.plot(
        fit.index,
        fit * 100,
        color="#b3b1aa",
        lw=1.2,
        marker="o",
        ms=3,
        label="fit target trial, placebo\nand pre-trends",
    )
    ax.plot(
        comp95.index,
        comp95 * 100,
        color=OBS,
        lw=1.6,
        marker="o",
        ms=3.4,
        mec=SURF,
        mew=0.4,
        label="… and early − late\n(95% CI)",
    )
    ax.plot(
        comp90.index,
        comp90 * 100,
        color=OBS3,
        lw=1.2,
        ls=(0, (4, 1.5)),
        marker="s",
        ms=2.8,
        label="… and early − late\n(90% CI)",
    )
    ax.axvspan(0.15, 0.212, color="#f6eef0", zorder=-3, lw=0)
    ax.text(
        0.181, 100 * fit.max() * 0.55, "not\ncompatible", ha="center", fontsize=FS * 4.8, color=SELC
    )
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlim(-0.012, 0.212)
    ax.set_ylim(0, 100 * fit.max() * 1.12)
    ax.set_xlabel("True effect of starting (SD)")
    ax.set_ylabel("Scenarios compatible (%)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    ax.legend(
        frameon=False,
        fontsize=FS * 4.4,
        loc="lower left",
        handlelength=1.6,
        labelspacing=0.4,
        borderaxespad=0.1,
    )
    r2 = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.55)
    # e years of decline
    ax = fig.add_subplot(r2[0])
    panel_title(ax, "d", "In years of age-related decline")
    hk = pd.read_csv("results_pool_hk.csv")
    gk = lambda an, y="Y_recall": (lambda r: (r.est, r.hk_lo, r.hk_hi))(
        hk[(hk.analysis == an) & (hk.y == y) & (hk.scale == "RD/MD")].iloc[0]
    )
    items = [
        ("Cross-sectional", *gk("cross-sectional"), "#b3b1aa"),
        ("Target trial", *gk("target trial"), "#8a8880"),
        ("Placebo", *gk("placebo lead 1"), OBS),
        ("Before anyone starts", *gk("pre-trend", "Y_pre"), OBS),
        ("Early vs late", *gk("early vs late"), OBS3),
    ]
    n = len(items)
    ax.axvspan(0, 3.0, color="#efe9f4", zorder=-3, lw=0)
    ax.text(
        1.5, 0.25, "compatible\n(≤0.15 SD)", ha="center", fontsize=FS * 4.5, color=EFF, va="bottom"
    )
    for i, (nm, e, lo, hi, col) in enumerate(items):
        yy = n - i
        ax.barh(yy, e / 0.05, height=0.56, color=col, lw=0)
        ax.plot([lo / 0.05, hi / 0.05], [yy] * 2, color=INK, lw=0.8)
        ax.text(
            max(hi / 0.05, e / 0.05) + 0.25,
            yy,
            f"{e / 0.05:.1f} y",
            va="center",
            fontsize=FS * 5.0,
            color=INK,
        )
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([s_[0] for s_ in items], fontsize=FS * 5.2)
    ax.set_ylim(0.2, n + 0.6)
    ax.set_xlim(-1.5, 11.5)
    ax.set_xlabel("Years (0.05 SD per year)")
    clean(ax)
    ax.grid(False)
    # f triangulated upper bound
    ax = fig.add_subplot(r2[1])
    panel_title(ax, "e", "Bounds on the effect of starting")
    TMP = pd.read_csv("results_tte_minus_placebo.csv")
    tmp = pooled(TMP[(TMP.y == "Y_recall") & TMP.cohort.isin(COH)])
    ESP = pd.read_csv("results_es_pooled.csv")
    esd = ESP[(ESP.y == "z_recall") & (ESP.e == "post_avg_adj")].iloc[0]
    sims = d[d.fit4 & d.EL.between(EL["lo"], EL["hi"])].beta
    rows = [
        ("Early vs late", EL["est"], EL["lo"], EL["hi"], EL["lo90"], EL["hi90"]),
        ("Target trial − placebo", tmp["est"], tmp["lo"], tmp["hi"], None, None),
        ("Event study, after vs trend", esd.est, esd.lo, esd.hi, None, None),
        ("Simulation, compatible effects", None, sims.min(), sims.max(), None, None),
    ]
    n = len(rows)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    for x_, nm, col in [
        (TT["est"], "target trial", "#8a8880"),
        (0.38, "cross-sectional", "#b3b1aa"),
    ]:
        ax.axvline(x_, color=col, lw=1.0, ls=(0, (1, 1.4)))
        ax.text(
            x_ - 0.006, 0.5, nm, fontsize=FS * 4.5, color=col, rotation=90, va="bottom", ha="right"
        )
    for i, (nm, e, lo, hi, l9, h9) in enumerate(rows):
        yy = n - i
        if e is None:
            ax.add_patch(Rectangle((lo, yy - 0.16), hi - lo, 0.32, color=OBS3, alpha=0.55, lw=0))
            ax.text(
                lo + 0.008, yy + 0.32, f"0 to {hi:.2f}", va="bottom", fontsize=FS * 4.8, color=OBS
            )
            continue
        ax.plot([lo, hi], [yy] * 2, color=OBS, lw=1.0)
        if l9 is not None:
            ax.plot([l9, h9], [yy] * 2, color=OBS, lw=2.6, solid_capstyle="butt")
        ax.plot(e, yy, "D", color=OBS, ms=4.2, mec=SURF, mew=0.5, zorder=4)
        ax.text(hi + 0.012, yy, f"≤{hi:.2f}", va="center", fontsize=FS * 4.8, color=OBS)
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([r_[0] for r_ in rows], fontsize=FS * 5.2)
    ax.set_ylim(0.4, n + 0.6)
    ax.set_xlim(-0.15, 0.42)
    ax.set_xlabel("Effect of starting on recall (SD)")
    clean(ax)
    ax.grid(False)
    save(fig, "NH3_Fig8")


def ed_choices(ax, L):
    panel_title(ax, L, "Target trial and placebo across analytic choices")
    pr = lambda design, m: SENS[
        (SENS.design == design) & (SENS.y == "Y_recall") & (SENS.method == m)
    ]
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
    ]
    n = len(specs)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    for i, (nm, a_, b_) in enumerate(specs):
        yy = n - i
        for g_, off, col, mk, fc in [
            (a_, 0.16, "#8a8880", "o", "#8a8880"),
            (b_, -0.16, OBS, "s", SURF),
        ]:
            p = pooled(ci(g_))
            ax.plot([max(p["lo"], -0.05), min(p["hi"], 0.36)], [yy + off] * 2, color=col, lw=1.0)
            ax.plot(p["est"], yy + off, mk, color=fc, mec=col, ms=3.6, mew=0.9, zorder=4)
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([s_[0] for s_ in specs], fontsize=FS * 5.2)
    ax.set_ylim(0.4, n + 1.3)
    ax.set_xlim(-0.05, 0.36)
    ax.set_xlabel("Recall difference (SD)")
    clean(ax)
    ax.grid(False)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color="#8a8880", ms=3.4, label="target trial"),
            Line2D([], [], marker="s", mfc=SURF, mec=OBS, color=OBS, ms=3.4, label="placebo"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=FS * 4.6,
        handlelength=1.3,
        borderaxespad=0.0,
    )


# ================================================================== new Fig. 7  implausible benefits, subgroups, implied cases
def fig7n():
    NP = pd.read_csv("results_negctrl_pooled.csv")
    NA = pd.read_csv("results_negctrl_all.csv")
    SQ = pd.read_csv("results_subgrp_pooled.csv")
    SA = pd.read_csv("results_subgrp_all.csv")
    D1 = pd.read_csv("results_main_hgb_trim0.025_lag_t1_extra.csv")
    D1 = D1[D1.y == "Y_death"]
    DMT = pd.read_csv("results_meta_hgb_trim0.025_lag_t1_extra.csv")
    DMT = DMT[(DMT.y == "Y_death") & (DMT.scale == "RR")].iloc[0]
    fig = plt.figure(figsize=(W, 6.9))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.42, 1.55],
        hspace=0.34,
        left=0.06,
        right=0.985,
        top=0.915,
        bottom=0.06,
    )
    top = gs[0].subgridspec(1, 6, wspace=0.42)
    TTR = pooled(ci(sel(TTE, y="Y_recall")))
    # a-d negative-control outcomes: target trial vs placebo
    spec = [
        ("a", "Death", None, "Risk ratio (log)", (0.5, 1.25)),
        ("b", "Grip strength", "grip", "Difference (SD)", (-0.1, 0.3)),
        ("c", "ADL difficulty", "adl", "Risk diff. (pp)", (-8, 6)),
        ("d", "Chronic disease", "ncnew", "Risk diff. (pp)", (-5, 5)),
        ("e", "Hospital stay", "hosp", "Risk diff. (pp)", (-12, 12)),
        ("f", "Cancer", "cancer", "Risk diff. (pp)", (-3, 3)),
    ]
    for j, (L, t, o, xl, lim) in enumerate(spec):
        ax = fig.add_subplot(top[j])
        panel_title(ax, L, t)
        if o is None:
            g = D1[D1.cohort.isin(COH)]
            rows = [
                (
                    "Target trial",
                    g.assign(est=np.log(g.rr), lo=np.log(g.rr_lo), hi=np.log(g.rr_hi)),
                    True,
                )
            ]
            ref = 1
        else:
            sc = 1 if o == "grip" else 100
            rows = [
                ("Target trial", NA[(NA.outcome == o) & (NA.design == "tte")], False),
                ("Placebo", NA[(NA.outcome == o) & (NA.design == "placebo")], False),
            ]
            ref = 0
        ax.axvline(ref, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        if o == "grip":
            ax.axvspan(TTR["est"] - 0.003, TTR["est"] + 0.003, color=EFF, alpha=0.0)
        for i, (nm, g, rr) in enumerate(rows):
            yy = len(rows) - 1 - i
            for k_, c in enumerate(COH):
                q = g[g.cohort == c]
                if not len(q):
                    continue
                v = np.exp(q.est.iloc[0]) if rr else q.est.iloc[0] * (1 if o == "grip" else 100)
                ax.plot(
                    np.clip(v, *lim), yy - 0.22 + k_ * 0.0, MK[c], color=COL[c], ms=2.4, alpha=0.85
                )
            p = (
                pooled(ci(g))
                if not rr
                else dict(est=np.log(DMT.est), lo=np.log(DMT.lo), hi=np.log(DMT.hi))
            )
            f_ = (lambda v: np.exp(v)) if rr else (lambda v: v * (1 if o == "grip" else 100))
            e, lo, hi = f_(p["est"]), f_(p["lo"]), f_(p["hi"])
            sig = (lo > ref) or (hi < ref)
            ax.plot([max(lo, lim[0]), min(hi, lim[1])], [yy + 0.08] * 2, color=OBS, lw=1.5)
            ax.plot(
                e, yy + 0.08, "D", color=OBS if sig else SURF, mec=OBS, ms=4.6, mew=1.0, zorder=5
            )
            ax.text(
                e,
                yy + 0.36,
                (f"{e:.2f}" if rr else (f"{e:+.2f}" if o == "grip" else f"{e:+.1f}")).replace(
                    "-", "−"
                ),
                ha="center",
                va="bottom",
                fontsize=FS * 4.8,
                color=OBS,
                weight="bold" if sig else "normal",
            )
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(([r[0] for r in rows][::-1] if j in (0, 1) else []), fontsize=FS * 5.0)
        ax.set_ylim(-0.55, len(rows) - 0.45 + (0.3 if len(rows) == 1 else 0))
        ax.set_xlabel(xl, fontsize=FS * 5.4)
        if rr:
            ax.set_xscale("log")
            ax.set_xticks([0.5, 0.75, 1])
            ax.set_xticklabels(["0.5", "0.75", "1"])
            ax.minorticks_off()
        ax.set_xlim(*lim)
        clean(ax)
        ax.grid(False)
        if o == "grip":
            ax.axvline(TTR["est"], color=OBS3, lw=0.8, ls=(0, (1, 1.5)))
            ax.text(
                TTR["est"] + 0.005,
                -0.5,
                "recall\ntarget trial",
                fontsize=FS * 4.0,
                color=OBS3,
                ha="left",
                va="bottom",
            )
    # e subgroups: target trial and placebo (left), difference (right)
    bot = gs[1].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.42)
    eb = bot[0].subgridspec(1, 3, width_ratios=[0.95, 1.0, 0.85], wspace=0.06)
    order = [
        ("Overall", "Overall"),
        ("Age 50-64", "Age 50–64"),
        ("Age 65-74", "Age 65–74"),
        ("Age 75+", "Age ≥75"),
        ("Men", "Men"),
        ("Women", "Women"),
        ("Education low", "Education low"),
        ("Education mid", "Education middle"),
        ("Education high", "Education high"),
        ("Wealth Q1-2", "Wealth Q1–2"),
        ("Wealth Q3-5", "Wealth Q3–5"),
        ("Lives alone", "Lives alone"),
        ("Lives with others", "Lives with others"),
        ("Spouse offline/none", "Spouse offline or none"),
        ("Spouse online", "Spouse online"),
        ("Rural", "Rural"),
        ("Urban", "Urban"),
        ("Fair/poor health", "Fair/poor health"),
        ("Good+ health", "Good health"),
        ("Elevated depressive symptoms", "Depressive symptoms"),
        ("No elevated symptoms", "No depressive symptoms"),
        ("Never used before", "Never used before"),
        ("Used before (returner)", "Used before"),
    ]
    order = [o_ for o_ in order if o_[0] in set(SQ.subgroup)]
    n = len(order)
    axl = fig.add_subplot(eb[0])
    axl.axis("off")
    ax1 = fig.add_subplot(eb[1])
    ax2 = fig.add_subplot(eb[2], sharey=ax1)
    lab(ax1, "g", x=-0.98, y=1.02, title="Selection-corrected estimates by subgroup")
    for ax in (ax1, ax2):
        ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax2.axvspan(-0.10, 0.10, color="#efeee9", zorder=-3, lw=0)
    for i, (k, nm) in enumerate(order):
        yy = n - 1 - i
        q = SQ[SQ.subgroup == k].iloc[0]
        if i in (1, 4, 6, 9, 11, 13, 15, 17, 19, 21):
            [a.axhline(yy + 0.5, color=GRID, lw=0.5) for a in (ax1, ax2)]
        ax1.plot([q.tte_lo, q.tte_hi], [yy + 0.15] * 2, color="#8a8880", lw=1.0)
        ax1.plot(q.tte, yy + 0.15, "o", color="#8a8880", ms=3.2, mec=SURF, mew=0.4)
        ax1.plot([q.pla_lo, q.pla_hi], [yy - 0.15] * 2, color="#b9b7b0", lw=1.0)
        ax1.plot(q.pla, yy - 0.15, "s", color=SURF, mec="#8a8880", ms=3.0, mew=0.8)
        sig = q["diff_lo"] > 0 or q["diff_hi"] < 0
        ax2.plot([max(q["diff_lo"], -0.4), min(q["diff_hi"], 0.4)], [yy] * 2, color=OBS, lw=1.3)
        if q["diff_hi"] > 0.4:
            ax2.plot(0.4, yy, ">", color=OBS, ms=2.6)
        if q["diff_lo"] < -0.4:
            ax2.plot(-0.4, yy, "<", color=OBS, ms=2.6)
        ax2.plot(q["diff"], yy, "D", color=OBS if sig else SURF, mec=OBS, ms=3.8, mew=0.9, zorder=4)
        axl.text(
            1.0,
            yy,
            nm,
            ha="right",
            va="center",
            fontsize=FS * 5.2,
            transform=axl.get_yaxis_transform(),
            weight="bold" if k == "Overall" else "normal",
        )
    axl.set_ylim(-0.6, n - 0.4)
    for ax in (ax1, ax2):
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks([])
        clean(ax)
        ax.grid(False)
    ax1.set_xlim(-0.2, 0.55)
    ax1.set_xlabel("Recall (SD)")
    ax2.set_xlim(-0.42, 0.42)
    ax2.set_xlabel("Difference (SD)")
    ax2.set_title("target trial − placebo", fontsize=FS * 5.0, color=OBS, pad=2)
    ax1.legend(
        handles=[
            Line2D([], [], color="#8a8880", marker="o", ms=3.2, lw=1, label="target trial"),
            Line2D(
                [],
                [],
                color="#b9b7b0",
                marker="s",
                mfc=SURF,
                mec="#8a8880",
                ms=3,
                lw=1,
                label="placebo",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        fontsize=FS * 4.6,
        handlelength=1.2,
        columnspacing=0.8,
        borderaxespad=0,
    )
    axl.set_xlim(0, 1)
    # f implied cases prevented per 1,000 people going online
    br = bot[1].subgridspec(3, 1, height_ratios=[0.12, 1.25, 0.55], hspace=0.0)
    ax = fig.add_subplot(br[1])
    panel_title(ax, "h", "Implied cases prevented per 1,000 starters")
    NO = pd.read_csv("results_newout_all.csv")
    DM = pd.read_csv("results_dementia_HRS.csv")
    get = lambda src, o, d: (
        lambda g: (
            g.est.iloc[0],
            g.se.iloc[0] if "se" in g else (g.hi.iloc[0] - g.lo.iloc[0]) / 3.92,
        )
    )(src[(src.outcome == o) & (src.design == d) & (src.cohort == "HRS")])
    groups = [("Cognitive impairment", NO, "cimp"), ("Dementia", DM, "dem")]
    ests = [
        ("Target trial", lambda t, p, e: (t[0], t[1])),
        ("Target trial − placebo", lambda t, p, e: (t[0] - p[0], np.hypot(t[1], p[1]))),
        ("Early vs late", lambda t, p, e: e),
    ]
    cols_ = ["#8a8880", OBS, OBS3]
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    yt, yl = [], []
    for gi, (gn, src, o) in enumerate(groups):
        t = get(src, o, "tte")
        p = get(src, o, "placebo")
        e = get(src, o, "early_vs_late")
        for k_, (nm, fn) in enumerate(ests):
            est, se = fn(t, p, e)
            v, lo, hi = -est * 1000, -(est + 1.96 * se) * 1000, -(est - 1.96 * se) * 1000
            yy = (len(groups) - 1 - gi) * 4.0 + (2 - k_) * 0.9
            ax.plot([lo, hi], [yy, yy], color=cols_[k_], lw=1.6)
            ax.plot(v, yy, "o", color=cols_[k_], ms=4.8, mec=SURF, mew=0.6, zorder=4)
            ax.text(
                1.0,
                yy,
                f"{v:+.0f}".replace("-", "−"),
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=FS * 5.0,
                color=cols_[k_],
                weight="bold" if k_ == 0 else "normal",
            )
            yt.append(yy)
            yl.append(nm)
        ax.text(
            0.0,
            (len(groups) - 1 - gi) * 4.0 + 2.45,
            gn + " (HRS)",
            transform=ax.get_yaxis_transform(),
            fontsize=FS * 5.4,
            weight="bold",
            color=INK,
            va="bottom",
            ha="right",
        )
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=FS * 5.0)
    ax.set_ylim(-0.6, 7.3)
    ax.set_xlim(-50, 66)
    ax.set_xlabel("Cases prevented per 1,000 people going online")
    clean(ax)
    ax.grid(False)
    cohort_legend(fig, y=0.955)
    save(fig, "NH3_Fig7")


def ed_subgroup_z(ax, L):
    panel_title(ax, L, "Subgroup differences vs chance")
    SQ = pd.read_csv("results_subgrp_pooled.csv")
    order = [(k, k) for k in SQ.subgroup if k != "Overall"]
    z = []
    for k, nm in order:
        if k == "Overall":
            continue
        q = SQ[SQ.subgroup == k].iloc[0]
        se = (q["diff_hi"] - q["diff_lo"]) / 3.92
        z.append(q["diff"] / se)
    z = np.sort(np.array(z))
    th = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    ax.fill_between(
        [-2.6, 2.6], [-2.6 - 1.0, 2.6 - 1.0], [-2.6 + 1.0, 2.6 + 1.0], color="#efeee9", lw=0
    )
    ax.plot([-2.6, 2.6], [-2.6, 2.6], color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.plot(th, z, "D", color=OBS, ms=3.4, mec=SURF, mew=0.4)
    ax.axhspan(1.96, 3, color=SELC, alpha=0.06, lw=0)
    ax.axhspan(-3, -1.96, color=SELC, alpha=0.06, lw=0)
    ax.text(
        0.02,
        0.97,
        f"{len(z)} subgroups; |z| > 1.96: {(np.abs(z) > 1.96).sum()}",
        transform=ax.transAxes,
        va="top",
        fontsize=FS * 4.8,
        color=INK2,
    )
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("Expected z under no differences")
    ax.set_ylabel("Observed z")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)


# ================================================================== Fig. 6  who goes online, with world map
import json
import nhfig as NH

ISO = {
    "USA (HRS)": "USA",
    "England": "GBR",
    "China (CHARLS)": "CHN",
    "India": "IND",
    "Austria": "AUT",
    "Germany": "DEU",
    "Sweden": "SWE",
    "Spain": "ESP",
    "Italy": "ITA",
    "France": "FRA",
    "Denmark": "DNK",
    "Greece": "GRC",
    "Switzerland": "CHE",
    "Belgium": "BEL",
    "Israel": "ISR",
    "Czech Republic": "CZE",
    "Poland": "POL",
    "Luxembourg": "LUX",
    "Slovenia": "SVN",
    "Estonia": "EST",
    "Croatia": "HRV",
}


def equal_earth(lon, lat):
    A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
    M = np.sqrt(3) / 2
    lam, phi = np.radians(lon), np.radians(lat)
    th = np.arcsin(M * np.sin(phi))
    x = (
        2
        * np.sqrt(3)
        * lam
        * np.cos(th)
        / (3 * (9 * A4 * th**8 + 7 * A3 * th**6 + 3 * A2 * th**2 + A1))
    )
    y = A4 * th**9 + A3 * th**7 + A2 * th**3 + A1 * th
    return x, y


def draw_map(ax, values, cmap, norm, proj=equal_earth, bbox=None, lw=0.25, label_codes=None):
    G = json.load(open("world110.geojson"))
    for f in G["features"]:
        code = f["properties"]["ADM0_A3"]
        if code == "ATA":
            continue
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        v = values.get("CHN" if code == "TWN" and "CHN" in values else code)
        fc = (
            cmap(norm(v))
            if v is not None and np.isfinite(v)
            else ("#e9e7e1" if v is None else "#cfcdc6")
        )
        for poly in polys:
            ring = np.array(poly[0])
            if bbox is not None:
                if (
                    ring[:, 0].max() < bbox[0]
                    or ring[:, 0].min() > bbox[1]
                    or ring[:, 1].max() < bbox[2]
                    or ring[:, 1].min() > bbox[3]
                ):
                    continue
            x, y = proj(ring[:, 0], ring[:, 1])
            ax.fill(x, y, fc=fc, ec=SURF, lw=lw, zorder=2 if v is not None else 1)
    ax.set_aspect("equal")
    ax.axis("off")


def fig6():
    AR, AP, GR = NH.AR, NH.AP, NH.GR
    SP_ = pd.read_csv("results_stop_pred.csv")
    SPP = pd.read_csv("results_stop_pred_pooled.csv")
    RB = pd.read_csv("results_robust_sel.csv")
    RBP = pd.read_csv("results_robust_sel_pooled.csv")
    fig = plt.figure(figsize=(W, 7.9))
    gs0 = fig.add_gridspec(
        2, 1, height_ratios=[1.12, 1.55], hspace=0.3, left=0.07, right=0.985, top=0.925, bottom=0.06
    )
    g1 = gs0[0].subgridspec(1, 3, width_ratios=[0.8, 1.08, 0.58], wspace=0.1)
    g2 = gs0[1].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.42)
    g2l = g2[0].subgridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.62)
    # a probability of starting by recall quintile
    ga = g1[0].subgridspec(1, 2, width_ratios=[1, 0.05], wspace=0)
    ax = fig.add_subplot(ga[0])
    panel_title(ax, "a", "Starting by recall quintile")
    pq = AR[AR.part == "prob_quintile"]
    for c in COH:
        v = pq[pq.cohort == c].sort_values("term").est.values * 100
        ax.plot(range(1, 6), v, color=COL[c], marker=MK[c], ms=3.4, lw=1.2, mec=SURF, mew=0.4)
        ax.text(
            5.18,
            v[-1] + {"SHARE": 0.9, "ELSA": -0.7}.get(c, 0),
            c,
            fontsize=FS * 5.2,
            color=COL[c],
            va="center",
        )
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"], fontsize=FS * 5.6)
    ax.set_xlim(0.8, 6.3)
    ax.set_ylim(0, 21)
    ax.set_ylabel("Started by next wave (%)")
    ax.set_xlabel("Recall quintile at t0")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    q5 = AP[(AP.part == "quintile") & (AP.term == "Q5")].iloc[0]
    ax.text(
        0.03,
        0.97,
        f"Q5 vs Q1\nOR {np.exp(q5.est):.1f} ({np.exp(q5.lo):.1f}–{np.exp(q5.hi):.1f})",
        transform=ax.transAxes,
        fontsize=FS * 5.0,
        va="top",
        color=INK2,
    )
    # b, c predictors of starting and of stopping (shared rows)
    terms = [
        ("lvl", "Memory, two-wave mean (per SD)"),
        ("drz", "Recent memory change (per SD)"),
        ("lvl3", "Memory, three-wave mean (per SD)"),
        ("slope3", "Memory, three-wave slope (per SD)"),
        ("edu", "Education (per SD)"),
        ("wealth_q", "Wealth (per SD)"),
        ("sp_internet", "Spouse uses internet"),
        ("partnered", "Partnered"),
        ("work", "In paid work"),
        ("live_alone", "Lives alone"),
        ("female", "Women"),
        ("ncond", "Chronic conditions (per SD)"),
        ("dep", "Depressive symptoms (per SD)"),
        ("srh_poor", "Fair/poor health"),
        ("iadl_any", "Any IADL difficulty"),
        ("age", "Age (per 10 years)"),
    ]
    n = len(terms)

    def orplot(ax, getp, getc, xlim, ticks, show_labels):
        ax.axvline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        for i, (t, nm) in enumerate(terms):
            yy = n - i
            if t in ("lvl", "drz", "lvl3", "slope3"):
                ax.axhspan(
                    yy - 0.45,
                    yy + 0.45,
                    color="#eef4f5" if t in ("lvl", "drz") else "#e3eef0",
                    zorder=-3,
                    lw=0,
                )
            if t == "slope3":
                ax.axhline(yy - 0.5, color=GRID, lw=0.8)
            p = getp(t)
            if p is None:
                continue
            e, lo, hi = np.exp(p[0]), np.exp(p[1]), np.exp(p[2])
            sig = lo > 1 or hi < 1
            for j, c in enumerate(COH):
                x = getc(t, c)
                if x is not None:
                    ax.plot(
                        np.clip(np.exp(x), *xlim),
                        yy - 0.28,
                        MK[c],
                        color=COL[c],
                        ms=2.0,
                        alpha=0.85,
                    )
            ax.plot([max(lo, xlim[0]), min(hi, xlim[1])], [yy, yy], color=OBS, lw=1.3)
            ax.plot(e, yy, "D", color=OBS if sig else SURF, mec=OBS, ms=4.0, mew=1.0, zorder=4)
            ax.text(
                1.0,
                yy,
                f"{e:.2f}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=FS * 5.0,
                color=OBS if sig else MUTED,
                weight="bold" if sig else "normal",
            )
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:g}" for v in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([n - i for i in range(n)])
        ax.set_yticklabels([t[1] for t in terms] if show_labels else [], fontsize=FS * 5.4)
        ax.set_ylim(0.4, n + 0.6)
        clean(ax)
        ax.set_xlabel("Odds ratio (log scale)")

    R3 = lambda mo, t: RBP[(RBP.model == mo) & (RBP.term == t)]
    gb = lambda t: (
        (lambda x: (x.est.iloc[0], x.lo.iloc[0], x.hi.iloc[0]))(R3("start", t))
        if t in ("lvl3", "slope3")
        else (lambda p: (p.est, p.lo, p.hi))(AP[(AP.part == "predictor") & (AP.term == t)].iloc[0])
    )
    gbc = lambda t, c: (lambda x: x.est.iloc[0] if len(x) else None)(
        RB[(RB.model == "start") & (RB.term == t) & (RB.cohort == c)]
        if t in ("lvl3", "slope3")
        else AR[(AR.part == "predictor") & (AR.term == t) & (AR.cohort == c)]
    )
    gc = lambda t: (lambda x: (x.est.iloc[0], x.lo.iloc[0], x.hi.iloc[0]) if len(x) else None)(
        R3("stop", t) if t in ("lvl3", "slope3") else SPP[SPP.term == t]
    )
    gcc = lambda t, c: (lambda x: x.est.iloc[0] if len(x) else None)(
        RB[(RB.model == "stop") & (RB.term == t) & (RB.cohort == c)]
        if t in ("lvl3", "slope3")
        else SP_[(SP_.term == t) & (SP_.cohort == c)]
    )
    gbb = g1[1].subgridspec(1, 2, width_ratios=[0.92, 1.0], wspace=0.0)
    axl = fig.add_subplot(gbb[0])
    axl.axis("off")
    ax = fig.add_subplot(gbb[1])
    orplot(ax, gb, gbc, (0.35, 3.0), [0.5, 1, 2], False)
    lab(ax, "b", x=-0.02, title="Predictors of starting")
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([t[1] for t in terms], fontsize=FS * 5.4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Odds ratio (log scale)\n ", fontsize=FS * 5.4)
    ax = fig.add_subplot(g1[2])
    orplot(ax, gc, gcc, (0.35, 3.0), [0.5, 1, 2], False)
    lab(ax, "c", x=-0.02, title="Predictors of stopping")
    for t, yy in [("lvl3", n - 2), ("slope3", n - 3)]:
        q = R3("stop2", t).iloc[0]
        ax.plot([np.exp(q.lo), np.exp(q.hi)], [yy + 0.3] * 2, color=OBS3, lw=0.9)
        ax.plot(np.exp(q.est), yy + 0.3, "^", color=SURF, mec=OBS3, ms=3.6, mew=0.9, zorder=4)
    ax.set_xlabel("Odds ratio (log scale)\ntriangles: offline two waves running", fontsize=FS * 5.4)
    # d decomposition (waterfall) with target-trial and placebo estimates
    ax = fig.add_subplot(g2l[0])
    panel_title(ax, "d", "Decomposition of the gap at t0")
    gp = AP[AP.part == "gelbach"].set_index("term")
    order = [
        ("Age and sex", "Age, sex"),
        ("Education, wealth and work", "Socioeconomic"),
        ("Health", "Health"),
        ("Household and spouse", "Household"),
        ("Lifestyle", "Lifestyle"),
        ("Earlier memory (t−1)", "Memory at t−1"),
        ("unexplained", "Not explained"),
    ]
    top = 0
    for i, (k, nm) in enumerate(order):
        v = gp.loc[k, "est"]
        col = OBS if k == "unexplained" else "#b9b7b0"
        ax.bar(i, v, bottom=top, color=col, width=0.72, lw=0)
        if i < len(order) - 1:
            ax.plot([i + 0.36, i + 0.64], [top + v] * 2, color=MUTED, lw=0.5)
        ax.text(
            i,
            top + v + 0.012,
            f"{v:.2f}",
            ha="center",
            fontsize=FS * 5.0,
            color=OBS if k == "unexplained" else INK2,
            weight="bold" if k == "unexplained" else "normal",
        )
        top += v
    base = top - gp.loc["unexplained", "est"]
    tt = pooled(ci(sel(TTE, y="Y_recall")))
    pl = pooled(ci(sel(PLA, y="Y_recall")))
    for x_, p_, nm in [(7.15, tt, "target\ntrial"), (8.05, pl, "placebo")]:
        ax.bar(x_, p_["est"], bottom=base, color=SURF, ec=OBS, lw=0.9, width=0.62, hatch="////")
        ax.text(
            x_,
            base + p_["est"] + 0.012,
            f"{p_['est']:.2f}",
            ha="center",
            fontsize=FS * 5.0,
            color=OBS,
        )
    su = (
        (gp.loc["unexplained", "hi"] - gp.loc["unexplained", "lo"]) / 3.92 if "hi" in gp else np.nan
    )
    st = (tt["hi"] - tt["lo"]) / 3.92
    dd = gp.loc["unexplained", "est"] - tt["est"]
    sd_ = np.hypot(su, st)
    ax.text(
        7.6,
        0.33,
        f"not explained − target trial\n{dd:+.2f} ({dd - 1.96 * sd_:+.2f} to {dd + 1.96 * sd_:+.2f})".replace(
            "-", "−"
        ),
        ha="center",
        va="top",
        fontsize=FS * 4.5,
        color=OBS,
    )
    ax.axhline(gp.loc["total", "est"], color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.text(
        -0.45,
        gp.loc["total", "est"] + 0.012,
        f"total gap {gp.loc['total', 'est']:.2f} SD",
        fontsize=FS * 5.2,
        color=INK2,
    )
    ax.set_xticks(list(range(len(order))) + [7.15, 8.05])
    ax.set_xticklabels(
        [o[1] for o in order] + ["Target trial", "Placebo"],
        fontsize=FS * 5.0,
        rotation=40,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.6, 8.5)
    ax.set_ylim(0, 0.66)
    ax.set_ylabel("Recall gap (SD)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    # f cross-country meta-regression, gap at t0 and target trial
    ax = fig.add_subplot(g2l[1])
    panel_title(ax, "f", "Cross-country gradient")
    tk = [0.05, 0.25, 0.5, 0.75]
    xs = np.linspace(np.log(0.03 / 0.97), np.log(0.85 / 0.15), 50)
    for dsg, col, mk_, off, nm in [
        ("pre", OBS, "o", -0.04, "gap at t0"),
        ("tte", "#8a8880", "s", 0.04, "target trial"),
    ]:
        d = GR[GR.design == dsg].copy()
        d["lp"] = np.log(d.prev / (1 - d.prev))
        m_ = F.metareg(d)
        for _, r in d.iterrows():
            lo, hi = r.est - 1.96 * r.se, r.est + 1.96 * r.se
            ax.scatter(
                r.lp + off,
                np.clip(r.est, -0.3, 0.7),
                s=np.clip(1 / r.se**2 / 40, 4, 55),
                marker=mk_,
                facecolor=col if dsg == "pre" else SURF,
                edgecolor=col,
                lw=0.7,
                zorder=4,
            )
        ax.plot(
            xs,
            m_["intercept"] + m_["slope"] * xs,
            color=col,
            lw=1.4,
            ls="-" if dsg == "pre" else (0, (4, 1.6)),
        )
        if dsg == "pre":
            m2 = F.metareg(d[d.unit != "CHARLS"])
            x2 = xs[xs > np.log(0.2 / 0.8)]
            ax.plot(x2, m2["intercept"] + m2["slope"] * x2, color=OBS3, lw=1.1, ls=(0, (1, 1.2)))
            ax.text(
                0.02,
                0.095,
                f"without China: slope {m2['slope']:+.3f}, P = {m2['p']:.2f}".replace("-", "−"),
                transform=ax.transAxes,
                fontsize=FS * 4.7,
                color=OBS3,
                ha="left",
                va="bottom",
                bbox=dict(fc=SURF, ec="none", pad=0.3, alpha=0.85),
            )
        ax.text(
            0.02,
            0.165 if dsg == "pre" else 0.025,
            f"{nm}: slope {m_['slope']:+.3f}, P = {(f"{m_['p']:.2f}" if m_['p'] >= 0.01 else f"{m_['p']:.3f}")}".replace(
                "-", "−"
            ),
            transform=ax.transAxes,
            fontsize=FS * 4.7,
            color=col,
            ha="left",
            va="bottom",
            weight="bold" if dsg == "pre" else "normal",
            bbox=dict(fc=SURF, ec="none", pad=0.3, alpha=0.85),
        )
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(np.log(np.array(tk) / (1 - np.array(tk))))
    ax.set_xticklabels([f"{int(t * 100)}%" for t in tk])
    ax.set_xlim(xs[0] - 0.2, xs[-1] + 0.2)
    ax.set_ylim(-0.3, 0.7)
    ax.set_xlabel("Internet use among adults ≥50")
    ax.set_ylabel("Recall difference (SD)")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    # e country forest, sorted by prevalence
    ax = fig.add_subplot(g2[1])
    lab(ax, "e", x=-0.02, y=1.055, title="Pre-adoption advantage by country")
    pre = GR[GR.design == "pre"].copy()
    pre["name"] = pre.unit.map(NH.uname)
    pre = pre.sort_values("prev")
    tte_ = GR[GR.design == "tte"].set_index("unit")
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    for i, (_, r) in enumerate(pre.iterrows()):
        yy = len(pre) - 1 - i
        col = COL.get(r.cohort, COL["SHARE"])
        if r.unit in tte_.index:
            q = tte_.loc[r.unit]
            lo_, hi_ = q.est - 1.96 * q.se, q.est + 1.96 * q.se
            ax.plot([max(lo_, -0.58), min(hi_, 0.88)], [yy - 0.22] * 2, color="#cfcdc6", lw=0.8)
            if hi_ > 0.88:
                ax.plot(0.88, yy - 0.22, ">", color="#bdbbb4", ms=2.4)
            if lo_ < -0.58:
                ax.plot(-0.58, yy - 0.22, "<", color="#bdbbb4", ms=2.4)
            ax.plot(
                np.clip(q.est, -0.58, 0.88),
                yy - 0.22,
                "s",
                color=SURF,
                mec="#9c9a93",
                ms=2.8,
                mew=0.7,
                zorder=3,
            )
        ax.plot([r.est - 1.96 * r.se, r.est + 1.96 * r.se], [yy + 0.12] * 2, color=OBS, lw=1.1)
        ax.plot(r.est, yy + 0.12, "o", color=OBS, ms=4.0, mec=SURF, mew=0.5, zorder=4)
        ax.text(
            1.0,
            yy,
            f"{r.prev * 100:.0f}%",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=FS * 5.0,
            color=INK2,
        )
    ax.text(
        1.0,
        len(pre) - 0.2,
        "online",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=FS * 4.8,
        color=INK2,
    )
    names = [n_.replace(" (HRS)", " (HRS)").replace(" (CHARLS)", " (CHARLS)") for n_ in pre.name]
    ax.set_yticks(range(len(pre)))
    ax.set_yticklabels(names[::-1], fontsize=FS * 5.2)
    for t_, (_, r) in zip(ax.get_yticklabels(), pre.iloc[::-1].iterrows()):
        if r.cohort != "SHARE":
            t_.set_color(COL[r.cohort])
            t_.set_weight("bold")
    ax.set_ylim(-0.7, len(pre) - 0.1)
    ax.set_xlim(-0.6, 0.9)
    ax.set_xlabel("Recall difference (SD)")
    clean(ax)
    ax.grid(False)
    ax.legend(
        handles=[
            Line2D([], [], color=OBS, marker="o", ms=4, lw=1.1, label="at t0, before starting"),
            Line2D(
                [], [], color="#8a8880", marker="s", mfc=SURF, ms=3, lw=0.8, label="target trial"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.45, 0.995),
        ncol=2,
        frameon=False,
        fontsize=FS * 5.0,
        handlelength=1.4,
    )
    cohort_legend(fig, y=0.958)
    save(fig, "NH2_Fig6")


# ================================================================== Extended Data Figs 1-2 (panels moved from the main figures)
def ed1():
    fig = plt.figure(figsize=(W, 5.6))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[0.9, 0.85, 1.2],
        wspace=0.75,
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.08,
    )
    c0 = gs[0].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.65)
    ed_attn(fig.add_subplot(c0[0]), "a")
    ed_subgroup_z(fig.add_subplot(c0[1]), "b")
    ed_subgroups(fig.add_subplot(gs[1]), "c")
    ed_spec_tte(fig, gs[2], "d")
    save(fig, "NH3_EDFig1")


def ed2():
    fig = plt.figure(figsize=(W, 5.3))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[0.8, 1.15], hspace=0.5, left=0.075, right=0.985, top=0.93, bottom=0.08
    )
    r1 = gs[0].subgridspec(1, 4, wspace=0.12)
    ed_cohort_traj(fig, [r1[k] for k in range(4)], "abcd")
    r2 = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.45)
    ed_choices(fig.add_subplot(r2[0]), "e")
    ed_pretrend_sens(fig.add_subplot(r2[1]), "f")
    h = [
        Line2D([], [], color=OBS, marker="o", ms=3, lw=1.4, label="will start at t1 (initiators)"),
        Line2D([], [], color=GREY, marker="o", ms=3, lw=1.1, label="stay offline"),
    ]
    fig.legend(
        handles=h,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        fontsize=FS * 6.0,
    )
    save(fig, "NH3_EDFig2")


def ed4():
    fig7()


if __name__ == "__main__":
    import sys

    for f in sys.argv[1:]:
        globals()[f]()
