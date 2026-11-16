import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52524e", "#8a8a85", "#e6e5e0", "#ffffff"
COL = {"HRS": "#2a78d6", "ELSA": "#eb6834", "SHARE": "#1baf7a", "CHARLS": "#eda100", "Pooled": INK}
MK = {"HRS": "o", "ELSA": "s", "SHARE": "^", "CHARLS": "D", "Pooled": "D"}
NAMES = {
    "HRS": "HRS (USA)",
    "ELSA": "ELSA (England)",
    "SHARE": "SHARE (Europe)",
    "CHARLS": "CHARLS (China)",
    "Pooled": "Pooled (random effects)",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK,
        "axes.linewidth": 0.6,
        "savefig.dpi": 300,
    }
)
COH = ["HRS", "ELSA", "SHARE", "CHARLS"]


def style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f"fig/{name}.png", bbox_inches="tight", facecolor=SURF)
    fig.savefig(f"fig/{name}.pdf", bbox_inches="tight", facecolor=SURF)


# ---------------- Figure 2: forest plot, primary model ----------------
def fig2():
    byc = pd.read_csv("results_main_hgb_trim0.025_lag_extra.csv")
    met = pd.read_csv("results_meta_hgb_trim0.025_lag_extra.csv")
    panels = [
        ("Y_recall", "a  Memory (word recall)", "Difference in SD units", False, 0),
        ("Y_iadl", "b  New IADL difficulty", "Risk ratio", True, 1),
        ("Y_dep", "c  Elevated depressive symptoms", "Risk ratio", True, 1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
    rows = COH + ["Pooled"]
    ypos = {r: len(rows) - i for i, r in enumerate(rows)}
    ypos["Pooled"] = 0
    for ax, (y, title, xl, rr, ref) in zip(axes, panels):
        for c in COH:
            r = byc[(byc.cohort == c) & (byc.y == y)].iloc[0]
            e, lo, hi = (r.rr, r.rr_lo, r.rr_hi) if rr else (r.est, r.lo, r.hi)
            ax.plot([lo, hi], [ypos[c]] * 2, color=COL[c], lw=2, solid_capstyle="round")
            ax.plot(e, ypos[c], MK[c], color=COL[c], ms=7, mec=SURF, mew=1.2)
            ax.text(
                1.02,
                ypos[c],
                f"{e:.2f} ({lo:.2f}–{hi:.2f})",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=7.5,
                color=INK2,
            )
        m = met[(met.y == y) & (met.scale == ("RR" if rr else "RD/MD"))].iloc[0]
        e, lo, hi = m.est, m.lo, m.hi
        ax.fill([lo, e, hi, e], [0, 0.28, 0, -0.28], color=INK)
        ax.text(
            1.02,
            0,
            f"{e:.2f} ({lo:.2f}–{hi:.2f})",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=7.5,
            color=INK,
            weight="bold",
        )
        ax.text(
            1.02,
            -0.85,
            f"I² = {100*m.I2:.0f}%",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=7,
            color=MUTED,
        )
        ax.axvline(ref, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        if rr:
            ax.set_xscale("log")
            ax.set_xticks([0.6, 0.8, 1.0, 1.25])
            ax.set_xticklabels(["0.6", "0.8", "1.0", "1.25"])
            ax.set_xlim(0.55, 1.4)
        else:
            ax.set_xlim(-0.1, 0.4)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, weight="bold")
        ax.set_xlabel(xl)
        style(ax)
        ax.set_ylim(-1.2, len(rows) - 0.4)
    axes[0].set_yticks([ypos[r] for r in rows])
    axes[0].set_yticklabels([NAMES[r] for r in rows])
    axes[0].get_yticklabels()[-1].set_weight("bold")
    fig.text(
        0.13,
        -0.08,
        "Favours non-initiation ←  → favours initiation (panel a);  favours initiation ←  → favours non-initiation (panels b, c)",
        fontsize=7,
        color=MUTED,
    )
    fig.subplots_adjust(wspace=0.55)
    save(fig, "Fig2_forest_primary")


# ---------------- Figure 3: adjustment ladder ----------------
def fig3():
    models = [
        ("hgb_trim0.025", "Demographics + baseline health"),
        ("hgb_trim0.025_lag", "+ prior-wave health trajectory"),
        (
            "hgb_trim0.025_lag_extra",
            "+ spouse use, hospital, grip, BMI, loneliness, social activity, contact (primary)",
        ),
        ("hgb_trim0.025_lag_t1_extra", "+ health at exposure ascertainment (conservative bound)"),
    ]
    shades = ["#9ec5f4", "#6da7ec", "#2a78d6", "#104281"]
    outs = [
        ("Y_recall", "Memory (SD)", False),
        ("Y_iadl", "New IADL difficulty (RR)", True),
        ("Y_dep", "Elevated depressive symptoms (RR)", True),
        ("Y_cancer", "New cancer (RR)\nnegative control: expected null", True),
        ("Y_death", "All-cause death (RR)\nbenchmark for residual confounding", True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[1, 2.2]))
    for ax, sel in [(axes[0], outs[:1]), (axes[1], outs[1:])]:
        for j, (y, lab, rr) in enumerate(sel):
            for k, (tag, mlab) in enumerate(models):
                m = pd.read_csv(f"results_meta_{tag}.csv")
                r = m[(m.y == y) & (m.scale == ("RR" if rr else "RD/MD"))].iloc[0]
                yy = (len(sel) - j) - 0.18 * (k - 1.5)
                ax.plot([r.lo, r.hi], [yy] * 2, color=shades[k], lw=2, solid_capstyle="round")
                ax.plot(
                    r.est,
                    yy,
                    "o",
                    color=shades[k],
                    ms=6,
                    mec=SURF,
                    mew=1,
                    label=mlab if j == 0 else None,
                )
        ax.set_yticks([len(sel) - j for j in range(len(sel))])
        ax.set_yticklabels([s[1] for s in sel])
        rrp = sel[0][2]
        ax.axvline(1 if rrp else 0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        if rrp:
            ax.set_xscale("log")
            ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.25])
            ax.set_xticklabels(["0.4", "0.6", "0.8", "1.0", "1.25"])
            ax.set_xlabel("Risk ratio (log scale)")
        else:
            ax.set_xlim(0, 0.22)
            ax.set_xlabel("Difference (SD units)")
        style(ax)
        ax.set_ylim(0.4, len(sel) + 0.6)
    axes[1].axhspan(0.5, 1.5, color="#f3f2ee", zorder=-1)
    axes[1].axhspan(1.5, 2.5, color="#f8f8f5", zorder=-1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12), fontsize=8
    )
    fig.subplots_adjust(wspace=0.75)
    save(fig, "Fig3_adjustment_ladder")


# ---------------- Figure 4: subgroup effects ----------------
def fig4():
    p = pd.read_csv("hte_subgroups_pooled.csv")
    order = [
        "Age 50-64",
        "Age 65-74",
        "Age 75+",
        "Men",
        "Women",
        "Education low",
        "Education mid",
        "Education high",
        "Wealth Q1-2",
        "Wealth Q3-5",
        "Lives alone",
        "Lives with others",
        "Rural",
        "Urban",
        "Good+ health",
        "Fair/poor health",
        "No elevated symptoms",
        "Elevated depressive symptoms",
        "Spouse online",
        "Spouse offline/none",
        "Never used before",
        "Used before (returner)",
    ]
    groups = [3, 2, 3, 2, 2, 2, 2, 2, 2, 2]
    met = pd.read_csv("results_meta_hgb_trim0.025_lag_extra.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 6.2), sharey=True)
    ys, y0, gi = {}, 0, 0
    for g in groups:
        for i in range(g):
            ys[order[gi + i]] = -(y0 + i)
        y0 += g + 0.8
        gi += g
    for ax, (y, title, xl, scale) in zip(
        axes,
        [
            ("Y_recall", "a  Memory", "Difference (SD units)", 1),
            ("Y_iadl", "b  New IADL difficulty", "Risk difference (percentage points)", 100),
        ],
    ):
        ov = met[(met.y == y) & (met.scale == "RD/MD")].iloc[0].est * scale
        ax.axvspan(ov - 0.0001, ov + 0.0001, color="#2a78d6")
        ax.axvline(ov, color="#2a78d6", lw=1, alpha=0.5)
        ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        for s in order:
            r = p[(p.y == y) & (p.subgroup == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot(
                [r.lo * scale, r.hi * scale],
                [ys[s]] * 2,
                color=INK2,
                lw=1.6,
                solid_capstyle="round",
            )
            ax.plot(r.est * scale, ys[s], "s", color=INK, ms=5, mec=SURF, mew=1)
        ax.set_title(title, loc="left", fontsize=9.5, weight="bold")
        ax.set_xlabel(xl)
        style(ax)
        ax.text(ov, 0.9, "overall", color="#2a78d6", fontsize=7, ha="center")
    axes[0].set_yticks(list(ys.values()))
    axes[0].set_yticklabels(list(ys.keys()))
    axes[0].set_ylim(min(ys.values()) - 0.8, 1.4)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "Fig4_subgroups")


# ---------------- Figure 1: design ----------------
def fig1():
    t = pd.read_csv("trials.csv.gz", low_memory=False)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    # timeline schematic
    for x, lab, sub in [
        (0.6, "t0  baseline", "non-user, aged ≥50\nconfounders L(t0), L(t−1)"),
        (3.0, "t1  exposure", "started using internet\nvs still non-user"),
        (5.4, "t2  outcome", "memory, IADL, depression,\ncancer (NCO), death"),
    ]:
        ax.add_patch(
            FancyBboxPatch(
                (x, 2.45),
                1.9,
                0.9,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fc="#f3f2ee",
                ec="none",
            )
        )
        ax.text(x + 0.95, 3.12, lab, ha="center", fontsize=9, weight="bold", color=INK)
        ax.text(x + 0.95, 2.72, sub, ha="center", va="center", fontsize=7.3, color=INK2)
    for x in [2.55, 4.95]:
        ax.annotate(
            "",
            xy=(x + 0.4, 2.9),
            xytext=(x, 2.9),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
        )
    ax.text(
        7.6,
        3.1,
        "~2 years between waves;\nevery eligible wave starts\na new nested trial",
        fontsize=7.5,
        color=INK2,
        va="top",
    )
    # per-cohort counts
    yy = 1.85
    ax.text(0.1, yy, "Cohort", fontsize=8, weight="bold")
    ax.text(2.1, yy, "Baseline waves", fontsize=8, weight="bold")
    ax.text(4.6, yy, "Person-trials", fontsize=8, weight="bold")
    ax.text(6.1, yy, "Persons", fontsize=8, weight="bold")
    ax.text(7.4, yy, "Initiators (%)", fontsize=8, weight="bold")
    waves = {
        "HRS": "2002–2016 (8 trials)",
        "ELSA": "2004–2014 (6 trials)",
        "SHARE": "2013, 2015 (2 trials)",
        "CHARLS": "2011, 2013, 2015 (3 trials)",
    }
    for i, c in enumerate(COH):
        d = t[t.cohort == c]
        y = yy - 0.38 * (i + 1)
        ax.plot(0.1, y + 0.05, MK[c], color=COL[c], ms=6)
        ax.text(0.3, y, NAMES[c], fontsize=8, color=INK)
        ax.text(2.1, y, waves[c], fontsize=8, color=INK2)
        ax.text(4.6, y, f"{len(d):,}", fontsize=8, color=INK2)
        ax.text(6.1, y, f"{d.id.nunique():,}", fontsize=8, color=INK2)
        ax.text(7.4, y, f"{int(d.A.sum()):,} ({100*d.A.mean():.1f}%)", fontsize=8, color=INK2)
    ax.text(
        0.1,
        yy - 0.38 * 5 + 0.05,
        "Counts before restriction to the propensity-overlap population (0.025 ≤ e ≤ 0.975).",
        fontsize=7,
        color=MUTED,
    )
    save(fig, "Fig1_design")


if __name__ == "__main__":
    import os

    os.makedirs("fig", exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4()
    print("ok")


# ---------------- NEW Figure 2: evidence ladder ----------------
def fig_ladder():
    d = pd.read_csv("ladder_all.csv")
    rungs = [
        "Cross-sectional association",
        "Target trial emulation",
        "Placebo: initiation 1 wave after outcome",
        "Placebo: initiation 2 waves after outcome",
    ]
    rlab = [
        "Cross-sectional\nassociation",
        "Target trial\nemulation",
        "Placebo: start 1 wave\nafter outcome",
        "Placebo: start 2 waves\nafter outcome",
    ]
    COLX = dict(COL)
    COLX["LASI"] = "#e87ba4"
    MKX = dict(MK)
    MKX["LASI"] = "v"
    panels = [
        ("Y_recall", "a  Memory (word recall)", "Difference (SD units)", False),
        ("Y_iadl", "b  IADL difficulty", "Risk ratio (log scale)", True),
        ("Y_dep", "c  Elevated depressive symptoms", "Risk ratio (log scale)", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    offs = {"HRS": -0.27, "ELSA": -0.15, "SHARE": -0.03, "CHARLS": 0.09, "LASI": 0.21}
    for ax, (y, title, xl, rr) in zip(axes, panels):
        for i, rg in enumerate(rungs):
            yy = len(rungs) - i
            if i >= 2:
                ax.axhspan(yy - 0.5, yy + 0.5, color="#f6f5f1", zorder=-2)
            g = d[(d.rung == rg) & (d.y == y)]
            for _, x in g[g.cohort != "Pooled"].iterrows():
                e, lo, hi = (x.rr, x.rr_lo, x.rr_hi) if rr else (x.est, x.lo, x.hi)
                ax.plot(
                    [lo, hi], [yy + offs[x.cohort]] * 2, color=COLX[x.cohort], lw=1.2, alpha=0.7
                )
                ax.plot(
                    e,
                    yy + offs[x.cohort],
                    MKX[x.cohort],
                    color=COLX[x.cohort],
                    ms=4.5,
                    mec=SURF,
                    mew=0.6,
                )
            p = g[g.cohort == "Pooled"].iloc[0]
            e, lo, hi = (p.rr, p.rr_lo, p.rr_hi) if rr else (p.est, p.lo, p.hi)
            ax.fill([lo, e, hi, e], [yy + 0.36, yy + 0.47, yy + 0.36, yy + 0.25], color=INK)
            ax.text(
                1.02,
                yy + 0.36,
                f"{e:.2f}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=7.5,
                weight="bold",
                color=INK,
            )
        ax.axvline(1 if rr else 0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        if rr:
            ax.set_xscale("log")
            ax.set_xticks([0.25, 0.5, 1.0, 1.5])
            ax.set_xticklabels(["0.25", "0.5", "1.0", "1.5"])
            ax.set_xlim(0.2, 1.8)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.set_xlim(-0.15, 0.65)
        ax.set_title(title, loc="left", fontsize=9.5, weight="bold")
        ax.set_xlabel(xl)
        style(ax)
    axes[0].set_yticks([len(rungs) - i for i in range(len(rungs))])
    axes[0].set_yticklabels(rlab)
    axes[0].set_ylim(0.45, len(rungs) + 0.6)
    from matplotlib.lines import Line2D

    h = [
        Line2D(
            [], [], marker=MKX[c], color=COLX[c], ls="", ms=5, label=NAMES.get(c, "LASI (India)")
        )
        for c in ["HRS", "ELSA", "SHARE", "CHARLS", "LASI"]
    ]
    h.append(
        Line2D(
            [],
            [],
            marker="D",
            color=INK,
            ls="",
            ms=6,
            label="Pooled (random effects); value at right",
        )
    )
    fig.legend(
        handles=h,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
        fontsize=7.5,
    )
    fig.text(
        0.01,
        -0.06,
        "Shaded rows: placebo exposures that occur after the outcome and cannot cause it. Cross-sectional IADL = prevalent difficulty; other rungs = incident difficulty.",
        fontsize=7,
        color=MUTED,
    )
    fig.subplots_adjust(wspace=0.22)
    save(fig, "Fig2_evidence_ladder")


if __name__ == "__main__":
    fig_ladder()


# ---------------- Figure 4: falsification tests (within-adopter + pre-trend) ----------------
def fig_falsify():
    from aipw import meta_dl

    w = pd.read_csv("results_within.csv")
    pre = pd.read_csv("results_pretrend.csv")
    tte = pd.read_csv("results_main_hgb_trim0.025_lag_extra.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    specs = [
        ("Y_recall", "Y_pre", "a  Memory", "Difference (SD units)", 0.10, 1),
        (
            "Y_iadl",
            "Y_iadl_pre",
            "b  New IADL difficulty",
            "Risk difference (percentage points)",
            2.0,
            100,
        ),
    ]
    rows = [
        ("Target trial: initiators vs non-initiators", "tte"),
        ("Pre-trend: before anyone started (t−1 to t0)", "pre"),
        ("Late initiators (after outcome) vs never", "late vs never"),
        ("Early initiators (before outcome) vs never", "early vs never"),
        ("Early vs late initiators (same adopters)", "early vs late"),
    ]
    for ax, (y, ypre, title, xl, delta, sc) in zip(axes, specs):
        for i, (lab, key) in enumerate(rows):
            yy = len(rows) - i
            if key == "tte":
                g = tte[tte.y == y]
            elif key == "pre":
                g = pre[pre.y == ypre]
            else:
                g = w[(w.contrast == key) & (w.y == y)]
            if key == "early vs late":
                ax.axhspan(yy - 0.45, yy + 0.45, color="#e7f1fb", zorder=-3)
                ax.fill_betweenx([yy - 0.45, yy + 0.45], -delta, delta, color="#d6e6f8", zorder=-2)
            for j, c in enumerate(COH):
                x = g[g.cohort == c]
                if x.empty:
                    continue
                x = x.iloc[0]
                off = -0.3 + 0.12 * j
                ax.plot([x.lo * sc, x.hi * sc], [yy + off] * 2, color=COL[c], lw=1.2, alpha=0.75)
                ax.plot(x.est * sc, yy + off, MK[c], color=COL[c], ms=4.5, mec=SURF, mew=0.6)
            m = meta_dl(g.est, g.se)
            e, lo, hi = m["est"] * sc, m["lo"] * sc, m["hi"] * sc
            ax.fill([lo, e, hi, e], [yy + 0.3, yy + 0.42, yy + 0.3, yy + 0.18], color=INK)
            ax.text(
                1.02,
                yy + 0.3,
                f"{e:+.2f}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=7.5,
                weight="bold",
            )
        ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(title, loc="left", fontsize=9.5, weight="bold")
        ax.set_xlabel(xl)
        style(ax)
        ax.text(
            0.0,
            0.05,
            f"darker band = equivalence margin ±{delta:g}",
            transform=ax.transAxes,
            fontsize=6.5,
            color=INK2,
        )
    axes[0].set_yticks([len(rows) - i for i in range(len(rows))])
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].set_ylim(0.3, len(rows) + 0.6)
    axes[1].set_xlim(-9, 5)
    from matplotlib.lines import Line2D

    h = [Line2D([], [], marker=MK[c], color=COL[c], ls="", ms=5, label=NAMES[c]) for c in COH] + [
        Line2D([], [], marker="D", color=INK, ls="", ms=6, label="Pooled")
    ]
    fig.legend(
        handles=h,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.55, 1.06),
        fontsize=7.5,
    )
    fig.subplots_adjust(wspace=0.22)
    save(fig, "Fig4_falsification")


# ---------------- Supplementary: calibrated simulation ----------------
def fig_sim():
    d = pd.read_csv("simulation_grid.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for ax, est, title in zip(
        axes,
        ["TTE", "placebo", "early_minus_late"],
        ["Target trial estimate", "Placebo estimate", "Early minus late initiators"],
    ):
        for k, (g, col) in enumerate(
            zip([0.5, 1.0, 1.5, 2.0], ["#9ec5f4", "#5598e7", "#1c5cab", "#0d366b"])
        ):
            s = d[(d.g == g) & (d.su == 0.6)]
            ax.plot(s.beta, s[est], "-o", color=col, ms=4, lw=1.5, label=f"selection strength {g}")
        ax.plot([0, 0.1], [0, 0.1], color=MUTED, ls=(0, (3, 2)), lw=0.8)
        ax.set_title(title, loc="left", fontsize=9, weight="bold")
        ax.set_xlabel("True effect (SD)")
        style(ax)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("Estimate (SD)")
    axes[2].axhspan(-0.032, 0.055, color="#f9eeda", zorder=-2)
    axes[2].text(
        0.098, -0.025, "observed early − late (95% CI)", fontsize=6.5, color=INK2, ha="right"
    )
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1), fontsize=7.5
    )
    save(fig, "SuppFig2_simulation")


# ---------------- Supplementary: DAG ----------------
def fig_dag():
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 4)
    nodes = {
        "L0": (0.8, 2),
        "Y0": (0.8, 0.8),
        "A1": (3.2, 3.1),
        "Y1": (3.2, 0.8),
        "Y2": (5.6, 0.8),
        "A3": (8.0, 3.1),
        "Y3": (8.0, 0.8),
        "U": (4.4, 3.5),
    }
    lab = {
        "L0": "L(t0)\nmeasured",
        "Y0": "Y(t0)",
        "A1": "A(t1)\nstart",
        "Y1": "Y(t1)",
        "Y2": "Y(t2)\noutcome",
        "A3": "A(t3)\nplacebo start",
        "Y3": "Y(t3)",
        "U": "U: unmeasured reserve / trajectory",
    }
    for k, (x, y) in nodes.items():
        ax.text(
            x,
            y,
            lab[k],
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.35", fc="#f3f2ee" if k != "U" else "#fbebeb", ec="none"
            ),
        )

    def arr(a, b, col=MUTED, ls="-", rad=0.0):
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=10,
                color=col,
                lw=1.1,
                ls=ls,
                shrinkA=18,
                shrinkB=18,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    for a, b in [
        ("Y0", "Y1"),
        ("Y1", "Y2"),
        ("Y2", "Y3"),
        ("L0", "A1"),
        ("L0", "Y1"),
        ("Y0", "A1"),
    ]:
        arr(a, b)
    arr("A1", "Y2", "#2a78d6")
    ax.text(4.55, 2.15, "effect of interest", fontsize=7, color="#2a78d6")
    for a, b in [("U", "A1"), ("U", "A3"), ("U", "Y2")]:
        arr(a, b, "#c0392b", (0, (3, 2)))
    arr("Y1", "A1", "#eb6834", rad=0.2)
    arr("Y2", "A3", "#eb6834", rad=0.0)
    arr("Y3", "A3", "#eb6834", rad=-0.2)
    ax.text(8.6, 1.95, "reverse path\n(outcome → start)", fontsize=7, color="#eb6834")
    ax.text(
        0.1,
        0.05,
        "A(t3) cannot affect Y(t2): any placebo association runs through U or the reverse path.\nEarly-vs-late contrast compares A(t1)=1 with A(t3)=1 among eventual adopters, cancelling U shared by adopters.",
        fontsize=7,
        color=INK2,
        va="bottom",
    )
    save(fig, "SuppFig3_DAG")


if __name__ == "__main__":
    fig_falsify()
    fig_sim()
    fig_dag()
