"""v1.1 (Nature Human Behaviour) figures. NH_Fig1-3 and NH_Fig7 are copies of the v1.0 figures;
NH_Fig4 adds sustained use; NH_Fig5 = other outcomes + dementia + event study; NH_Fig6 = anatomy of selection;
NH_Fig8 = bounds (old Fig. 5) + placebo/target-trial ratios."""

import shutil
import numpy as np
import pandas as pd
import ncfig
from ncfig import *
import ncfig6 as F
from gradient import metareg

for a, b in [(1, 1), (2, 2), (3, 3), (7, 7)]:
    for ext in ["png", "pdf"]:
        shutil.copy(f"fig/NC_Fig{a}.{ext}", f"fig/NH_Fig{b}.{ext}")

DM = pd.read_csv("results_dementia_HRS.csv")
DM["outcome"] = DM.outcome.replace({"dem": "dem"})
F.NO = pd.concat([F.NO, DM[DM.outcome == "dem"]], ignore_index=True)


def fig5():
    fig = plt.figure(figsize=(W, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 0.95], hspace=0.3)
    top = gs[0].subgridspec(2, 2, wspace=0.9, hspace=0.3, height_ratios=[3.7, 1.6])
    ax = fig.add_subplot(top[:, 0])
    lab(ax, "a", x=-0.40, title="Other cognitive domains (SD)")
    r, h = F.rows_for(
        [("ser7", "Serial 7s"), ("verbf", "Verbal fluency"), ("gcog", "Two-domain composite")]
    )
    forest(
        ax,
        r,
        (-0.3, 0.5),
        shade=[i for i in range(5)] + [i for i in range(10, 15)],
        band=0.10,
        band_rows=[4, 9, 14],
    )
    F.style_heads(ax, h, len(r))
    ax.set_xlabel("Difference, initiators − comparators (SD)")
    ax = fig.add_subplot(top[0, 1])
    lab(ax, "b", x=-0.40, title="Impairment, dementia and health (pp)")
    r, h = F.rows_for(
        [
            ("cimp", "Cognitive impairment (HRS)"),
            ("dem", "Dementia (HRS)"),
            ("srh", "Incident fair/poor health"),
        ]
    )
    forest(ax, r, (-12, 8), scale=100, fmt="{:.1f}", shade=list(range(5)) + list(range(10, 15)))
    F.style_heads(ax, h, len(r))
    ax.set_xlabel("Risk difference (percentage points)")
    ax = fig.add_subplot(top[1, 1])
    lab(ax, "c", x=-0.40, title="Life satisfaction (SD)")
    r, h = F.rows_for([("lsat", "Life satisfaction")])
    forest(ax, r[1:], (-0.4, 0.4), band=0.10, band_rows=[3])
    ax.set_xlabel("Difference (SD)")
    mid = gs[1].subgridspec(1, 3, wspace=0.42)
    F.es_panel(
        fig.add_subplot(mid[0]),
        "z_recall",
        1,
        (-0.25, 0.3),
        "Event study: word recall",
        "d",
        "Difference (SD)",
    )
    F.es_panel(
        fig.add_subplot(mid[1]),
        "z_gcog",
        1,
        (-0.25, 0.3),
        "Event study: two-domain composite",
        "e",
        "",
    )
    F.es_panel(
        fig.add_subplot(mid[2]),
        "srh",
        100,
        (-8, 8),
        "Event study: fair/poor health",
        "f",
        "Difference (pp)",
    )
    cohort_legend(fig, y=0.90, ncol=5)
    save(fig, "NH_Fig5")


# ---------------------------------------------------------------- Fig. 6 anatomy of selection
AR = pd.read_csv("results_anatomy.csv")
AP = pd.read_csv("results_anatomy_pooled.csv")
GR = pd.read_csv("results_gradient.csv", dtype={"unit": str})
TILE = {
    "USA (HRS)": (0, 2.5, "US"),
    "England": (3, 1, "EN"),
    "Sweden": (6, 0, "SE"),
    "Estonia": (7, 0, "EE"),
    "Denmark": (5, 1, "DK"),
    "Belgium": (4, 2, "BE"),
    "Germany": (5, 2, "DE"),
    "Poland": (6, 2, "PL"),
    "France": (3, 3, "FR"),
    "Luxembourg": (4, 3, "LU"),
    "Czech Republic": (5, 3, "CZ"),
    "Spain": (2, 4, "ES"),
    "Switzerland": (4, 4, "CH"),
    "Austria": (5, 4, "AT"),
    "Slovenia": (6, 4, "SI"),
    "Croatia": (6, 5, "HR"),
    "Italy": (5, 5, "IT"),
    "Greece": (7, 5.5, "GR"),
    "Israel": (8, 5.5, "IL"),
    "China (CHARLS)": (10, 2.5, "CN"),
}
CTRY = {
    "11": "Austria",
    "12": "Germany",
    "13": "Sweden",
    "15": "Spain",
    "16": "Italy",
    "17": "France",
    "18": "Denmark",
    "19": "Greece",
    "20": "Switzerland",
    "23": "Belgium",
    "25": "Israel",
    "28": "Czech Republic",
    "29": "Poland",
    "31": "Luxembourg",
    "34": "Slovenia",
    "35": "Estonia",
    "47": "Croatia",
}


def uname(u):
    return (
        CTRY[u.split("-")[1]]
        if u.startswith("SHARE-")
        else {"HRS": "USA (HRS)", "ELSA": "England", "CHARLS": "China (CHARLS)", "LASI": "India"}[u]
    )


def fig6():
    fig = plt.figure(figsize=(W, 6.6))
    gs0 = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.42)
    gt = gs0[0].subgridspec(1, 2, width_ratios=[0.8, 1.6], wspace=1.15)
    gbm = gs0[1].subgridspec(1, 3, width_ratios=[0.85, 1.45, 1.0], wspace=0.35)
    gs = {(0, 0): gt[0], (0, 1): gt[1], (1, 0): gbm[0], (1, 1): gbm[1], (1, 2): gbm[2]}
    # a probability of starting by recall quintile
    ax = fig.add_subplot(gs[(0, 0)])
    lab(ax, "a", title="Starting by memory at t0")
    pq = AR[AR.part == "prob_quintile"]
    for c in COH:
        v = pq[pq.cohort == c].sort_values("term").est.values * 100
        ax.plot(range(1, 6), v, color=COL[c], marker=MK[c], ms=3, lw=1)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"], fontsize=5.6)
    ax.set_ylabel("Started by next wave (%)")
    ax.set_xlabel("Word recall quintile at t0")
    clean(ax, "y")
    ax.spines["left"].set_visible(True)
    q5 = AP[(AP.part == "quintile") & (AP.term == "Q5")].iloc[0]
    ax.text(
        0.03,
        0.97,
        f"Q5 vs Q1: OR {np.exp(q5.est):.1f} ({np.exp(q5.lo):.1f}–{np.exp(q5.hi):.1f})",
        transform=ax.transAxes,
        fontsize=5.6,
        va="top",
        color=INK2,
    )
    # b mutually adjusted predictors
    ax = fig.add_subplot(gs[(0, 1)])
    lab(ax, "b", title="Mutually adjusted predictors of starting (odds ratio)")
    terms = [
        ("sp_internet", "Spouse uses internet"),
        ("lvl", "Memory, two-wave mean (per SD)"),
        ("edu", "Education (per SD)"),
        ("wealth_q", "Wealth (per SD)"),
        ("partnered", "Partnered"),
        ("work", "In paid work"),
        ("drz", "Recent memory change (per SD)"),
        ("ncond", "Chronic conditions (per SD)"),
        ("dep", "Depressive symptoms (per SD)"),
        ("live_alone", "Lives alone"),
        ("female", "Women"),
        ("srh_poor", "Fair/poor health"),
        ("iadl_any", "Any IADL difficulty"),
        ("age", "Age (per 10 years)"),
    ]
    pr = AR[AR.part == "predictor"]
    rows = [
        (
            nm,
            pr[pr.term == t].assign(
                lo=lambda z: z.est - 1.96 * z.se, hi=lambda z: z.est + 1.96 * z.se
            )[["cohort", "est", "lo", "hi", "se"]],
        )
        for t, nm in terms
    ]
    n = len(rows)
    offs = np.linspace(-0.27, 0.09, 4)
    for i, (nm, g) in enumerate(rows):
        yy = n - i
        for j, c in enumerate(COH):
            x = g[g.cohort == c]
            if len(x):
                ax.plot(np.exp(x.est.iloc[0]), yy + offs[j], MK[c], color=COL[c], ms=2.3, alpha=0.7)
        p = AP[(AP.part == "predictor") & (AP.term == terms[i][0])].iloc[0]
        ax.plot([np.exp(p.lo), np.exp(p.hi)], [yy + 0.25] * 2, color=INK, lw=0.8)
        ax.plot(np.exp(p.est), yy + 0.25, "D", color=INK, ms=3)
        ax.text(
            1.01,
            yy,
            f"{np.exp(p.est):.2f}",
            transform=ax.get_yaxis_transform(),
            fontsize=5.8,
            va="center",
            weight="bold",
        )
    ax.set_xscale("log")
    ax.axvline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(0.3, 3.5)
    ax.set_xticks([0.4, 0.6, 1, 1.5, 2.5])
    ax.set_xticklabels(["0.4", "0.6", "1", "1.5", "2.5"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.8)
    ax.set_ylim(0.4, n + 0.6)
    clean(ax)
    # c decomposition of the t0 gap
    ax = fig.add_subplot(gs[(1, 0)])
    lab(ax, "c", title="What the gap at t0 consists of")
    gp = AP[AP.part == "gelbach"].set_index("term")
    order = [
        "Age and sex",
        "Education, wealth and work",
        "Health",
        "Household and spouse",
        "Lifestyle",
        "Earlier memory (t−1)",
        "unexplained",
    ]
    cols = ["#9db7d9", "#6f97cc", "#4f7fbf", "#2f62a8", "#1c4f91", "#0f3a73", "#c2410c"]
    left = 0
    for k, col in zip(order, cols):
        v = gp.loc[k, "est"]
        ax.barh(0, v, left=left, color=col, height=0.5, lw=0.4, ec=SURF)
        if v > 0.06:
            ax.text(
                left + v / 2, 0, f"{v:.2f}", ha="center", va="center", fontsize=5.3, color="white"
            )
        left += v
    tot = gp.loc["total"]
    ax.plot([tot.est] * 2, [0.32, 0.42], color=INK, lw=0.8)
    ax.text(tot.est, 0.45, f"total {tot.est:.2f} SD", ha="center", fontsize=5.6)
    for i, (k, col) in enumerate(zip(order, cols)):
        ax.add_patch(
            Rectangle(
                (0.0 + (i % 2) * 0.33, -0.58 - (i // 2) * 0.17),
                0.02,
                0.1,
                color=col,
                transform=ax.transData,
            )
        )
        ax.text(
            0.03 + (i % 2) * 0.33,
            -0.53 - (i // 2) * 0.17,
            k.replace("unexplained", "Not explained by measured factors"),
            fontsize=5.0,
            va="center",
        )
    ax.set_ylim(-1.25, 0.6)
    ax.set_xlim(0, 0.68)
    ax.set_yticks([])
    ax.set_xlabel("Initiator − non-initiator recall at t0 (SD)")
    for s_ in ["left", "top", "right"]:
        ax.spines[s_].set_visible(False)
    # d tile map of the pre-adoption gap
    ax = fig.add_subplot(gs[(1, 1)])
    lab(ax, "d", title="Gap at t0 by country")
    pre = GR[GR.design == "pre"].copy()
    pre["name"] = pre.unit.map(uname)
    prev = GR[GR.design == "xs"].set_index(GR[GR.design == "xs"].unit.map(uname)).prev
    cmap = plt.get_cmap("Blues")
    vmax = 0.32
    for _, r in pre.iterrows():
        x, y, code = TILE[r["name"]]
        col = cmap(0.15 + 0.85 * min(max(r.est, 0), vmax) / vmax)
        ax.add_patch(Rectangle((x, -y), 0.94, 0.94, color=col, ec=SURF, lw=0.8))
        tc = "white" if r.est > 0.17 else INK
        ax.text(
            x + 0.47, -y + 0.62, code, ha="center", va="center", fontsize=6, weight="bold", color=tc
        )
        ax.text(
            x + 0.47,
            -y + 0.27,
            f'{prev[r["name"]] * 100:.0f}%',
            ha="center",
            va="center",
            fontsize=4.8,
            color=tc,
        )
    ax.set_xlim(-0.1, 11.0)
    ax.set_ylim(-6.6, 1.05)
    ax.axis("off")
    ax.text(0.47, -2.85, "USA", ha="center", va="top", fontsize=5, color=INK2)
    ax.text(10.47, -2.85, "China", ha="center", va="top", fontsize=5, color=INK2)
    ax.text(8.6, -0.1, "Europe and Israel\n(ELSA, SHARE)", ha="center", fontsize=5, color=INK2)
    cax = ax.inset_axes([0.02, 0.03, 0.42, 0.045])
    sm = plt.cm.ScalarMappable(
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", [cmap(0.15), cmap(1.0)]),
        norm=plt.Normalize(0, vmax),
    )
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Recall gap at t0 (SD); % = internet use", fontsize=5)
    cb.ax.tick_params(labelsize=5, length=1.5)
    # e gap vs prevalence
    F.grad_panel(fig.add_subplot(gs[(1, 2)]), "pre", "e", "Gap at t0 vs internet use")
    cohort_legend(fig, y=0.93, ncol=5)
    save(fig, "NH_Fig6")


if __name__ == "__main__":
    ncfig.FIG4NAME = "NH_Fig4"
    ncfig.fig4()
    ncfig.FIG5NAME = "NH_Fig8"
    ncfig.fig5()
    fig5()
    fig6()
