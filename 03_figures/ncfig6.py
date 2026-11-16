"""Fig. 6: beyond word recall - other cognitive domains, cognitive impairment, health and well-being;
staggered-adoption event study; cross-country selection gradient."""

import numpy as np
import pandas as pd
from ncfig import *
from ncfig import pool
from gradient import metareg

NO = pd.read_csv("results_newout_all.csv")
ES = pd.read_csv("results_es_all.csv")
ESP = pd.read_csv("results_es_pooled.csv")
GR = pd.read_csv("results_gradient.csv")
DES = [
    ("tte", "Target trial"),
    ("placebo", "Placebo (later starters)"),
    ("pretrend", "Pre-trend (gap at t0)"),
    ("early_vs_late", "Early vs late adopters"),
]


def rows_for(outs):
    rows, heads = [], []
    for o, name in outs:
        heads.append(len(rows))
        rows.append((name, None))
        for d, dn in DES:
            rows.append(
                (dn, NO[(NO.outcome == o) & (NO.design == d)][["cohort", "est", "lo", "hi", "se"]])
            )
    return rows, heads


def style_heads(ax, heads, n):
    for t, lbl in zip(ax.get_yticklabels(), range(n)):
        if lbl in heads:
            t.set_fontweight("bold")
            t.set_fontsize(6.6)


def es_panel(ax, y, scale, ylim, title, letter, ylabel):
    lab(ax, letter, title=title)
    p = ESP[ESP.y == y].copy()
    ev = p[p.e.str.lstrip("-").str.isdigit() & (p.k >= 3)].copy()
    ev["e"] = ev.e.astype(int)
    ev = pd.concat([ev, pd.DataFrame([dict(e=-1, est=0, lo=0, hi=0)])]).sort_values("e")
    for c in COH:
        x = ES[
            (ES.y == y) & (ES.cohort == c) & ES.e.astype(str).str.lstrip("-").str.isdigit()
        ].copy()
        x["e"] = x.e.astype(int)
        x = pd.concat([x, pd.DataFrame([dict(e=-1, est=0)])]).sort_values("e")
        ax.plot(
            x.e + (COH.index(c) - 1.5) * 0.06,
            x.est * scale,
            color=COL[c],
            lw=0.6,
            alpha=0.55,
            marker=MK[c],
            ms=1.8,
        )
    sl = p[p.e == "slope_pre"].est.iloc[0]
    xs = np.array([-4, 3])
    ax.plot(xs, sl * (xs + 1) * scale, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.fill_between(ev.e, ev.lo * scale, ev.hi * scale, color=INK, alpha=0.10, lw=0)
    ax.plot(
        ev.e, ev.est * scale, color=INK, lw=1.1, marker="o", ms=2.8, mec=SURF, mew=0.4, zorder=5
    )
    ax.axvline(-0.5, color=MUTED, lw=0.5)
    ax.axhline(0, color=GRID, lw=0.6, zorder=-1)
    ax.set_xticks(range(-4, 4))
    ax.set_xlim(-4.4, 3.4)
    ax.set_ylim(ylim[0], ylim[1] + 0.22 * (ylim[1] - ylim[0]))
    ax.set_xlabel("Waves since first internet use")
    ax.set_ylabel(ylabel)
    pa = p[p.e == "post_avg_adj"].iloc[0]
    ps = p[p.e == "slope_pre"].iloc[0]
    f = lambda v: f"{v * scale:+.2f}".replace("-", "−")
    ax.text(
        0.02,
        0.97,
        f"Pre-trend slope {f(ps.est)}/wave ({f(ps.lo)}, {f(ps.hi)})\n"
        f"Post − trend {f(pa.est)} ({f(pa.lo)}, {f(pa.hi)})",
        transform=ax.transAxes,
        va="top",
        fontsize=FS * 5.2,
        color=INK2,
        bbox=dict(fc=SURF, ec="none", pad=0.6, alpha=0.9),
        zorder=6,
    )
    clean(ax, grid="y")
    ax.spines["left"].set_visible(True)


def grad_panel(ax, design, letter, title):
    lab(ax, letter, title=title)
    d = GR[GR.design == design].copy()
    d["lp"] = np.log(d.prev / (1 - d.prev))
    m = metareg(d)
    for _, r in d.iterrows():
        c = r.cohort
        s = 1 / r.se**2
        ax.errorbar(
            r.lp, r.est, yerr=1.96 * r.se, fmt="none", ecolor=COL[c], elinewidth=0.4, alpha=0.5
        )
        ax.scatter(
            r.lp,
            r.est,
            s=np.clip(s / 40, 4, 60),
            color=COL[c],
            marker=MK[c],
            edgecolor=SURF,
            lw=0.3,
            zorder=4,
        )
    xs = np.linspace(d.lp.min() - 0.2, d.lp.max() + 0.2, 50)
    ax.plot(xs, m["intercept"] + m["slope"] * xs, color=INK, lw=0.8)
    ax.axhline(0, color=GRID, lw=0.6, zorder=-1)
    tk = [0.05, 0.25, 0.5, 0.75]
    ax.set_xticks(np.log(np.array(tk) / (1 - np.array(tk))))
    ax.set_xticklabels([f"{int(t * 100)}%" for t in tk])
    ax.set_xlabel("Internet use among adults ≥50 (baseline waves)")
    ax.set_ylabel("Recall difference (SD)")
    f = lambda v: f"{v:+.3f}".replace("-", "−")
    ax.text(
        0.98,
        0.97,
        f'Slope per logit {f(m["slope"])}\n({f(m["lo"])}, {f(m["hi"])}); P = {m["p"]:.2f}; k = {m["k"]}',
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FS * 5.4,
        color=INK2,
        bbox=dict(fc=SURF, ec="none", pad=0.6, alpha=0.9),
        zorder=6,
    )
    ax.set_ylim(-0.7, 1.2) if design == "tte" else ax.set_ylim(-0.2, 0.55)
    clean(ax, grid="y")
    ax.spines["left"].set_visible(True)


def fig6():
    fig = plt.figure(figsize=(W, 9.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.75, 0.9, 0.9], hspace=0.40)
    top = gs[0].subgridspec(2, 2, wspace=0.9, hspace=0.32, height_ratios=[3, 2])
    ax = fig.add_subplot(top[:, 0])
    lab(ax, "a", x=-0.40, title="Other cognitive domains (SD)")
    r, h = rows_for(
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
    style_heads(ax, h, len(r))
    ax.set_xlabel("Difference, initiators − comparators (SD)")
    ax = fig.add_subplot(top[0, 1])
    lab(ax, "b", x=-0.40, title="Cognitive impairment and health (pp)")
    r, h = rows_for([("cimp", "Cognitive impairment (HRS)"), ("srh", "Incident fair/poor health")])
    forest(ax, r, (-12, 8), scale=100, fmt="{:.1f}", shade=range(5))
    style_heads(ax, h, len(r))
    ax.set_xlabel("Risk difference (percentage points)")
    ax = fig.add_subplot(top[1, 1])
    lab(ax, "c", x=-0.40, title="Life satisfaction (SD)")
    r, h = rows_for([("lsat", "Life satisfaction")])
    forest(ax, r[1:], (-0.4, 0.4), band=0.10, band_rows=[3])
    ax.set_xlabel("Difference (SD)")
    mid = gs[1].subgridspec(1, 3, wspace=0.42)
    es_panel(
        fig.add_subplot(mid[0]),
        "z_recall",
        1,
        (-0.25, 0.3),
        "Event study: word recall",
        "d",
        "Difference (SD)",
    )
    es_panel(
        fig.add_subplot(mid[1]),
        "z_gcog",
        1,
        (-0.25, 0.3),
        "Event study: two-domain composite",
        "e",
        "",
    )
    es_panel(
        fig.add_subplot(mid[2]),
        "srh",
        100,
        (-8, 8),
        "Event study: fair/poor health",
        "f",
        "Difference (pp)",
    )
    bot = gs[2].subgridspec(1, 2, wspace=0.3)
    grad_panel(fig.add_subplot(bot[0]), "pre", "g", "Pre-adoption recall gap (t0) by country")
    grad_panel(fig.add_subplot(bot[1]), "tte", "h", "Target-trial recall estimate by country")
    cohort_legend(fig, y=0.895, ncol=5)
    save(fig, "NC_Fig6")


if __name__ == "__main__":
    fig6()
