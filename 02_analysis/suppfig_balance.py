"""Supplementary Fig. 7: covariate balance (moved from Fig. 1d in v1.4)."""

from ncfig import *

fig = plt.figure(figsize=(3.6, 3.3))
ax = fig.add_subplot(111)
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
    "dep": "Depression score",
    "ncond": "Chronic conditions",
    "partnered": "Partnered",
    "live_alone": "Lives alone",
    "female": "Female",
    "rural": "Rural",
    "ever_before": "Used before t0",
    "year": "Calendar year",
    "recall_lag": "Recall, t−1",
    "sp_internet": "Spouse online",
}
for i, v in enumerate(order):
    for _, r in b[b["var"] == v].iterrows():
        ax.plot(abs(r.smd_raw), i, "o", mfc="none", mec=COL[r.cohort], ms=3.4, mew=0.7)
        ax.plot(abs(r.smd_w), i, MK[r.cohort], color=COL[r.cohort], ms=3.4, mec=SURF, mew=0.3)
ax.axvline(0.1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
ax.set_yticks(range(len(order)))
ax.set_yticklabels([nice.get(v, v) for v in order])
ax.set_xlabel("|Standardised mean difference|")
ax.set_xlim(0, max(0.8, b.smd_raw.abs().max() + 0.05))
clean(ax)
h = [Line2D([], [], marker=MK[c], color=COL[c], ls="", ms=4, label=c) for c in COH] + [
    Line2D([], [], marker="o", mfc="none", mec=INK2, ls="", ms=4, label="unweighted"),
    Line2D([], [], marker="o", color=INK2, ls="", ms=4, label="weighted"),
]
ax.legend(handles=h, frameon=False, loc="lower right", fontsize=FS * 5.4, ncol=1)
save(fig, "SuppFig7_balance")
