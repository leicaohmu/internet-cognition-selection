import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK = "#1f1e1b"
INK2 = "#55534d"
BOX = "#f3f2ee"
COL = {"HRS": "#2a78d6", "ELSA": "#eb6834", "SHARE": "#1baf7a", "CHARLS": "#eda100"}
C = ["HRS", "ELSA", "SHARE", "CHARLS"]
elig = {"HRS": 76560, "ELSA": 20875, "SHARE": 66298, "CHARLS": 55171}
fu = {"HRS": (65350, 6908), "ELSA": (15163, 2184), "SHARE": (37525, 5256), "CHARLS": (47930, 4872)}
tr = {"HRS": (48643, 6693), "ELSA": (14586, 2180), "SHARE": (31052, 5194), "CHARLS": (21980, 4717)}
pers = {"HRS": 18133, "ELSA": 5693, "SHARE": 28591, "CHARLS": 17753}
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)


def box(x, y, w, h, txt, fc=BOX, fs=8, bold=False):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="none")
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        txt,
        ha="center",
        va="center",
        fontsize=fs,
        color=INK,
        weight="bold" if bold else "normal",
    )


def arrow(x, y1, y2):
    ax.annotate("", (x, y2), (x, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))


s = lambda d, i=None: f"{sum((v if i is None else v[i]) for v in d.values()):,}"
box(
    1.4,
    8.6,
    5.6,
    1.1,
    f"Eligible person-trials at baseline (t0): {s(elig)}\noffline, aged ≥50, self-respondent; HRS 2002–2016, ELSA 2004–2014,\nSHARE 2013/2015, CHARLS 2011–2018",
    bold=False,
)
arrow(4.2, 8.6, 7.75)
box(
    7.25,
    7.85,
    2.7,
    0.75,
    f"Excluded {sum(elig.values())-sum(v[0] for v in fu.values()):,}: not interviewed\nat t1 or internet use unknown",
    fc="#fbefe6",
    fs=7.5,
)
box(
    1.4,
    6.6,
    5.6,
    1.1,
    f"Person-trials with known status at t1: {s(fu,0)}\n({sum(pers.values()):,} people); initiators {s(fu,1)}",
    bold=False,
)
arrow(4.2, 6.6, 5.75)
box(
    7.25,
    5.85,
    2.7,
    0.75,
    f"Outside propensity overlap\n(0.025–0.975): {sum(v[0] for v in fu.values())-sum(v[0] for v in tr.values()):,}",
    fc="#fbefe6",
    fs=7.5,
)
box(
    1.4,
    4.6,
    5.6,
    1.1,
    f"Analysed (overlap population): {s(tr,0)} person-trials\ninitiators {s(tr,1)}; outcomes at t2, missing outcomes weighted",
    bold=True,
)
for i, c in enumerate(C):
    x = 0.3 + i * 2.4
    ax.plot([4.2, x + 1.0], [4.6, 3.75], color=INK2, lw=0.8)
    ax.add_patch(
        FancyBboxPatch(
            (x, 0.4), 2.0, 3.3, boxstyle="round,pad=0.02,rounding_size=0.12", fc=BOX, ec="none"
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, 3.35), 2.0, 0.35, boxstyle="round,pad=0.02,rounding_size=0.08", fc=COL[c], ec="none"
        )
    )
    ax.text(x + 1, 3.52, c, ha="center", va="center", fontsize=8.5, color="white", weight="bold")
    ax.text(
        x + 1,
        1.95,
        f"Eligible     {elig[c]:>7,}\nKnown at t1  {fu[c][0]:>7,}\n  people     {pers[c]:>7,}\n  initiators {fu[c][1]:>7,}\nAnalysed     {tr[c][0]:>7,}\n  initiators {tr[c][1]:>7,}\n  retained   {100*tr[c][0]/fu[c][0]:>6.0f}%",
        ha="center",
        va="center",
        fontsize=7.5,
        color=INK,
        linespacing=1.35,
        family="DejaVu Sans Mono",
    )
fig.savefig("fig/SuppFig4_flow.png", dpi=200, bbox_inches="tight")
fig.savefig("fig/SuppFig4_flow.pdf", bbox_inches="tight")
