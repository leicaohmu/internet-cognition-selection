from figures import *
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(9, 3.9))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(-0.7, 4.3)
N = {
    "L0": (0.6, 2.0),
    "Y0": (1.4, 0.6),
    "A1": (3.6, 2.4),
    "Y1": (3.6, 0.6),
    "Y2": (5.9, 0.6),
    "A3": (8.2, 2.4),
    "Y3": (8.2, 0.6),
    "U": (5.9, 3.85),
}
LAB = {
    "L0": "L(t0)\nmeasured",
    "Y0": "Y(t0)",
    "A1": "A(t1)\nstart",
    "Y1": "Y(t1)",
    "Y2": "Y(t2)\noutcome",
    "A3": "A(t3)\nlate / placebo start",
    "Y3": "Y(t3)",
    "U": "U: unmeasured reserve or trajectory",
}
for k, (x, y) in N.items():
    ax.text(
        x,
        y,
        LAB[k],
        ha="center",
        va="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fbebeb" if k == "U" else "#f3f2ee", ec="none"),
    )


def arr(a, b, col=MUTED, ls="-", rad=0.0, sA=20, sB=20):
    ax.add_patch(
        FancyArrowPatch(
            N[a],
            N[b],
            arrowstyle="-|>",
            mutation_scale=10,
            color=col,
            lw=1.1,
            ls=ls,
            shrinkA=sA,
            shrinkB=sB,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


for a, b in [("Y0", "Y1"), ("Y1", "Y2"), ("Y2", "Y3"), ("L0", "A1"), ("L0", "Y1"), ("Y0", "A1")]:
    arr(a, b)
arr("A1", "Y2", "#2a78d6")
ax.text(4.95, 1.75, "effect of interest", fontsize=7.5, color="#2a78d6")
for b, rad in [("A1", 0.15), ("A3", -0.15), ("Y0", 0.25), ("Y1", 0.1), ("Y2", 0.0), ("Y3", -0.1)]:
    arr("U", b, "#c0392b", (0, (3, 2)), rad, sA=8)
arr("Y1", "A1", "#eb6834", rad=0.25)
arr("Y2", "A3", "#eb6834")
arr("Y3", "A3", "#eb6834", rad=-0.25)
ax.text(9.05, 1.5, "reverse paths\n(health → start)", fontsize=7.5, color="#eb6834")
ax.text(
    0.0,
    -0.55,
    "A(t3) cannot affect Y(t2): a placebo association runs through U or the reverse path Y(t2) → A(t3).\n"
    "Early vs late compares A(t1)=1 with A(t3)=1; it removes U only if U is shared by early and late adopters (an assumption).\n"
    "Not drawn: survival and response to t3, which define the late group and may depend on Y(t2).",
    fontsize=7.2,
    color=INK2,
    va="bottom",
)
save(fig, "SuppFig3_DAG")
