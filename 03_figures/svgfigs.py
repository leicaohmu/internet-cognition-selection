import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"  # keep text as editable text
import ncfig
import nh3fig as N


def save(fig, name):
    fig.savefig(f"fig/svg/{name}.svg", bbox_inches="tight", facecolor=ncfig.SURF)
    plt.close(fig)
    print("svg", name)


N.save = save
for f in ["fig2", "fig3", "fig4", "fig5", "fig6", "fig7n", "fig8", "ed1", "ed2", "ed3", "ed4"]:
    plt.rcParams["svg.fonttype"] = "none"
    getattr(N, f)()
