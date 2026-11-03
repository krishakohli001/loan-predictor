"""Dark-blue plotting theme shared by the figures and the Streamlit app."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, PANEL, TEXT, GRID = "#0b1d3a", "#12294f", "#e6efff", "#2a4170"
GOOD, BAD, ACC, WARN = "#4ade80", "#f87171", "#4da3ff", "#fbbf24"
RISK_COLORS = {0: GOOD, 1: BAD}


def apply():
    plt.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
        "grid.color": GRID, "axes.grid": True, "grid.alpha": 0.5,
        "axes.titlesize": 12, "axes.titleweight": "bold", "legend.facecolor": BG,
        "legend.edgecolor": GRID, "figure.dpi": 110, "savefig.dpi": 130,
    })


apply()
