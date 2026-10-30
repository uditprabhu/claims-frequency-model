"""Charts saved to results/ and embedded in the README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import calibration_table, lorenz_curve

# Fixed colour per model, so the same model is the same colour in every chart
COLORS = {"Baseline": "#1baf7a", "Poisson GLM": "#2a78d6",
          "Gradient boosting": "#eb6834"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 130,
})


def plot_lorenz(y, preds: dict, exposure, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="Random ranking")
    for name, (pred, g) in preds.items():
        x, cum = lorenz_curve(y, pred, exposure)
        ax.plot(x, cum, lw=2, color=COLORS[name], label=f"{name} (Gini {g:.3f})")
    ax.set(xlabel="Cumulative share of exposure (safest → riskiest)",
           ylabel="Cumulative share of claims",
           title="Ranking power: who does the model flag as risky?",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(y, preds: dict, exposure, path):
    names = [n for n in preds if n != "Baseline"]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.2),
                             sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        tab = calibration_table(y, preds[name][0], exposure)
        ax.plot(tab["bin"] + 1, tab["observed"], "o", ms=7, color=INK,
                label="Observed")
        ax.plot(tab["bin"] + 1, tab["predicted"], lw=2, color=COLORS[name],
                label="Predicted")
        ax.set(title=name, xlabel="Risk decile (1 = safest)",
               xticks=range(1, 11))
        ax.legend(loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("Claims per policy-year")
    fig.suptitle("Calibration: predicted vs observed frequency by risk decile",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
