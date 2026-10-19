"""Recreate the README scientific figure from recorded CSV metrics. No training."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "results/processed_test_metrics.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    models = ["ANN", "TCN", "XGBoost", "TCN_ANN_XGBoost", "Persistence"]
    labels = ["ANN", "TCN", "XGBoost", "Three-model ensemble", "Persistence"]
    colors = ["#7895a3", "#247b90", "#83b4a2", "#087e75", "#bac1c5"]
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.8), sharey=True)
    fig.subplots_adjust(top=.77, bottom=.20, left=.065, right=.985, wspace=.15)
    for ax, horizon in zip(axes, [1, 6, 12]):
        values = [float(next(r["RMSE"] for r in rows if int(r["horizon"]) == horizon and r["model"] == model)) for model in models]
        ax.bar(range(5), values, color=colors, width=.72, zorder=3)
        ax.set_title(f"+{horizon} hour{'s' if horizon > 1 else ''}", pad=16)
        ax.set_xticks(range(5), ["ANN", "TCN", "XGB", "Ens.", "Pers."])
        ax.set_ylim(0, 27)
        ax.grid(axis="y", color="#e6ebed", zorder=0)
        ax.tick_params(length=0)
        for index, value in enumerate(values):
            ax.text(index, value + .5, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Test RMSE (μg/m³) · lower is better")
    fig.suptitle("PM10 forecasting · recorded test performance", x=.065, ha="left", y=.97, fontsize=17, fontweight="bold", color="#183342")
    fig.text(.065, .88, "24-hour input history · common test cases within each horizon · one fitted seed", fontsize=10, color="#536874")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .07), ncol=5, frameon=False, fontsize=9)
    fig.text(.065, .025, "Source: results/processed_test_metrics.csv. The ensemble shown combines TCN, ANN and XGBoost.", fontsize=8, color="#536874")
    output = ROOT / "docs/images/test_rmse.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
