"""Plot the reported cycle-6 RUL forecasts for both battery runs."""

from pathlib import Path

import matplotlib.pyplot as plt

METHODS = (
    "Linear extrapolation",
    "Ridge baseline",
    "Ridge + ΔQ variance",
    "Ridge + ΔQ variance + charge duration",
    "LSTM",
    "Transformer",
)
COLORS = ("#007A70", "#8456A4", "#8456A4", "#8456A4", "#2166AC", "#2166AC")

# Cycle-6 results reported in Task 4. Predictions follow the order in METHODS.
RUNS = (
    ("201214_002_4", 17.27, (15.90, 5.45, 4.35, 18.72, 8.07, 8.44)),
    ("201217_001_2", 5.96, (4.34, 12.48, 10.77, 0.30, 9.31, 12.43)),
)


def main():
    """Save a two-panel comparison with the measured RUL marked for each run."""
    positions = range(len(METHODS))
    fig, axes = plt.subplots(
        1, 2, sharey=True, figsize=(12.6, 4.6), layout="constrained"
    )

    for ax, (run_id, measured_rul, predictions) in zip(axes, RUNS):
        ax.barh(positions, predictions, color=COLORS, height=0.56)
        ax.axvline(
            measured_rul,
            color="#263244",
            linestyle="--",
            linewidth=1.6,
            label=f"Measured reference · {measured_rul:.2f}",
        )
        for row, prediction in enumerate(predictions):
            ax.text(prediction + 0.25, row, f"{prediction:.2f}", va="center", fontsize=9)

        ax.set(
            title=f"Run {run_id}",
            xlabel="Predicted RUL (cycles)",
            xlim=(0, 21),
            xticks=range(0, 21, 5),
        )
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1.01), frameon=False, fontsize=8)

    axes[0].set_yticks(positions, METHODS)
    axes[0].set_ylim(len(METHODS) - 0.4, -0.9)
    fig.suptitle("Early remaining useful life forecasts at cycle 6",
                 fontsize=15, fontweight="bold")

    output = Path(__file__).resolve().parent / "results" / "rul_comparison_cycle_6.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
