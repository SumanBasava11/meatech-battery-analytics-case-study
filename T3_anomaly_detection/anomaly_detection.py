"""Screen cycles 2–100 for unusual deviations within each battery run.

Use a capacity-only screening rule and an Isolation Forest that jointly
scores capacity, coulombic efficiency, and mean discharge voltage deviations.
The neighbor baseline uses the following cycle, so screening is retrospective.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

SIGNALS = ["discharge_capacity_ah", "coulombic_efficiency_pct", "mean_discharge_voltage_v"]
DEVIATIONS = [f"{signal}_deviation" for signal in SIGNALS]
CLASSES = ["healthy", "outlier", "not_scored"]

def prepare_cycles(summary):
    """Subtract the mean of the previous and next cycle for each signal.

    Estimate mean discharge voltage as discharge Wh divided by discharge Ah.
    Leave deviations missing at boundaries, gaps, and missing measurements.
    """
    cycles = summary.loc[summary["Cycle_Index"].between(2, 100)].copy()
    cycles = cycles.sort_values(["run_id", "Cycle_Index"]).reset_index(drop=True)
    if cycles.empty or cycles.duplicated(["run_id", "Cycle_Index"]).any():
        raise ValueError("Expected one row per run and cycle in the range 2–100.")

    capacity = cycles["discharge_capacity_ah"]
    voltage = cycles["discharge_energy_wh"] / capacity.where(capacity > 0)
    cycles["mean_discharge_voltage_v"] = voltage
    cycles[SIGNALS] = cycles[SIGNALS].replace([np.inf, -np.inf], np.nan)

    by_run = cycles.groupby("run_id")
    cycle = cycles["Cycle_Index"]
    adjacent = (
        by_run["Cycle_Index"].shift(1).eq(cycle - 1)
        & by_run["Cycle_Index"].shift(-1).eq(cycle + 1)
    )
    expected = (by_run[SIGNALS].shift(1) + by_run[SIGNALS].shift(-1)) / 2
    deviations = cycles[SIGNALS].sub(expected).where(adjacent)
    cycles[DEVIATIONS] = deviations.set_axis(DEVIATIONS, axis=1)
    return cycles

def classify_cycles(cycles):
    """Assign labels and scores in place using one Isolation Forest per run.

    The 5% contamination setting is a screening assumption. Healthy means
    typical under the model. Missing inputs leave a cycle unscored.
    Higher anomaly scores indicate more unusual feature combinations.
    """
    cycles["cycle_class"] = "not_scored"
    cycles["isolation_label"] = pd.Series(pd.NA, index=cycles.index, dtype="Int64")
    cycles["anomaly_score"] = np.nan
    for _, run in cycles.groupby("run_id"):
        inputs = run[DEVIATIONS].dropna()
        if len(inputs) < 2:
            continue
        model = IsolationForest(contamination=0.05, random_state=42)
        labels = model.fit_predict(inputs)
        cycles.loc[inputs.index, "isolation_label"] = labels
        cycles.loc[inputs.index, "anomaly_score"] = -model.score_samples(inputs)
        cycles.loc[inputs.index, "cycle_class"] = np.where(labels == -1, "outlier", "healthy")

def print_metrics(cycles):
    """Report screening coverage and agreement with the capacity-only rule."""
    print("No reference labels available for accuracy, precision, or recall.")
    for run_id, run in cycles.groupby("run_id"):
        counts = run["cycle_class"].value_counts().reindex(CLASSES, fill_value=0)
        scored = counts["healthy"] + counts["outlier"]
        print(f"\n{run_id}")
        print(counts.to_string())
        print(f"Scoring coverage: {scored}/{len(run)} ({scored / len(run):.1%})")
        if scored:
            print(f"Outlier rate among scored cycles: {counts['outlier'] / scored:.1%}")

        # Compare both approaches only where Isolation Forest assigned a label.
        deviation = run[DEVIATIONS[0]].abs() * 1000
        threshold = deviation.quantile(0.95)
        capacity_flags = deviation.gt(threshold)
        model_flags = run["cycle_class"].eq("outlier")
        comparable = run["isolation_label"].notna()
        print(f"Capacity-only 95th percentile: {threshold:.2f} mAh")
        print(f"Capacity-only flagged cycles: {run.loc[capacity_flags, 'Cycle_Index'].tolist()}")
        print(f"Isolation Forest outliers: {run.loc[model_flags, 'Cycle_Index'].tolist()}")
        print(f"Among scored cycles, flagged by both: {(capacity_flags & model_flags).sum()}")
        print(f"Capacity rule only: {(comparable & capacity_flags & ~model_flags).sum()}")
        print(f"Isolation Forest only: {(~capacity_flags & model_flags).sum()}")


def save_plot(cycles, path):
    """Plot all three signed deviations with the same joint classification."""
    runs = list(cycles.groupby("run_id"))
    panels = [
        (DEVIATIONS[0], 1000, "Capacity deviation (mAh)"),
        (DEVIATIONS[1], 1, "Coulombic efficiency deviation (pp)"),
        (DEVIATIONS[2], 1000, "Average discharge voltage deviation (mV)"),
    ]
    styles = [
        ("healthy", "Typical (healthy label)", "#2563EB", "o", 16),
        ("outlier", "Outlier", "#C2410C", "D", 42),
        ("not_scored", "Not scored", "#64748B", "x", 25),
    ]
    fig, axes = plt.subplots(
        3, len(runs), figsize=(6 * len(runs), 8),
        sharex="col", squeeze=False, layout="constrained",
    )

    for col, (run_id, run) in enumerate(runs):
        axes[0, col].set_title(run_id.removeprefix("LR1865SZ_cycles"))
        for row, (signal, scale, ylabel) in enumerate(panels):
            ax = axes[row, col]
            ax.axhline(0, color="0.55", linewidth=1)
            for label, name, color, marker, size in styles:
                points = run.loc[run["cycle_class"].eq(label)].dropna(subset=[signal])
                ax.scatter(points["Cycle_Index"], points[signal] * scale,
                           color=color, marker=marker, s=size, label=name)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
        axes[-1, col].set_xlabel("Cycle index")

    axes[0, 0].legend(frameon=False)
    fig.suptitle("Isolation Forest outliers across three cycle features")
    fig.savefig(path, dpi=200)
    plt.close(fig)

def main():
    """Read the Task 1 cycle summary and export Task 3 screening results."""
    task_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=task_dir.parent / "data/processed/cycle_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=task_dir / "results")
    args = parser.parse_args()

    cycles = prepare_cycles(pd.read_csv(args.summary))
    classify_cycles(cycles)
    print_metrics(cycles)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "anomaly_classification.csv"
    plot_path = args.output_dir / "isolation_forest_cycle_classification.png"
    columns = ["run_id", "Cycle_Index", *SIGNALS, *DEVIATIONS,
               "anomaly_score", "isolation_label", "cycle_class"]
    cycles[columns].to_csv(csv_path, index=False)
    save_plot(cycles, plot_path)
    print(f"\nSaved {csv_path.resolve()}")
    print(f"Saved {plot_path.resolve()}")


if __name__ == "__main__":
    main()