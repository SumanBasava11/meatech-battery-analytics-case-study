"""Estimate end of life (EOL) and remaining useful life (RUL) from early capacity.

Check 80% of rated Ah and benchmark linear and exponential extrapolation
against 80% of measured cycle-2 capacity. Save PNGs in the Task 4 results
folder and report forecast errors in cycles.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ("Linear", "Exponential")
WINDOWS = range(3, 11)
REFERENCE_CYCLE = 2
FRACTION = 0.80
CONSECUTIVE = 3

def measured_eol(capacity, threshold):
    """Interpolate the first bracketed drop confirmed by three below-limit cycles."""
    if capacity.empty:
        return np.nan, "No valid measurements"
    if capacity.iloc[0] < threshold:
        return np.nan, "Below threshold at first measurement"

    for cycle in capacity.index:
        previous = capacity.get(cycle - 1, np.nan)
        following = capacity.reindex(range(cycle, cycle + CONSECUTIVE))
        if previous >= threshold and following.lt(threshold).all():
            fraction = (previous - threshold) / (previous - following.iloc[0])
            return cycle - 1 + fraction, "Measured EOL confirmed"

    return np.nan, "Measured End-of-Life unavailable"

def fit_forecast(early, threshold, method):
    """Fit an early capacity window and solve for its threshold crossing.

    Linear regression uses capacity in Ah. Exponential regression uses
    log capacity. Return NaN for a flat or rising fit or a past crossing.
    """
    x = early.index.to_numpy(dtype=float)
    y = early.to_numpy(dtype=float)
    exponential = method == "Exponential"
    slope, intercept = np.polyfit(x, np.log(y) if exponential else y, 1)
    target = np.log(threshold) if exponential else threshold
    predicted_eol_cycle = (target - intercept) / slope if slope < 0 else np.nan

    def curve(cycles):
        fitted = slope * np.asarray(cycles) + intercept
        return np.exp(fitted) if exponential else fitted

    return curve, predicted_eol_cycle if predicted_eol_cycle > x[-1] else np.nan

def evaluate(run_id, capacity, early_series, targets):
    """Score EOL forecasts from the first 3–10 valid samples.

    Fit only the samples available at each cutoff. Use later measurements
    to confirm EOL and score prediction error. RUL is EOL minus cutoff.
    """
    rows = []
    for name, threshold in targets.items():
        history = capacity if name == "80% of nominal capacity" else early_series
        measured_eol_cycle, _ = measured_eol(history, threshold)
        for n in WINDOWS:
            early = early_series.iloc[:n]
            if len(early) < n:
                continue
            last = early.index[-1]
            for method in METHODS:
                predicted = np.nan
                if history.loc[:last].ge(threshold).all():
                    _, predicted = fit_forecast(early, threshold, method)
                error = predicted - measured_eol_cycle if measured_eol_cycle > last else np.nan
                rows.append({
                    "Run": run_id, "Target": name, "Early samples": n,
                    "Last cycle seen": last, "Method": method, "Threshold (Ah)": threshold,
                    "Predicted EOL cycle": predicted, "Measured EOL cycle": measured_eol_cycle,
                    "Predicted RUL (cycles)": predicted - last,
                    "Measured RUL (cycles)": measured_eol_cycle - last if measured_eol_cycle > last else np.nan,
                    "Signed error (cycles)": error, "Absolute error (cycles)": abs(error),
                })
    return rows


def plot_run(run_id, capacity, targets, results, output_dir):
    """Save capacity fits and cycle-2 benchmark errors for one run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    ax, errors = axes
    ax.plot(capacity.index, capacity, color="#334155", label="Measured capacity")
    early = capacity.iloc[:5]
    ax.scatter(early.index, early, color="#334155", label="First five measurements")
    colors = {"Linear": "#2563EB", "Exponential": "#0F766E"}

    for name, threshold in targets.items():
        ax.axhline(threshold, linestyle=":", label=f"{name} ({threshold:.2f} Ah)")

    name = "80% of cycle-2 capacity"
    threshold = targets.get(name)
    benchmark = results.loc[results["Target"].eq(name)]
    if threshold is not None and len(early) == 5 and early.ge(threshold).all():
        for method in METHODS:
            curve, predicted_eol_cycle = fit_forecast(early, threshold, method)
            end = min(predicted_eol_cycle, capacity.index[-1]) if np.isfinite(predicted_eol_cycle) else capacity.index[-1]
            x = np.linspace(early.index[0], end, 150)
            ax.plot(x, curve(x), "--", color=colors[method], label=f"{method} fit")

    for method in METHODS:
        rows = benchmark.loc[benchmark["Method"].eq(method)]
        if rows["Absolute error (cycles)"].notna().any():
            errors.plot(rows["Early samples"], rows["Absolute error (cycles)"],
                        marker="o", color=colors[method], label=method)
            
    ax.set(xlabel="Cycle index", ylabel="Discharge capacity (Ah)", title="Capacity and thresholds")
    ax.legend(fontsize=8)
    errors.set(xlabel="Number of early measurements", ylabel="Absolute EOL error (cycles)",
               title="80% of cycle-2 capacity")
    if errors.lines:
        errors.legend()
    else:
        errors.text(0.5, 0.5, "No evaluable forecasts", ha="center", transform=errors.transAxes)
    for panel in axes:
        panel.grid(alpha=0.2)
    fig.suptitle(run_id)
    fig.savefig(output_dir / f"capacity_fade_{run_id}.png", dpi=200)
    plt.close(fig)

def plot_rul_by_cutoff(results, output, title, seed_results=None):
    """Plot remaining cycles at each forecast origin for each battery run."""
    runs = list(results.groupby("Run", sort=False))
    if not runs:
        return
    fig, axes = plt.subplots(
        1, len(runs), figsize=(6 * len(runs), 4.4),
        squeeze=False, layout="constrained",
    )
    for ax, (run, rows) in zip(axes[0], runs):
        measured = rows.sort_values("Last cycle seen").drop_duplicates("Last cycle seen")
        ax.plot(measured["Last cycle seen"], measured["Measured RUL"], "--",
                color="#334155", label="RUL from measurements")

        for method, forecasts in rows.groupby("Method", sort=False):
            forecasts = forecasts.sort_values("Last cycle seen")
            line, = ax.plot(forecasts["Last cycle seen"], forecasts["Predicted RUL"],
                            marker="o", label=method)
            if seed_results is not None:
                seeds = seed_results.loc[
                    seed_results["Run"].eq(run) & seed_results["Method"].eq(method)
                ]
                ax.scatter(seeds["Last cycle seen"], seeds["Predicted RUL"],
                           s=12, alpha=0.3, color=line.get_color())

        ax.set(title=run.removeprefix("LR1865SZ_cycles"),
               xlabel="Last observed cycle", ylabel="Remaining useful life (cycles)",
               xticks=sorted(rows["Last cycle seen"].unique()), ylim=(0, None))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    """Validate cycle summaries, report EOL and RUL, and save PNGs."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=script_dir.parent / "data/processed/cycle_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    valid = summary.loc[summary["has_discharge"].eq(True)].copy()
    valid = valid.loc[np.isfinite(valid["discharge_capacity_ah"])
                      & valid["discharge_capacity_ah"].gt(0)]
    if valid.empty:
        parser.error("The cycle summary contains no valid discharge capacities.")
    if valid.duplicated(["run_id", "Cycle_Index"]).any():
        parser.error("Expected one discharge capacity per run and cycle.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for run_id, group in valid.groupby("run_id"):
        # Include earlier valid measurements when checking the nominal threshold.
        capacity = group.set_index("Cycle_Index")["discharge_capacity_ah"].sort_index()
        nominal = group["nominal_capacity_ah"].dropna().unique()
        if len(nominal) != 1 or not np.isfinite(nominal[0]) or nominal[0] <= 0:
            parser.error(f"{run_id} needs one positive nominal capacity.")
        targets = {"80% of nominal capacity": FRACTION * nominal[0]}
        rated_threshold = targets["80% of nominal capacity"]
        rated_eol, rated_status = measured_eol(capacity, rated_threshold)
        print(f"\n{run_id} | Rated threshold {rated_threshold:.3f} Ah | {rated_status}")
        if np.isfinite(rated_eol):
            print(f"Rated-threshold measured EOL cycle: {rated_eol:.2f}")
        if REFERENCE_CYCLE in capacity.index:
            targets["80% of cycle-2 capacity"] = FRACTION * capacity.loc[REFERENCE_CYCLE]

        print(f"\n{run_id} | {len(capacity)} valid discharge measurements")
        early_series = capacity.loc[REFERENCE_CYCLE:]
        if len(early_series) < min(WINDOWS):
            print("Insufficient measurements from cycle 2 onward.")
            continue
        results = pd.DataFrame(evaluate(run_id, capacity, early_series, targets))
        all_results.append(results)
        plot_run(run_id, early_series, targets, results, args.output_dir)

    if not all_results:
        return
    results = pd.concat(all_results, ignore_index=True)
    results = results.loc[results["Target"].eq("80% of cycle-2 capacity")]
    if results.empty:
        print("No run has a valid cycle-2 benchmark.")
        return
    print("\nThe early forecast benchmark below uses 80% of measured cycle-2 capacity.")
    plot_rul_by_cutoff(
        results.rename(columns={
            "Predicted RUL (cycles)": "Predicted RUL",
            "Measured RUL (cycles)": "Measured RUL",
        }),
        args.output_dir / "RUL distribution_capacity_fade.png",
        "Capacity-fade RUL forecasts by cutoff",
    )
    columns = ["Early samples", "Last cycle seen", "Method", "Predicted EOL cycle",
               "Measured EOL cycle", "Predicted cycles remaining", "Signed error (cycles)"]
    results = results.rename(columns={"Predicted RUL (cycles)": "Predicted cycles remaining"})
    for (run_id, target), rows in results.groupby(["Run", "Target"], sort=False):
        print(f"\n{run_id} | {target} ({rows['Threshold (Ah)'].iloc[0]:.4f} Ah)\n")
        print(rows[columns].round(2).to_string(index=False, na_rep="—"))

    metrics = results.groupby("Method").agg(
        Attempts=("Early samples", "size"),
        Evaluated=("Absolute error (cycles)", "count"),
        MAE_cycles=("Absolute error (cycles)", "mean"),
    )
    print("\nSummary across runs and early windows\n")
    print(metrics.round(2).to_string(na_rep="—"))
    print(f"\nPlots saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
