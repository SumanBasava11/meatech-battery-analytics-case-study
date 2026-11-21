"""Benchmark LSTM and Transformer RUL forecasts on held-out runs.

Each model predicts the next capacity change from a three-cycle window
and the ΔQ variance available at the forecast cutoff. Roll forward using
predicted capacities and a fixed cutoff variance until capacity falls
below 80% of measured cycle-2 capacity or the horizon ends. Save two PNGs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

if __package__:
    from .a_capacity_fade_extrapolation import FRACTION, REFERENCE_CYCLE, WINDOWS, measured_eol, plot_rul_by_cutoff
else:
    from a_capacity_fade_extrapolation import FRACTION, REFERENCE_CYCLE, WINDOWS, measured_eol, plot_rul_by_cutoff

WINDOW = 3
HORIZON = 30
SEEDS = range(5)


class CapacityLSTM(nn.Module):
    """Predict the next capacity change with a 16-unit LSTM.

    The output layer maps the final hidden state to one standardized
    capacity change per input window."""

    def __init__(self):
        """Initialize the sequence encoder and scalar output layer."""
        super().__init__()
        self.lstm = nn.LSTM(2, 16, batch_first=True)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        """Map a (batch, three cycles, two features) tensor to (batch, one change)."""
        sequence, _ = self.lstm(x)
        return self.output(sequence[:, -1])


class CapacityTransformer(nn.Module):
    """Predict the next capacity change with a Transformer encoder.

    Learned position embeddings preserve cycle order. The output layer
    maps the final position's representation to one standardized capacity
    change per input window."""

    def __init__(self):
        """Initialize the sequence encoder and scalar output layer."""
        super().__init__()
        self.input_projection = nn.Linear(2, 32)
        self.position = nn.Parameter(torch.empty(1, WINDOW, 32))
        nn.init.normal_(self.position, std=0.02)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, dropout=0, batch_first=True
        )
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        """Map a (batch, three cycles, two features) tensor to (batch, one change)."""
        encoded = self.encoder(self.input_projection(x) + self.position)
        return self.output(encoded[:, -1])


def delta_q_variances(measurements):
    """Calculate variance of Q_cycle(V) minus Q_cycle2(V) in Ah².

    Use the feature-based Ridge definition: average capacities at duplicate
    voltages and interpolate both curves on 200 shared voltage points.
    """
    voltage, charge = "Voltage(V)", "Capacity(Ah)"
    discharge = measurements.loc[
        measurements["phase"].eq("discharge")
        & np.isfinite(measurements[voltage])
        & np.isfinite(measurements[charge])
    ]
    variances = {}
    for run, rows in discharge.groupby("run_id"):
        curves = {
            cycle: group.groupby(voltage)[charge].mean().sort_index()
            for cycle, group in rows.groupby("Cycle_Index")
        }
        reference = curves.get(REFERENCE_CYCLE)
        if reference is None or len(reference) < 2:
            continue
        values = {}
        for cycle, curve in curves.items():
            low = max(reference.index.min(), curve.index.min())
            high = min(reference.index.max(), curve.index.max())
            if low < high:
                grid = np.linspace(low, high, 200)
                delta = np.interp(grid, curve.index, curve.values)
                delta -= np.interp(grid, reference.index, reference.values)
                values[cycle] = float(np.var(delta))
        variances[run] = pd.Series(values, dtype=float)
    return variances


def window_features(capacity, variance):
    """Pair capacity changes with the ΔQ variance at the forecast origin."""
    changes = np.diff(capacity, prepend=capacity[0])
    return np.column_stack([changes, np.full(len(capacity), variance)])


def training_windows(capacities, variances):
    """Build windows from consecutive cycles with a known origin variance."""
    windows, changes = [], []
    for run, capacity in capacities.items():
        variance = variances.get(run, pd.Series(dtype=float))
        for cycle in capacity.index[WINDOW:]:
            window = capacity.reindex(range(cycle - WINDOW, cycle + 1))
            origin_variance = variance.get(cycle - 1, np.nan)
            if window.notna().all() and np.isfinite(origin_variance):
                windows.append(window_features(window.iloc[:-1].to_numpy(), origin_variance))
                changes.append(window.iloc[-1] - window.iloc[-2])
    return np.asarray(windows), np.asarray(changes)


def fit_predictors(windows, changes, epochs):
    """Fit each architecture with five seeds and training-only scaling."""
    mean = windows.mean(axis=(0, 1))
    scale = np.maximum(windows.std(axis=(0, 1)), 1e-8)
    target_scale = max(float(changes.std()), 1e-8)
    inputs = (windows - mean) / scale
    targets = changes / target_scale

    x = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    for name, architecture in [("LSTM", CapacityLSTM), ("Transformer", CapacityTransformer)]:
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = architecture()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model(x), y)
                loss.backward()
                optimizer.step()
            model.eval()

            def predict(values, fitted=model):
                """Convert a two-feature window to a capacity-fraction change."""
                tensor = torch.tensor((values - mean) / scale, dtype=torch.float32)
                with torch.no_grad():
                    return fitted(tensor).item() * target_scale

            yield name, predict


def forecast_eol(predict, early, variance):
    """Roll forward with the last observed variance and interpolate EOL."""
    window = early.iloc[-WINDOW:].to_numpy()
    for step in range(HORIZON):
        previous = window[-1]
        features = window_features(window, variance)[None, :, :]
        current = previous + predict(features)
        if not np.isfinite(current):
            return np.nan
        if current < FRACTION:
            return early.index[-1] + step + (previous - FRACTION) / (previous - current)
        window = np.r_[window[1:], current]
    return np.nan


def compare(capacities, variances, epochs, *, seed_rows=None):
    """Evaluate each run using models trained on the other runs.

    Use the median EOL among seeds that cross the threshold. Record how
    many seeds cross and optionally collect individual RUL estimates.
    """
    rows = []
    for run, capacity in capacities.items():
        measured, _ = measured_eol(capacity, FRACTION)
        windows, changes = training_windows(
            {other: cap for other, cap in capacities.items() if other != run},
            variances,
        )
        if not len(windows):
            raise ValueError(f"{run}: no complete training windows from other runs.")

        variance = variances.get(run, pd.Series(dtype=float))
        early_windows = {}
        for size in WINDOWS:
            early = capacity.reindex(range(REFERENCE_CYCLE, REFERENCE_CYCLE + size))
            if early.notna().all() and early.ge(FRACTION).all() and early.index[-1] in variance.index:
                early_windows[size] = early
        if not early_windows:
            continue

        forecasts = {}
        for name, predict in fit_predictors(windows, changes, epochs):
            for size, early in early_windows.items():
                forecasts.setdefault((name, size), []).append(forecast_eol(predict, early, variance.loc[early.index[-1]]))

        for (method, size), predictions in forecasts.items():
            cutoff = early_windows[size].index[-1]
            if seed_rows is not None:
                seed_rows.extend(
                    (run.removeprefix("LR1865SZ_cycles"), method, cutoff, eol - cutoff)
                    for eol in predictions
                )
            successful = np.asarray(predictions)[np.isfinite(predictions)]
            eol = np.median(successful) if len(successful) else np.nan
            rows.append({
                "Run": run.removeprefix("LR1865SZ_cycles"),
                "Method": method,
                "Early cycles": size,
                "Last cycle seen": cutoff,
                "Delta Q variance (Ah²)": variance.loc[cutoff],
                "Predicted EOL": eol,
                "Measured EOL": measured,
                "Predicted RUL": eol - cutoff,
                "Measured RUL": measured - cutoff,
                "Signed error": eol - measured,
                "Absolute error": abs(eol - measured),
                "Seed SD": successful.std() if len(successful) > 1 else np.nan,
                "Crossings": len(successful),
                "Attempts": len(predictions),
            })
    return pd.DataFrame(rows)


def report(results, output):
    """Save the EOL comparison and plot seed variability where available."""
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    runs = list(results.groupby("Run"))
    fig, axes = plt.subplots(
        1, len(runs), figsize=(6 * len(runs), 4.5), squeeze=False, layout="constrained"
    )
    for ax, (run, subset) in zip(axes[0], runs):
        for method, points in subset.groupby("Method", sort=False):
            points = points.sort_values("Early cycles")
            line, = ax.plot(points["Early cycles"], points["Predicted EOL"], "o-", label=method)
            spread = points.dropna(subset=["Seed SD"])
            if not spread.empty:
                ax.errorbar(spread["Early cycles"], spread["Predicted EOL"],
                            yerr=spread["Seed SD"], fmt="none", color=line.get_color(), capsize=3)
        measured = subset["Measured EOL"].iloc[0]
        if np.isfinite(measured):
            ax.axhline(measured, color="black", linestyle=":", label="Measured EOL")
        ax.set(title=run, xlabel="Number of early cycles used", ylabel="Predicted EOL cycle")
        ax.grid(alpha=0.2)
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.savefig(output / "sequence_rul_comparison.png", dpi=180)
    plt.close(fig)

    cycle_six = results.loc[results["Last cycle seen"].eq(6)]
    print("Cycle-6 comparison")
    print(cycle_six.drop(columns="Seed SD").to_string(
        index=False, na_rep="—", float_format=lambda value: f"{value:.2f}",
        formatters={"Delta Q variance (Ah²)": "{:.3e}".format},
    ))
    metrics = results.groupby("Method").agg(
        MAE_cycles=("Absolute error", "mean"),
        Evaluated=("Absolute error", "count"),
        Cases=("Predicted EOL", "size"),
        Crossings=("Crossings", "sum"),
        Attempts=("Attempts", "sum"),
    )
    print("\nAcross all eligible cutoffs")
    print(metrics.round(2).to_string(na_rep="—"))


def main(argv=None):
    """Load both processed CSVs, report errors, and save Task 4 plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    folder = Path(__file__).resolve().parent
    parser.add_argument("--summary", type=Path,
                        default=folder.parent / "data" / "processed" / "cycle_summary.csv")
    parser.add_argument("--measurements", type=Path,
                        default=folder.parent / "data" / "processed" / "measurements_clean.csv")
    parser.add_argument("--out", "--output-dir", type=Path, default=folder / "results")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive.")

    summary = pd.read_csv(args.summary)
    capacity_column = "discharge_capacity_ah"
    valid = summary.loc[
        summary["has_discharge"].eq(True)
        & summary["Cycle_Index"].ge(REFERENCE_CYCLE)
        & summary[capacity_column].gt(0)
        & np.isfinite(summary[capacity_column])
    ]
    if valid.duplicated(["run_id", "Cycle_Index"]).any():
        parser.error("Expected one capacity per run and cycle.")
    capacities = {}
    for run, group in valid.groupby("run_id"):
        cap = group.set_index("Cycle_Index")[capacity_column].sort_index()
        if REFERENCE_CYCLE in cap.index:
            capacities[run] = cap / cap.loc[REFERENCE_CYCLE]
    if len(capacities) < 2:
        parser.error("Need at least two runs with valid cycle-2 capacity.")

    torch.set_num_threads(1)
    measurements = pd.read_csv(args.measurements, usecols=[
        "run_id", "Cycle_Index", "phase", "Voltage(V)", "Capacity(Ah)",
    ])
    seed_rows = []
    results = compare(
        capacities, delta_q_variances(measurements), args.epochs,
        seed_rows=seed_rows,
    )
    if results.empty:
        parser.error("No eligible early-cycle windows to evaluate.")
    report(results, args.out)
    plot_rul_by_cutoff(
        results, args.out / "RUL distribution_sequence_models.png",
        "Sequence-model RUL by cutoff · median and individual seeds",
        pd.DataFrame(seed_rows, columns=["Run", "Method", "Last cycle seen", "Predicted RUL"]),
    )
    return results


if __name__ == "__main__":
    main()
