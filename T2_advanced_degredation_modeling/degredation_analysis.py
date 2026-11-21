"""Compare cell capacity forecasts and examine differential-voltage signatures.

Dependencies: numpy, pandas, scipy, scikit-learn, torch, lmfit, matplotlib.
Fit on cycles 2–80 and score forecasts on cycles 81–100. Compare discharge
curves at cycles 2, 50, and 100. Save three PNGs and print held-out errors.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.signal import savgol_filter

KEYS = ["run_id", "Cycle_Index"]
CUTOFF = 80
DVA_CYCLES = [2, 50, 100]
GRID_STEP_AH = 0.002
WINDOWS = (11, 21)

def exponential_forecast(n_train, q_train, n_test):
    """Fit Q = A exp(-n / decay) with lmfit and report A and 1 / decay."""
    from lmfit.models import ExponentialModel

    model = ExponentialModel()
    params = model.make_params(
        amplitude={"value": q_train[0], "min": 0},
        decay={"value": 100, "min": 1e-6},
    )
    fitted = model.fit(q_train, params, x=n_train)
    return fitted.eval(x=n_test), {
        "exponential_start_ah": fitted.params["amplitude"].value,
        "exponential_decay_per_cycle": 1 / fitted.params["decay"].value,
    }

def sqrt_time_forecast(elapsed, q_train, cycles_ahead):
    """Fit a nonincreasing capacity curve against square-root elapsed hours.

    Estimate future timestamps from training intervals only. This empirical
    relationship is physics-informed and does not identify a fade mechanism.
    """
    # Estimate future cycle-end times from the last ten training intervals.
    hours_per_cycle = np.median(np.diff(elapsed)[-10:])
    forecast_time = elapsed[-1] + cycles_ahead * hours_per_cycle
    reference_time = np.sqrt(elapsed[0])
    train_offset = np.sqrt(elapsed) - reference_time
    test_offset = np.sqrt(forecast_time) - reference_time

    # Fit Q = Q_start - fade_rate * offset with both coefficients nonnegative.
    train_design = np.column_stack([np.ones_like(train_offset), -train_offset])
    coefficients, _ = nnls(train_design, q_train)
    test_design = np.column_stack([np.ones_like(test_offset), -test_offset])
    return test_design @ coefficients, {
        "sqrt_time_start_ah": coefficients[0],
        "sqrt_time_fade_ah_per_sqrt_h": coefficients[1],
        "forecast_hours_per_cycle": hours_per_cycle,
    }

def lstm_forecast(q_train, horizon):
    """Fit an 8-unit LSTM and recursively predict without test measurements."""
    import torch
    from sklearn.preprocessing import MinMaxScaler
    from torch import nn

    class CapacityLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=8, batch_first=True)
            self.output = nn.Linear(8, 1)

        def forward(self, inputs):
            sequence, _ = self.lstm(inputs)
            return self.output(sequence[:, -1, :])

    window_size = 5
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(q_train[:, None])
    windows = np.stack([
        scaled[i - window_size:i] for i in range(window_size, len(scaled))
    ])
    inputs = torch.tensor(windows, dtype=torch.float32)
    targets = torch.tensor(scaled[window_size:], dtype=torch.float32)
    model = CapacityLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    window = scaled[-window_size:].copy()
    predictions = []
    with torch.no_grad():
        for _ in range(horizon):
            next_capacity = model(torch.tensor(window[None], dtype=torch.float32)).item()
            predictions.append(next_capacity)
            window = np.vstack([window[1:], [[next_capacity]]])

    return scaler.inverse_transform(np.asarray(predictions)[:, None]).ravel()

def compare_capacity_models(cycles, measurements):
    """Fit four models per run and return held-out forecasts and errors.

    Train on cycles 2–80. Report MAE, RMSE, and signed prediction bias in Ah.
    Held-out measurements are used to evaluate forecasts.
    """
    import torch

    end_times = (
        measurements.groupby(KEYS, as_index=False)["elapsed_seconds"]
        .max()
        .assign(elapsed_test_h=lambda rows: rows["elapsed_seconds"] / 3600)
    )
    model_data = (
        cycles.loc[cycles["Cycle_Index"].between(2, 100), KEYS + ["discharge_capacity_ah"]]
        .merge(end_times[KEYS + ["elapsed_test_h"]], on=KEYS, how="left", validate="one_to_one")
        .sort_values(KEYS)
    )
    if model_data.empty or not np.isfinite(
        model_data[["discharge_capacity_ah", "elapsed_test_h"]].to_numpy()
    ).all():
        raise ValueError("Missing or invalid discharge capacity or elapsed time in cycles 2–100.")

    # Seed once before the sorted run loop to match the notebook training order.
    torch.manual_seed(42)
    scores, forecasts, parameters = [], [], []
    for run_id, run in model_data.groupby("run_id"):
        if run["Cycle_Index"].tolist() != list(range(2, 101)):
            raise ValueError(f"{run_id} must contain exactly cycles 2–100.")
        train = run.loc[run["Cycle_Index"].le(CUTOFF)]
        test = run.loc[run["Cycle_Index"].gt(CUTOFF)]
        n_train = train["Cycle_Index"].to_numpy() - 2
        n_test = test["Cycle_Index"].to_numpy() - 2
        q_train = train["discharge_capacity_ah"].to_numpy()
        actual = test["discharge_capacity_ah"].to_numpy()
        elapsed = train["elapsed_test_h"].to_numpy()
        if elapsed[0] < 0 or np.any(np.diff(elapsed) <= 0):
            raise ValueError(f"{run_id} has invalid training cycle-end times.")

        linear_params = np.polyfit(n_train, q_train, 1)
        exp_predictions, exp_parameters = exponential_forecast(n_train, q_train, n_test)
        cycles_ahead = test["Cycle_Index"].to_numpy() - CUTOFF
        sqrt_predictions, sqrt_parameters = sqrt_time_forecast(elapsed, q_train, cycles_ahead)

        parameters.append({
            "run_id": run_id,
            "linear_intercept_ah": linear_params[1],
            "linear_slope_ah_per_cycle": linear_params[0],
            **exp_parameters,
            **sqrt_parameters,
        })

        predictions = [
            ("Statistical", "Linear baseline", np.polyval(linear_params, n_test)),
            ("Statistical", "Exponential", exp_predictions),
            ("Physics-informed", "Square-root time", sqrt_predictions),
            ("Machine learning", "LSTM (8 units)", lstm_forecast(q_train, len(test))),
        ]
        for approach, model, predicted in predictions:
            error = predicted - actual
            scores.append({
                "run_id": run_id, "approach": approach, "model": model,
                "test_mae_ah": np.abs(error).mean(),
                "test_rmse_ah": np.sqrt(np.mean(error ** 2)),
                "test_bias_ah": error.mean(),
            })
            forecasts.append(test[KEYS + ["discharge_capacity_ah"]].assign(
                model=model, predicted_ah=predicted, error_ah=error
            ))

    return {
        "model_comparison": pd.DataFrame(scores),
        "capacity_forecasts": pd.concat(forecasts, ignore_index=True),
        "model_parameters": pd.DataFrame(parameters),
    }


def differential_voltage(measurements, run_ids):
    """Estimate dV/dQ in V/Ah on each run's shared discharge-capacity grid.

    Compare two Savitzky–Golay windows and trim their boundary regions.
    Changes in the curves are clues for follow-up electrode analysis.
    """
    selected = measurements.loc[
        measurements["run_id"].isin(run_ids)
        & measurements["Cycle_Index"].isin(DVA_CYCLES)
        & measurements["phase"].eq("discharge")
    ].sort_values(KEYS + ["Data_Point"])
    selected = selected.loc[
        np.isfinite(selected["Capacity(Ah)"]) & np.isfinite(selected["Voltage(V)"])
    ]
    bounds = selected.groupby(KEYS)["Capacity(Ah)"].agg(start_ah="min", end_ah="max")
    shared = bounds.groupby(level="run_id").agg(
        shared_start_ah=("start_ah", "max"), shared_end_ah=("end_ah", "min")
    )

    derivatives = []
    edge = max(WINDOWS) // 2
    for run_id in run_ids:
        run = selected.loc[selected["run_id"].eq(run_id)]
        if sorted(run["Cycle_Index"].unique()) != DVA_CYCLES:
            raise ValueError(f"{run_id} needs discharge traces for cycles {DVA_CYCLES}.")
        start, end = shared.loc[run_id]
        if end <= start:
            raise ValueError(f"{run_id} has no shared discharge-capacity range.")
        grid = np.arange(start, end, GRID_STEP_AH)
        if len(grid) <= 2 * edge:
            raise ValueError(f"{run_id} has too little overlap for the smoothing windows.")

        for cycle, curve in run.groupby("Cycle_Index"):
            # Average repeated capacity readings before linear interpolation.
            voltage = curve.groupby("Capacity(Ah)")["Voltage(V)"].mean()
            voltage_grid = np.interp(grid, voltage.index.to_numpy(), voltage.to_numpy())
            derivative = pd.DataFrame({
                "run_id": run_id, "cycle": cycle,
                "capacity_ah": grid, "voltage_v": voltage_grid,
            })
            for window in WINDOWS:
                derivative[f"dvdq_{window}pt"] = savgol_filter(
                    voltage_grid, window, 3, deriv=1, delta=GRID_STEP_AH
                )
            # Compare both windows only where the wider window fits fully.
            derivatives.append(derivative.iloc[edge:-edge])

    return pd.concat(derivatives, ignore_index=True)

def load_results(processed_dir):
    """Calculate Task 2 results in memory from the two Task 1 CSVs."""
    cycles = pd.read_csv(processed_dir / "cycle_summary.csv")
    measurements = pd.read_csv(processed_dir / "measurements_clean.csv")
    results = compare_capacity_models(cycles, measurements)
    run_ids = results["model_comparison"]["run_id"].unique()
    results["dvdq_curves"] = differential_voltage(measurements, run_ids)
    results["cycle_overview"] = cycles
    return results


def save_pngs(results, output_dir):
    """Save the three static figures without a browser dependency."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator

    scores = results["model_comparison"]
    forecasts = results["capacity_forecasts"]
    cycles = results["cycle_overview"]
    runs = scores["run_id"].unique()
    models = scores["model"].unique()
    colors = ["#64748B", "#2563EB", "#B45309", "#0F766E"]
    cycle_colors = dict(zip(DVA_CYCLES, colors[1:]))
    output_dir.mkdir(parents=True, exist_ok=True)

    def canvas(rows, cols, *, sharey=False):
        fig = Figure(figsize=(max(8, 6 * cols), 4 * rows), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(rows, cols, squeeze=False, sharey=sharey).ravel()
        for ax in axes:
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        return fig, axes

    def save(fig, title, target):
        fig.suptitle(title, fontsize=14)
        fig.savefig(target, dpi=180, facecolor="white")

    def run_panels(x_label, y_label):
        fig, axes = canvas(1, len(runs), sharey=True)
        axes[0].set_ylabel(y_label)
        for ax, run in zip(axes, runs):
            ax.set(title=run.removeprefix("LR1865SZ_cycles"), xlabel=x_label)
        return fig, axes

    fig, axes = run_panels("Cycle count", "Discharge capacity (Ah)")
    for ax, run in zip(axes, runs):
        actual = cycles.loc[cycles["run_id"].eq(run)]
        for start, end, label, color in [
            (2, CUTOFF, "Training measurements", "#94A3B8"),
            (CUTOFF + 1, 100, "Held-out measurements", "#111827"),
        ]:
            rows = actual.loc[actual["Cycle_Index"].between(start, end)]
            ax.scatter(rows["Cycle_Index"], rows["discharge_capacity_ah"],
                       s=13, color=color, label=label, zorder=3)
        for color, model in zip(colors, models):
            curve = forecasts.loc[
                forecasts["run_id"].eq(run) & forecasts["model"].eq(model)
            ].sort_values("Cycle_Index")
            ax.plot(curve["Cycle_Index"], curve["predicted_ah"], color=color, label=model)
        ax.axvline(CUTOFF + 0.5, color="#64748B", linestyle="--", linewidth=1)
    axes[0].legend(fontsize=8, frameon=False)
    save(fig, "Capacity vs. cycle count with predictive curves",
         output_dir / "Capacity vs. cycle count with predictive curves.png")

    fig, axes = run_panels("Discharged capacity (Ah)", "dV/dQ (V/Ah)")
    dvdq = results["dvdq_curves"]
    for ax, run in zip(axes, runs):
        for cycle, curve in dvdq.loc[dvdq["run_id"].eq(run)].groupby("cycle"):
            ax.plot(curve["capacity_ah"], curve["dvdq_11pt"],
                    color=cycle_colors[cycle], label=f"Cycle {cycle}")
            ax.plot(curve["capacity_ah"], curve["dvdq_21pt"],
                    color=cycle_colors[cycle], linestyle="--")
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "Differential voltage · Solid 11-point · Dashed 21-point",
         output_dir / "dvdq_curves.png")

    fig, axes = canvas(1, 2, sharey=True)
    positions = np.arange(len(runs))
    width = 0.8 / len(models)
    labels = [run.removeprefix("LR1865SZ_cycles") for run in runs]
    metrics = ["test_mae_ah", "test_rmse_ah"]
    max_error = scores[metrics].to_numpy().max()
    axes[0].set_ylim(0, max_error * 1.2 + 0.001)
    for ax, metric, title in zip(axes, metrics, ["MAE", "RMSE"]):
        values = scores.pivot(index="run_id", columns="model", values=metric)
        for index, (model, color) in enumerate(zip(models, colors)):
            offset = (index - (len(models) - 1) / 2) * width
            bars = ax.bar(positions + offset, values.loc[runs, model], width * 0.94,
                          color=color, label=model)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set_title(title)
        ax.set_xticks(positions, labels)
    axes[0].set_ylabel("Error (Ah)")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "Discharge capacity forecast errors · Cycles 81–100",
         output_dir / "model_comparison.png")


def run_analysis(processed_dir, output_dir):
    """Save the three Task 2 figures as PNGs."""
    results = load_results(processed_dir)
    save_pngs(results, output_dir)
    return results["model_comparison"]


def main():
    """Read CLI paths, run the analysis, and print held-out scores."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--processed-dir", type=Path, default=project_root / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    scores = run_analysis(args.processed_dir, args.output_dir)
    print(scores.round(4).to_string(index=False))
    print(f"Saved capacity forecast plot to {args.output_dir / 'Capacity vs. cycle count with predictive curves.png'}")
    print(f"Saved all Task 2 plots to {args.output_dir}")


if __name__ == "__main__":
    main()
    