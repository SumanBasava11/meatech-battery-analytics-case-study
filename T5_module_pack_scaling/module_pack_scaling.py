"""Scale measured cell capacity fade to an illustrative four-module 8S4P pack.

Compare cell variation, a prescribed temperature gradient, and usable
capacity lost to residual state-of-charge mismatch after balancing.
Benchmark three model resolutions against the cell-resolved simulation.
Pack topology and stress parameters are assumptions. The two cell runs
calibrate the reference fade rate, and pack measurements would be needed
to validate the resulting capacity projections.
"""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULES, SERIES, PARALLEL = 4, 8, 4 # Gives 32 groups and 128 cells in total.
REFERENCE_CYCLE, FIT_END = 2, 80
INITIAL_AH = 2.5
RESULTS = Path(__file__).resolve().parent / "results"


def fit_cell_rates(summary_path):
    """Fit one exponential fade rate per run using cycles 2–80."""
    data = pd.read_csv(summary_path)
    data = data.loc[
        data["has_discharge"].eq(True)
        & data["Cycle_Index"].between(REFERENCE_CYCLE, FIT_END)
        & np.isfinite(data["discharge_capacity_ah"])
        & data["discharge_capacity_ah"].gt(0)
    ]
    rows = []
    for run, records in data.groupby("run_id"):
        capacity = records.set_index("Cycle_Index")["discharge_capacity_ah"].sort_index()
        if not capacity.index.is_unique:
            raise ValueError(f"{run}: duplicate cycle indices.")
        if REFERENCE_CYCLE not in capacity or len(capacity) < 3:
            continue
        x = capacity.index.to_numpy() - REFERENCE_CYCLE
        y = np.log(capacity.to_numpy() / capacity.loc[REFERENCE_CYCLE])
        rate = -np.dot(x, y) / np.dot(x, x)
        if rate > 0:
            rows.append((run, rate))
    if not rows:
        raise ValueError("No run with valid cycle-2 capacity and a positive fade rate.")
    return pd.DataFrame(rows, columns=["Cell run", "Fade rate per cycle"])

def cell_inputs(rate):
    """Assign cell capacity and fade-rate variation with a fixed random seed.

    Use 3% initial-capacity spread, 15% fade-rate spread, and a prescribed
    25–35 °C gradient. These are illustrative inputs.
    """
    rng = np.random.default_rng(42)
    shape = (MODULES, SERIES, PARALLEL)
    capacity = INITIAL_AH * np.clip(rng.normal(1, 0.03, shape), 0.85, 1.15)
    fade = rate * np.clip(rng.normal(1, 0.15, shape), 0.5, 1.5)
    temperature = np.broadcast_to(
        25 + np.linspace(0, 5, MODULES)[:, None, None]
        + np.linspace(0, 5, SERIES)[None, :, None], shape
    )
    return capacity, fade, temperature


def simulate(cycles, capacity, rate, temperature, headroom_loss=0):
    """Advance cell fade and approximate usable capacity for a series pack.

    Parallel cells contribute their Ah to each group. A group-specific
    allowance represents inaccessible charge from residual SOC mismatch.
    The first limiting group caps module and pack Ah. The assumed fade
    multiplier is 1.5 per 10 °C above 25 °C.
    """
    effective_rate = rate * 1.5 ** ((temperature - 25) / 10)
    cell_ah = capacity * np.exp(-cycles[:, None, None, None] * effective_rate)
    group_ah = cell_ah.sum(axis=-1) * (1 - headroom_loss)
    module_ah = group_ah.min(axis=-1)
    return module_ah, module_ah.min(axis=-1)

def analyze(summary_path=None, horizon=100):
    """Return calibration, pack trajectories, module summaries, and timings.

    Add one stress factor per scenario. Represent imperfect balancing
    as a fixed 0.5–4% usable-capacity loss across parallel groups.
    Compare two cheaper approximations to the cell-resolved simulation
    using runtime and mean absolute capacity difference over all cycles.
    """
    if horizon < 1:
        raise ValueError("The forecast horizon must be at least one cycle.")
    calibration = (
        fit_cell_rates(summary_path) if summary_path is not None else
        pd.DataFrame([("Illustrative rate", 0.004)], columns=["Cell run", "Fade rate per cycle"])
    )
    rate = calibration["Fade rate per cycle"].median()
    cycles = np.arange(horizon + 1)
    shape = (MODULES, SERIES, PARALLEL)
    cell_ah, cell_rate, temperature = cell_inputs(rate)
    uniform_ah = np.full(shape, INITIAL_AH)
    uniform_rate = np.full(shape, rate)
    ambient = np.full(shape, 25.0)
    mismatch = np.linspace(0.005, 0.04, MODULES * SERIES).reshape(MODULES, SERIES)

    scenarios = {
        "Uniform cells": (uniform_ah, uniform_rate, ambient, 0),
        "+ Cell variation": (cell_ah, cell_rate, ambient, 0),
        "+ Temperature gradient": (cell_ah, cell_rate, temperature, 0),
        "+ Residual SOC mismatch": (cell_ah, cell_rate, temperature, mismatch),
    }
    traces = pd.DataFrame({"Cycles since reference": cycles})
    tables = []
    for label, inputs in scenarios.items():
        modules, pack = simulate(cycles, *inputs)
        traces[label] = pack
        tables.append({
            "Scenario": label,
            "Initial usable pack Ah": pack[0],
            "Final usable pack Ah": pack[-1],
            "Final retention vs own start (%)": 100 * pack[-1] / pack[0],
            **{f"Module {i + 1} final Ah": value for i, value in enumerate(modules[-1])},
        })

    # Keep group losses in the grouped model and use the worst loss in the pack average.
    full = traces["+ Residual SOC mismatch"].to_numpy()
    approximations = {
        "Pack-average cell": (
            np.full((1, 1, 1), PARALLEL * cell_ah.mean()),
            cell_rate.mean(), temperature.mean(), mismatch.max()),
        "One representative cell per parallel group": (
            cell_ah.sum(axis=-1, keepdims=True), cell_rate.mean(axis=-1, keepdims=True),
            temperature.mean(axis=-1, keepdims=True), mismatch),
        "Cell-resolved reference": scenarios["+ Residual SOC mismatch"],
    }
    timing = []
    for label, inputs in approximations.items():
        samples = []
        for _ in range(15):
            start = perf_counter()
            _, pack = simulate(cycles, *inputs)
            samples.append(1000 * (perf_counter() - start))
        timing.append({
            "Resolution": label, "Median simulation time (ms)": np.median(samples),
            "MAE vs cell-resolved model (Ah)": np.abs(pack - full).mean(),
            "Final pack capacity (Ah)": pack[-1],
        })
    return calibration, traces, pd.DataFrame(tables), pd.DataFrame(timing)

def save_plots(traces, timing, output=RESULTS):
    """Save capacity trajectories and the resolution benchmark as PNGs."""
    output.mkdir(parents=True, exist_ok=True)
    ax = traces.plot(x="Cycles since reference", figsize=(8, 4.5),
                     ylabel="Usable pack capacity (Ah)", title="Pack capacity across scenarios")
    ax.legend(frameon=False)
    ax.figure.savefig(output / "pack_capacity_scenarios.png", dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    labels = timing["Resolution"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, column, title in zip(axes, timing.columns[1:3],
                                ["Simulation runtime (ms)", "MAE vs cell-resolved model (Ah)"]):
        bars = ax.barh(labels, timing[column], color="#2563EB")
        ax.bar_label(bars, fmt="%.3g", padding=3)
        ax.margins(x=0.25)
        ax.set_xlabel(title)
        ax.invert_yaxis()
    fig.savefig(output / "scaling_accuracy_runtime.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run the scaling exercise and print its analysis tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--summary", type=Path)
    inputs.add_argument("--demo", action="store_true", help="Use an illustrative cell fade rate.")
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()
    summary = args.summary
    if summary is None and not args.demo:
        summary = Path(__file__).resolve().parent.parent / "data" / "processed" / "cycle_summary.csv"
    calibration, traces, scenarios, timing = analyze(summary, args.horizon)
    for heading, table in [("Cell calibration", calibration), ("Pack scenarios", scenarios),
                           ("Resolution benchmark", timing)]:
        print(f"\n{heading}\n{table.round(5).to_string(index=False)}")
    save_plots(traces, timing)
    print(f"\nPlots saved under {RESULTS}")


if __name__ == "__main__":
    main()