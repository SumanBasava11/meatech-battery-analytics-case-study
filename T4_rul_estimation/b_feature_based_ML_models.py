"""Estimate remaining useful life from early-cycle features-based Ridge regression model.

Hold out one run at a time. Compare capacity fraction, recent fade slope,
and forecast horizon with variants that add discharge-curve ΔQ variance
and mean charge duration. Use 80% of measured cycle-2 capacity as EOL.
"""


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .a_capacity_fade_extrapolation import FRACTION, REFERENCE_CYCLE, measured_eol, plot_rul_by_cutoff
else:
    from a_capacity_fade_extrapolation import FRACTION, REFERENCE_CYCLE, measured_eol, plot_rul_by_cutoff

CUTOFF = 6
LAST_CUTOFF = 11
HORIZON = 30
BASE_FEATURES = ["capacity_fraction", "recent_slope", "horizon"]

def features_at_cycle(capacity, curves, charge_minutes, run, cycle):
    """Build features from data available through the forecast cycle.

    Return the latest capacity fraction, its five-cycle slope, and the
    mean charge duration in minutes over those same cycles. Capacities
    are normalized to cycle 2. Compute variance in Ah² of the difference
    from the cycle-2 discharge curve on 200 shared voltage points.
    Return None if the feature window is incomplete.
    """
    recent = capacity.reindex(range(cycle - 4, cycle + 1))
    if recent.isna().any():
        return None

    durations = charge_minutes.get(run)
    if durations is None:
        return None
    durations = durations.reindex(recent.index)
    if durations.isna().any():
        return None

    reference = curves.get((run, REFERENCE_CYCLE))
    current = curves.get((run, cycle))
    if reference is None or current is None or min(len(reference), len(current)) < 2:
        return None

    # Use the shared voltage interval for this pair, avoiding interpolation beyond either curve.
    low = max(reference.index.min(), current.index.min())
    high = min(reference.index.max(), current.index.max())
    if low >= high:
        return None
    voltage = np.linspace(low, high, 200)
    delta_q = np.interp(voltage, current.index, current.values)
    delta_q -= np.interp(voltage, reference.index, reference.values)
    return {
        "capacity_fraction": recent.iloc[-1],
        "recent_slope": np.polyfit(recent.index, recent, 1)[0],
        "delta_q_variance": np.var(delta_q),
        "mean_charge_minutes": durations.mean(),
    }

def training_examples(capacity, curves, charge_minutes, run):
    """Pair complete pre-EOL windows with future measured capacity.

    Horizon is the number of cycles from the window end to the target,
    capped at 30. The target is discharge capacity relative to cycle 2.
    """
    rows = []
    for cycle in capacity.loc[CUTOFF:].index:
        if capacity.loc[:cycle].lt(FRACTION).any():
            break
        features = features_at_cycle(capacity, curves, charge_minutes, run, cycle)
        if features is None:
            continue
        for future, value in capacity.loc[cycle + 1:cycle + HORIZON].items():
            rows.append(dict(features, horizon=future - cycle, target=value))
    return pd.DataFrame(rows)

def predicted_eol(forecast, initial_capacity, cutoff=CUTOFF):
    """Interpolate the first drop below 80% of cycle-2 capacity.

    Anchor the forecast at the measured capacity at the cutoff. Return NaN
    if the threshold is never crossed. RUL is the EOL cycle minus the cutoff.
    """
    # The last measured capacity anchors the first interpolation interval.
    previous = initial_capacity
    for horizon, capacity in enumerate(forecast, start=1):
        if capacity < FRACTION:
            fraction = (previous - FRACTION) / (previous - capacity)
            return cutoff + horizon - 1 + fraction
        previous = capacity
    return np.nan

def main(argv=None):
    """Evaluate Ridge forecasts and return the data used for notebook diagnostics.

    Train on the other runs and forecast at each eligible early cutoff.
    Compare the baseline, baseline plus ΔQ variance, and the model with both
    ΔQ variance and charge duration. Fit the scaler on training runs only.
    Print EOL and RUL results and return cycle-6 inputs for diagnostics.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"
    parser.add_argument("--summary", type=Path, default=processed / "cycle_summary.csv")
    parser.add_argument(
        "--measurements", type=Path, default=processed / "measurements_clean.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)

    summary = pd.read_csv(args.summary)
    valid = summary.loc[
        summary["has_discharge"].eq(True)
        & summary["Cycle_Index"].ge(REFERENCE_CYCLE)
        & summary["discharge_capacity_ah"].gt(0)
        & np.isfinite(summary["discharge_capacity_ah"])
    ]

    keys = ["run_id", "Cycle_Index"]
    if valid.duplicated(keys).any():
        parser.error("Expected one capacity per run and cycle.")

    # Normalize each run's capacity to its measured capacity at cycle 2.
    capacities = {}
    for run, group in valid.groupby("run_id"):
        capacity = group.set_index("Cycle_Index")["discharge_capacity_ah"].sort_index()
        if REFERENCE_CYCLE in capacity.index:
            capacities[run] = capacity / capacity.loc[REFERENCE_CYCLE]

    if len(capacities) < 2:
        parser.error("Need at least two runs with valid cycle-2 capacity.")

    raw = pd.read_csv(
        args.measurements,
        usecols=[*keys, "phase", "Voltage(V)", "Capacity(Ah)", "elapsed_seconds"],
    )
    discharge = raw.loc[
        raw["phase"].eq("discharge")
        & np.isfinite(raw["Voltage(V)"])
        & np.isfinite(raw["Capacity(Ah)"])
    ]
    curves = {
        key: group.groupby("Voltage(V)")["Capacity(Ah)"].mean().sort_index()
        for key, group in discharge.groupby(keys)
    }

    charge = raw.loc[raw["phase"].eq("charge") & np.isfinite(raw["elapsed_seconds"])]
    bounds = charge.groupby(keys)["elapsed_seconds"].agg(["min", "max"])
    spans = (bounds["max"] - bounds["min"]) / 60
    charge_minutes = {
        run: values.droplevel("run_id")
        for run, values in spans.where(spans.gt(0)).groupby(level="run_id")
    }


    examples = {
        run: training_examples(capacity, curves, charge_minutes, run)
        for run, capacity in capacities.items()
    }
    results = []
    diagnostic_data = {}

    for run, cap in capacities.items():
        train = pd.concat([rows for other, rows in examples.items() if other != run])
        if train.empty:
            print(f"{run}: no eligible training examples from other runs.")
            continue

        models = {}
        for label, columns in [
            ("Baseline", BASE_FEATURES),
            ("+ ΔQ variance", [*BASE_FEATURES, "delta_q_variance"]),
            (
                "+ Charge duration",
                [*BASE_FEATURES, "delta_q_variance", "mean_charge_minutes"],
            ),
        ]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(train[columns], train["target"])
            models[label] = model, columns

        measured, _ = measured_eol(cap, FRACTION)
        for cutoff in range(CUTOFF, LAST_CUTOFF + 1):
            early_capacity = cap.loc[:cutoff]
            features = features_at_cycle(early_capacity, curves, charge_minutes, run, cutoff)
            if features is None or early_capacity.lt(FRACTION).any():
                continue
            horizons = range(1, int(train["horizon"].max()) + 1)
            test = pd.DataFrame([dict(features, horizon=h) for h in horizons])

            # Collect all three forecasts for this held-out run.
            forecasts = {}
            for label, (model, columns) in models.items():
                forecast = model.predict(test[columns])
                forecasts[label] = forecast
                eol = predicted_eol(forecast, features["capacity_fraction"], cutoff)

                results.append({
                    "Mean charge (min)": features["mean_charge_minutes"],
                    "Held-out run": run.removeprefix("LR1865SZ_cycles"),
                    "Ridge": label,
                    "Predicted EOL": eol,
                    "Measured EOL": measured,
                    "Last cycle seen": cutoff,
                    "Predicted cycles remaining": eol - cutoff,
                    "Measured RUL": measured - cutoff if measured > cutoff else np.nan,
                    "Absolute EOL error": abs(eol - measured),
                    "Horizon tested": len(horizons),
                })


            # Save the inputs and predictions the notebook needs.
            if cutoff == CUTOFF:
                diagnostic_data[run] = {
                    "training_features": train[list(features)].copy(),
                    "cycle_6_features": pd.Series(features),
                    "measured_capacity": cap,
                    "forecasts": forecasts,
                }

    if not results:
        parser.error("No held-out run could be evaluated.")

    print(f"Forecasts from eligible cutoffs starting at cycle {CUTOFF}. EOL threshold is 80% of measured cycle-2 capacity.")
    print("A blank predicted EOL means the forecast did not fall below the threshold within the tested horizon.\n")
    table = pd.DataFrame(results)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}", na_rep="—"))

    # Average errors over runs with numerical results for all three variants.
    paired = table.loc[table["Last cycle seen"].eq(CUTOFF)].pivot(
        index="Held-out run", columns="Ridge", values="Absolute EOL error"
    ).dropna()

    if not paired.empty:
        print(f"\nCycle-6 mean absolute EOL error across {len(paired)} paired runs (cycles)")
        print(paired.mean().round(2).to_string())
    else:
        print("\nNo paired numerical EOL errors to average.")

    plot_rul_by_cutoff(
        table.rename(columns={
            "Held-out run": "Run", "Ridge": "Method",
            "Predicted cycles remaining": "Predicted RUL",
        }),
        args.output_dir / "RUL distribution_feature_based.png",
        "Feature-based Ridge RUL forecasts by cutoff",
    )
    return diagnostic_data


if __name__ == "__main__":
    main()


