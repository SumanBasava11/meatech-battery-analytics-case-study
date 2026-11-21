"""Clean battery cycling exports and calculate features per test run and cycle.

Standardize elapsed time, omit the unused temperature field, and label each
measurement by current direction. Validate the phase counters and calculate
C-rates, observed cycle duration, coulombic efficiency, and energy throughput.
Save both CSVs in data/processed and the efficiency trend in results.
"""

from pathlib import Path
import pandas as pd

# Published nominal capacity for the LR1865SZ model named in both exports.
# Source: https://doi.org/10.3389/fenrg.2021.773838 (Table 1).
NOMINAL_CAPACITY_AH = {
    "LR1865SZ_cycles201217_001_2": 2.5,
    "LR1865SZ_cycles201214_002_4": 2.5,
}
NUMERICAL_COLUMNS = [
    "Data_Point", 
    "Cycle_Index", 
    "Current(A)", 
    "Voltage(V)", 
    "Capacity(Ah)", 
    "Energy(Wh)"
]
TEMPERATURE_COLUMN = "Temperature(℃)"
COUNTERS = {"Capacity(Ah)": "capacity_ah", "Energy(Wh)": "energy_wh"}


def load_measurement_data(raw_folder, run_id):
    """Read a raw export and return measurements ordered by sample identifier.

    Positive current is charge, negative current is discharge, zero is
    rest, and missing current is unknown. Reject ambiguous sample IDs
    and invalid elapsed timestamps so cycle durations remain meaningful.
    """
    csv_file = raw_folder / f"{run_id}.csv"
    run = pd.read_csv(csv_file)

    expected = {*NUMERICAL_COLUMNS, "Test_Time(s)", "Date_Time"}
    missing = expected.difference(run.columns)
    if missing or run.empty:
        raise ValueError(f"{run_id}: {len(run)} rows, missing columns: {sorted(missing)}")

    raw_count = len(run)
    run = run.drop_duplicates().copy()
    for column in NUMERICAL_COLUMNS:
        run[column] = pd.to_numeric(run[column], errors="coerce").replace(
            [float("inf"), float("-inf")], float("nan")
        )

    keys = ["Data_Point", "Cycle_Index"]
    if run[keys].isna().any().any() or not run["Data_Point"].is_unique:
        raise ValueError(f"{run_id}: missing identifiers or repeated Data_Point values")
    run = run.sort_values("Data_Point").reset_index(drop=True)

    # Examples: 00:41:28 and 1-00:00:00 (one day into the test).
    time_text = run["Test_Time(s)"].astype("string").str.strip()
    time_text = time_text.str.replace(r"^(\d+)-", r"\1 days ", regex=True)
    elapsed = pd.to_timedelta(time_text, errors="coerce").dt.total_seconds()
    if elapsed.isna().any() or not elapsed.is_monotonic_increasing:
        raise ValueError(f"{run_id}: invalid or backward elapsed times in Data_Point order")
    run["elapsed_seconds"] = elapsed
    run = run.drop(columns=TEMPERATURE_COLUMN, errors="ignore")

    current = run["Current(A)"]
    run["phase"] = "unknown"
    run.loc[current > 0, "phase"] = "charge"
    run.loc[current == 0, "phase"] = "rest"
    run.loc[current < 0, "phase"] = "discharge"
    run["run_id"] = run_id
    print(f"{run_id}: {len(run):,} measurements, {raw_count - len(run):,} duplicates removed")
    return run

def summarize_cycles(run, run_id, nominal_capacity_ah):
    """Return one row per cycle from measurements sorted by Data_Point.

    Duration is the observed time span in seconds. Phase C-rate is median
    absolute current divided by nominal Ah. Coulombic efficiency is 100
    times discharge Ah divided by charge Ah. Throughput sums charge and
    discharge Wh. Leave unavailable features missing and flag bad counters.
    """
    cycles = run.groupby("Cycle_Index")
    bounds = cycles["elapsed_seconds"].agg(["min", "max"])
    table = pd.DataFrame(index=bounds.index)
    table["run_id"] = run_id
    table["observed_duration_s"] = bounds["max"] - bounds["min"]
    table["has_unknown_phase"] = cycles["phase"].agg(lambda phase: phase.eq("unknown").any())
    table["nominal_capacity_ah"] = nominal_capacity_ah

    for phase in ("charge", "discharge"):
        active = run.loc[run["phase"].eq(phase)]
        first = active.drop_duplicates("Cycle_Index").set_index("Cycle_Index")
        last = active.drop_duplicates("Cycle_Index", keep="last").set_index("Cycle_Index")
        table[f"has_{phase}"] = table.index.isin(last.index)

        counter_issue = pd.Series(False, index=table.index)
        for counter, output_column in COUNTERS.items():
            # These exports reset both counters to zero at the start of each active phase.
            # Compare successive recorded values, even across missing samples.
            recorded = active.dropna(subset=[counter])
            decreases = (
                recorded.groupby("Cycle_Index")[counter].diff().lt(0)
                .groupby(recorded["Cycle_Index"]).any()
                .reindex(first.index, fill_value=False)
            )
            valid_counter = first[counter].eq(0) & last[counter].notna() & ~decreases
            table[f"{phase}_{output_column}"] = last[counter].where(valid_counter)
            counter_issue.loc[valid_counter.index] |= ~valid_counter
        table[f"{phase}_counter_issue"] = counter_issue

        median_current = active["Current(A)"].abs().groupby(active["Cycle_Index"]).median()
        table[f"{phase}_median_current_a"] = median_current
        table[f"{phase}_c_rate"] = median_current / nominal_capacity_ah

    charge = table["charge_capacity_ah"]
    efficiency = 100 * table["discharge_capacity_ah"] / charge.where(charge > 0)
    table["coulombic_efficiency_pct"] = efficiency.mask(table["has_unknown_phase"])
    table["efficiency_above_100"] = table["coulombic_efficiency_pct"].gt(100)
    throughput = table["charge_energy_wh"] + table["discharge_energy_wh"]
    table["energy_throughput_wh"] = throughput.mask(table["has_unknown_phase"])
    return table.reset_index()

def plot_coulombic_efficiency(summary, output_path):
    """Save efficiency by run and cycle, leaving missing values as gaps."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    runs = summary.sort_values(["run_id", "Cycle_Index"]).groupby("run_id")
    fig = Figure(figsize=(5.8 * runs.ngroups, 4), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, runs.ngroups, sharey=True, squeeze=False)[0]

    for ax, (run_id, run) in zip(axes, runs):
        ax.plot(run["Cycle_Index"], run["coulombic_efficiency_pct"],
                color="#2563EB", marker=".", markersize=4)
        ax.axhline(100, color="#64748B", linestyle="--", linewidth=1)
        ax.set(title=run_id.removeprefix("LR1865SZ_cycles"), xlabel="Cycle count")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Coulombic efficiency (%)")
    fig.suptitle("Coulombic efficiency by cycle")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white")


def main():
    """Process the configured runs and write analysis inputs and the plot."""
    project_root = Path(__file__).resolve().parent.parent
    data_folder = project_root / "data"
    measurements, cycle_tables = [], []
    for run_id, nominal_capacity_ah in NOMINAL_CAPACITY_AH.items():
        run = load_measurement_data(data_folder / "raw", run_id)
        measurements.append(run)
        cycle_tables.append(summarize_cycles(run, run_id, nominal_capacity_ah))

    processed = data_folder / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.concat(measurements, ignore_index=True).to_csv(processed / "measurements_clean.csv", index=False)
    summary = pd.concat(cycle_tables, ignore_index=True)
    summary.to_csv(processed / "cycle_summary.csv", index=False)
    plot_path = project_root / "results" / "coulombic_efficiency_trends.png"
    plot_coulombic_efficiency(summary, plot_path)
    print(f"Saved coulombic efficiency plot to {plot_path}")
    print(f"Saved measurement and cycle CSVs in {processed} ({len(summary):,} cycle rows)")


if __name__ == "__main__":
    main()
