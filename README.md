Advanced Cell-Level Degradation Analysis & Scaling

This case study analyzes laboratory cycling data from two LR1865SZ lithium-ion cell runs. It covers preprocessing, capacity fade, cycle anomalies, remaining useful life (RUL), and an illustrative module and pack simulation.

Local setup

Use Python 3.11 and run the commands from the repository root. Open that folder in VS Code if you want to run the notebooks.

Windows PowerShell

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1

If PowerShell blocks activation, use .\.venv\Scripts\python.exe in place of python in the commands below.

macOS or Linux

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

Verify the environment before running the models:

python -m pip check
python -c "import numpy, pandas, scipy, sklearn, matplotlib, lmfit, torch"

For notebooks, install the VS Code Python and Jupyter extensions, open the notebook, select the .venv interpreter as its kernel, and run the cells in order.

Input data

Place the original CSV exports supplied with the case study in data/raw/ using these names:

LR1865SZ_cycles201217_001_2.csv

LR1865SZ_cycles201214_002_4.csv

Task 1 expects the export headers Data_Point, Cycle_Index, Current(A), Voltage(V), Capacity(Ah), Energy(Wh), Test_Time(s), and Date_Time. If the raw exports are absent from the checkout, obtain them from the case-study provider. If you already have both Task 1 processed CSVs, place them in data/processed/ and start at Task 2.

Run the analysis

Run Task 1 first. Tasks 2–5 read its outputs and can then run independently. The comparison plot in Task 4 uses recorded forecasts.

Task 1 — Data engineering and preprocessing

python T1_data_engineering_preprocessing/preprocessing.py

Validates and cleans measurements, labels charge and discharge, and computes cycle duration, phase C-rates, coulombic efficiency, and energy throughput. Writes measurements_clean.csv and cycle_summary.csv to data/processed/, and coulombic_efficiency_trends.png to the root results/ folder.

Task 2 — Advanced degradation modeling

python T2_advanced_degredation_modeling/degredation_analysis.py

Reads both processed CSVs. Fits linear, exponential, square-root time, and LSTM models on cycles 2–80 and compares them on cycles 81–100. Computes dV/dQ for discharge cycles 2, 50, and 100. Prints held-out errors and saves the capacity forecast, dV/dQ, and model comparison plots under T2_advanced_degredation_modeling/results/.

Task 3 — Anomaly detection

python T3_anomaly_detection/anomaly_detection.py

Reads cycle_summary.csv and screens for unusual deviations using a capacity-only rule and an Isolation Forest. Writes anomaly_classification.csv and isolation_forest_cycle_classification.png to T3_anomaly_detection/results/. Open T3_anomaly_detection/anomaly_detection.ipynb for the DBSCAN/PCA analysis and run the deviation-cluster plot cell to export deviation_clusters.png to the path configured in the notebook.

Task 4 — Remaining useful life estimation

python T4_rul_estimation/capacity_fade_extrapolation.py
python T4_rul_estimation/feature_based_ML_models.py
python T4_rul_estimation/sequence_models.py
python T4_rul_estimation/plot_rul_comparison.py

Script

Inputs and outputs

capacity_fade_extrapolation.py

Uses the cycle summary to fit linear and exponential forecasts. Prints errors by early observation window and saves capacity-fade and RUL plots.

feature_based_ML_models.py

Uses both CSVs. Compares Ridge variants using capacity features, ΔQ variance, and charge duration in held-out runs. Prints EOL and RUL estimates and saves a RUL-by-cutoff plot.

sequence_models.py

Uses both CSVs. Trains LSTM and Transformer models and reports error and results across training seeds. Saves a model comparison and a RUL-by-cutoff plot.

plot_rul_comparison.py

Draws a cycle-6 comparison from recorded values in RUNS. Refresh those values after rerunning the models if you want the plot to reflect new results.

All four write figures to T4_rul_estimation/results/ by default. The extrapolation script checks the 80% rated-capacity threshold. Since the runs were already below that threshold at cycle 2, the early-forecast comparison uses 80% of measured cycle-2 capacity as the end-of-life reference.

Task 5 — Module and pack scaling

python T5_module_pack_scaling/module_pack_scaling.py

Uses cycle_summary.csv to calibrate cell fade and simulate four series-connected 8S4P modules. Compares cell variation, temperature gradients, and residual state-of-charge mismatch, then benchmarks the runtime and agreement of three model resolutions. Prints the calibration, scenario, and benchmark tables and saves pack_capacity_scenarios.png and scaling_accuracy_runtime.png in T5_module_pack_scaling/results/. Run with --demo to exercise the simulation with an illustrative fade rate if the cycle summary is unavailable.

Required visualizations

Deliverable

Default location or generator

Capacity vs. cycle count with predictive curves

Task 2 capacity plot

Coulombic efficiency trends

Task 1 efficiency plot

Deviation clusters (anomalous cycles)

Task 3 anomaly notebook, deviation-cluster export cell

Forecasted RUL distributions

Capacity fade · Feature-based ML · Sequence models

Generated figure links resolve once the corresponding PNGs are present in the checkout.