# Evaluation scripts for PROLB_M

Put these files into your repo, for example:

```bash
mkdir -p scripts/evaluation
cp eval_common.py exp1_baseline.py exp2_qr_tuning.py exp3_pf_resampling.py generate_experiment_configs.py scripts/evaluation/
```

Install dependencies if needed:

```bash
pip install pandas numpy matplotlib pyyaml
```

## Experiment 1: Baseline KF/EKF/PF

Use one `trajectory_log.csv` with `source` values `amcl`, `kf`, `ekf`, `pf`.

```bash
python3 scripts/evaluation/exp1_baseline.py \
  --csv trajectory_log.csv \
  --out results/exp1_baseline \
  --sources kf ekf pf
```

Outputs: RMSE/MAE/max/final error, covariance trace, 2-sigma coverage, runtime if your CSV contains a runtime column.

## Experiment 2: Q/R tuning

Generate parameter configs:

```bash
python3 scripts/evaluation/generate_experiment_configs.py \
  --base config/filter_params.yaml \
  --out config/experiments
```

Run the exact same trajectory once per generated config. Save each CSV, for example:

```text
runs/qr/p0.5_m0.5/trajectory_log.csv
runs/qr/p0.5_m1.0/trajectory_log.csv
...
```

Then create or edit a manifest:

```csv
csv,process_scale,measurement_scale,setting
runs/qr/p0.5_m0.5/trajectory_log.csv,0.5,0.5,p0.5_m0.5
runs/qr/p0.5_m1.0/trajectory_log.csv,0.5,1.0,p0.5_m1.0
```

Evaluate:

```bash
python3 scripts/evaluation/exp2_qr_tuning.py \
  --manifest runs/qr/manifest.csv \
  --out results/exp2_qr \
  --sources kf ekf pf
```

## Experiment 3: PF resampling strategies / thresholds

Run the same trajectory once per PF resampling config. Save each CSV, for example:

```text
runs/pf_resampling/threshold_t0.0/trajectory_log.csv
runs/pf_resampling/threshold_t0.5/trajectory_log.csv
runs/pf_resampling/threshold_t1.0/trajectory_log.csv
runs/pf_resampling/threshold_t1.5/trajectory_log.csv
```

Manifest:

```csv
csv,method,threshold,setting
runs/pf_resampling/threshold_t0.0/trajectory_log.csv,threshold,0.0,threshold_t0.0
runs/pf_resampling/threshold_t0.5/trajectory_log.csv,threshold,0.5,threshold_t0.5
```

Evaluate:

```bash
python3 scripts/evaluation/exp3_pf_resampling.py \
  --manifest runs/pf_resampling/manifest.csv \
  --out results/exp3_pf_resampling
```

## Runtime logging

The scripts automatically plot runtime if your CSV contains one of these columns:

- `runtime_ms`
- `update_time_ms`
- `compute_time_ms`
- `dt_runtime_ms`

For PF-specific plots, add these columns if possible:

- `ess`
- `resampling_triggered` with 0/1
- `particle_var_x`, `particle_var_y` or `particle_cov_trace`

Without those columns, the script still computes RMSE/MAE/max/final error and covariance consistency from your current CSV.
