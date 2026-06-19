#!/usr/bin/env python3
"""
Experiment 1: Baseline comparison of KF, EKF and PF.

Example:
    python3 scripts/evaluation/exp1_baseline.py \
        --csv trajectory_log.csv \
        --out results/exp1_baseline \
        --sources kf ekf pf

Outputs:
    summary_baseline.csv
    baseline_trajectories.png
    baseline_error_time.png
    baseline_cov_trace.png
    baseline_rmse_bar.png
    baseline_coverage_bar.png
    baseline_runtime_bar.png  (only meaningful if runtime columns exist)
"""

from __future__ import annotations

import argparse
import pandas as pd
from eval_common import (
    compute_metrics,
    ensure_outdir,
    load_csv,
    plot_bar,
    plot_cov_trace_time,
    plot_error_time,
    plot_trajectories,
    save_metrics_table,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="trajectory_log.csv", help="Trajectory log CSV")
    ap.add_argument("--out", default="results/exp1_baseline", help="Output directory")
    ap.add_argument("--ref", default="amcl", help="Reference source, usually amcl")
    ap.add_argument("--sources", nargs="+", default=["kf", "ekf", "pf"], help="Sources to compare")
    ap.add_argument("--max-time-diff", type=float, default=0.5, help="Maximum timestamp difference to reference [s]")
    ap.add_argument("--ellipse-every", type=int, default=100, help="Draw every Nth covariance ellipse")
    args = ap.parse_args()

    out = ensure_outdir(args.out)
    df = load_csv(args.csv)

    metrics = []
    aligned = {}
    for src in args.sources:
        try:
            m, a = compute_metrics(df, src, ref_source=args.ref, max_time_diff=args.max_time_diff)
        except ValueError as exc:
            print(f"WARNING: {exc}")
            continue
        metrics.append(m)
        aligned[src] = a
        a.to_csv(out / f"aligned_{src}.csv", index=False)

    if not metrics:
        raise SystemExit("No filter data found. Check --sources and CSV source names.")

    table = save_metrics_table(metrics, out / "summary_baseline.csv")
    print(table.to_string(index=False))

    plot_trajectories(df, list(aligned.keys()), out / "baseline_trajectories.png", ref_source=args.ref, ellipse_every=args.ellipse_every)
    plot_error_time(aligned, out / "baseline_error_time.png")
    plot_cov_trace_time(aligned, out / "baseline_cov_trace.png")

    plot_data = table.copy()
    plot_data["filter"] = plot_data["source"].str.upper()
    plot_bar(plot_data, "filter", "rmse_m", "Baseline RMSE", "RMSE [m]", out / "baseline_rmse_bar.png", sort=False)
    plot_bar(plot_data, "filter", "coverage_2sigma_pct", "2-sigma covariance coverage", "coverage [%]", out / "baseline_coverage_bar.png", sort=False)

    if "mean_runtime_ms" in table.columns and table["mean_runtime_ms"].notna().any():
        plot_bar(plot_data, "filter", "mean_runtime_ms", "Mean runtime per update", "runtime [ms]", out / "baseline_runtime_bar.png", sort=False)
    else:
        print("No runtime column found. Add runtime_ms/update_time_ms/compute_time_ms to CSV for runtime plots.")

    print(f"\nSaved plots and CSV files to: {out}")


if __name__ == "__main__":
    main()
