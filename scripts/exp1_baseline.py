#!/usr/bin/env python3
"""
Experiment 1 – Baseline: KF vs EKF vs PF

Usage:
    python3 scripts/exp1_baseline.py \
        --csv trajectory_log.csv \
        --out results/exp1_baseline

Outputs (all in --out):
    summary_baseline.csv        – metrics per filter
    baseline_trajectories.png   – paths + 2σ uncertainty ellipses
    baseline_rmse.png           – RMSE bar chart
    baseline_runtime.png        – mean update time bar chart  (if logged)
    baseline_cov_trace.png      – covariance trace over time
"""

import argparse

import pandas as pd
from eval_common import (
    load_csv, ensure_outdir, compute_metrics,
    plot_trajectories, plot_cov_trace_time, plot_bar,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",           default="trajectory_log.csv")
    ap.add_argument("--out",           default="results/exp1_baseline")
    ap.add_argument("--ref",           default="amcl",
                    help="Reference source used as ground-truth proxy")
    ap.add_argument("--sources",       nargs="+", default=["kf", "ekf", "pf"])
    ap.add_argument("--max-dt",        type=float, default=0.5,
                    help="Max allowed time gap when matching filter to reference [s]")
    ap.add_argument("--ellipse-every", type=int, default=30,
                    help="Draw a 2σ covariance ellipse every N poses (0 = off)")
    args = ap.parse_args()

    out = ensure_outdir(args.out)
    df  = load_csv(args.csv)

    # ── Compute metrics for each filter ───────────────────────────────────────
    all_metrics, aligned = [], {}
    for src in args.sources:
        try:
            m, a = compute_metrics(df, src, ref_source=args.ref, max_dt=args.max_dt)
        except ValueError as e:
            print(f"  skip {src}: {e}")
            continue
        all_metrics.append(m)
        aligned[src] = a

    if not all_metrics:
        raise SystemExit("No filter data found – check --sources and the 'source' column in the CSV.")

    table = pd.DataFrame(all_metrics)
    table.to_csv(out / "summary_baseline.csv", index=False)
    print(table[["source", "rmse_m", "max_error_m", "mean_cov_trace", "mean_runtime_ms"]].to_string(index=False))

    # Use uppercase filter names on the x-axis of bar charts
    plot_table = table.copy()
    plot_table["Filter"] = plot_table["source"].str.upper()

    # ── Plots ─────────────────────────────────────────────────────────────────

    # 1. Trajectories with 2σ covariance ellipses
    plot_trajectories(df, list(aligned.keys()), out / "baseline_trajectories.png",
                      ref_source=args.ref, ellipse_every=args.ellipse_every)

    # 2. RMSE bar chart
    plot_bar(plot_table, "Filter", "rmse_m",
             "RMSE", "RMSE [m]",
             out / "baseline_rmse.png", sort=False)

    # 3. Runtime bar chart (skipped with a note if no node logged update times)
    if table["mean_runtime_ms"].notna().any():
        plot_bar(plot_table, "Filter", "mean_runtime_ms",
                 "Update Runtime", "mean runtime [ms]",
                 out / "baseline_runtime.png", sort=False)
    else:
        print("  Note: no runtime data found – add runtime_ms / update_time_ms to the CSV.")

    # 4. Covariance trace over time
    plot_cov_trace_time(aligned, out / "baseline_cov_trace.png")

    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()