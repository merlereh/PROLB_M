#!/usr/bin/env python3
"""
Experiment 3: Particle Filter resampling strategy / threshold comparison.

Recommended manifest format:

    csv,method,threshold,setting
    runs/pf_resampling/multinomial_t0.0/trajectory_log.csv,multinomial,0.0,multinomial_t0.0
    runs/pf_resampling/threshold_t0.5/trajectory_log.csv,threshold,0.5,threshold_t0.5
    ...

Example:
    python3 scripts/evaluation/exp3_pf_resampling.py \
        --manifest runs/pf_resampling/manifest.csv \
        --out results/exp3_pf_resampling

Alternative auto-scan:
    python3 scripts/evaluation/exp3_pf_resampling.py --root runs/pf_resampling

Auto-scan parses filenames/folders containing t<threshold> and uses the parent
folder prefix as method name when possible.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eval_common import compute_metrics, ensure_outdir, load_csv, plot_bar, plot_error_time

THRESHOLD_PATTERN = re.compile(r"(?:^|[_-])t(?P<t>[0-9]+(?:\.[0-9]+)?)")


def read_manifest(manifest: str | Path | None, root: str | Path | None) -> pd.DataFrame:
    if manifest:
        mf = pd.read_csv(manifest)
        needed = {"csv", "method", "threshold"}
        missing = needed - set(mf.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        if "setting" not in mf.columns:
            mf["setting"] = mf["method"].astype(str) + "_t" + mf["threshold"].astype(str)
        return mf

    if not root:
        raise ValueError("Use either --manifest or --root")
    rows = []
    for csv_path in Path(root).rglob("*.csv"):
        text = str(csv_path)
        match = THRESHOLD_PATTERN.search(text)
        if not match:
            continue
        threshold = float(match.group("t"))
        parent = csv_path.parent.name
        method = parent.split("_t")[0].split("-t")[0] or "pf"
        rows.append({
            "csv": str(csv_path),
            "method": method,
            "threshold": threshold,
            "setting": f"{method}_t{threshold}",
        })
    if not rows:
        raise ValueError("No CSV files with threshold pattern t<threshold> found under --root")
    return pd.DataFrame(rows).sort_values(["method", "threshold"])


def plot_metric_vs_threshold(table: pd.DataFrame, y_col: str, ylabel: str, title: str, out_file: Path):
    data = table.dropna(subset=[y_col]).copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel("resampling threshold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    for method, sub in data.groupby("method"):
        sub = sub.sort_values("threshold")
        ax.plot(sub["threshold"], sub[y_col], marker="o", label=str(method))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_optional_time_series(aligned_runs: dict[str, pd.DataFrame], col: str, ylabel: str, title: str, out_file: Path):
    has_any = any(col in a.columns and a[col].notna().any() for a in aligned_runs.values())
    if not has_any:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title(title)
    ax.set_xlabel("time from start [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    t0 = min(float(a["time"].min()) for a in aligned_runs.values() if not a.empty)
    for setting, a in aligned_runs.items():
        if col in a.columns and a[col].notna().any():
            ax.plot(a["time"] - t0, a[col], lw=1.0, label=setting)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="CSV manifest with columns csv,method,threshold")
    ap.add_argument("--root", help="Root directory for auto-scan")
    ap.add_argument("--out", default="results/exp3_pf_resampling", help="Output directory")
    ap.add_argument("--ref", default="amcl")
    ap.add_argument("--source", default="pf", help="PF source name in trajectory CSV")
    ap.add_argument("--max-time-diff", type=float, default=0.5)
    args = ap.parse_args()

    out = ensure_outdir(args.out)
    manifest = read_manifest(args.manifest, args.root)
    manifest.to_csv(out / "used_manifest_pf_resampling.csv", index=False)

    rows = []
    aligned_runs = {}
    for _, run in manifest.iterrows():
        df = load_csv(run["csv"])
        try:
            m, a = compute_metrics(df, args.source, ref_source=args.ref, max_time_diff=args.max_time_diff)
        except ValueError as exc:
            print(f"WARNING {run['csv']}: {exc}")
            continue
        setting = str(run["setting"])
        m["csv"] = run["csv"]
        m["method"] = run["method"]
        m["threshold"] = float(run["threshold"])
        m["setting"] = setting
        rows.append(m)
        aligned_runs[setting] = a
        a.to_csv(out / f"aligned_{setting}.csv", index=False)

    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("No PF metrics computed. Check source name and manifest paths.")
    table.to_csv(out / "summary_pf_resampling.csv", index=False)
    print(table.to_string(index=False))

    # Main PF comparison plots.
    plot_metric_vs_threshold(table, "rmse_m", "RMSE [m]", "PF RMSE vs resampling threshold", out / "pf_rmse_vs_threshold.png")
    plot_metric_vs_threshold(table, "max_error_m", "max error [m]", "PF max error vs resampling threshold", out / "pf_max_error_vs_threshold.png")
    plot_metric_vs_threshold(table, "mean_ess", "mean ESS", "PF mean effective sample size", out / "pf_mean_ess_vs_threshold.png")
    plot_metric_vs_threshold(table, "min_ess", "min ESS", "PF minimum effective sample size", out / "pf_min_ess_vs_threshold.png")
    plot_metric_vs_threshold(table, "resampling_count", "count", "PF number of resampling steps", out / "pf_resampling_count_vs_threshold.png")
    plot_metric_vs_threshold(table, "mean_particle_spread", "mean particle spread", "PF particle diversity", out / "pf_particle_spread_vs_threshold.png")
    plot_metric_vs_threshold(table, "mean_runtime_ms", "runtime [ms]", "PF runtime vs threshold", out / "pf_runtime_vs_threshold.png")

    plot_error_time(aligned_runs, out / "pf_error_time_all_settings.png")
    plot_optional_time_series(aligned_runs, "ess", "ESS", "PF effective sample size over time", out / "pf_ess_time_all_settings.png")

    # Particle spread time series: create unified column if possible.
    for setting, a in aligned_runs.items():
        if "particle_cov_trace" in a.columns:
            a["particle_spread"] = a["particle_cov_trace"]
        elif {"particle_var_x", "particle_var_y"}.issubset(a.columns):
            a["particle_spread"] = a["particle_var_x"] + a["particle_var_y"]
    plot_optional_time_series(aligned_runs, "particle_spread", "particle spread", "PF particle diversity over time", out / "pf_particle_spread_time_all_settings.png")

    if table["mean_runtime_ms"].notna().any():
        plot_bar(table, "setting", "mean_runtime_ms", "PF mean runtime per update", "runtime [ms]", out / "pf_runtime_bar.png", sort=False)

    print(f"\nSaved PF resampling plots and CSV files to: {out}")


if __name__ == "__main__":
    main()
