#!/usr/bin/env python3
"""
Experiment 2: Systematic Q/R tuning evaluation.

This script expects one CSV per run. Recommended manifest format:

    csv,process_scale,measurement_scale,setting
    runs/qr/p0.5_m0.5/trajectory_log.csv,0.5,0.5,p0.5_m0.5
    runs/qr/p0.5_m1.0/trajectory_log.csv,0.5,1.0,p0.5_m1.0
    ...

Here "process_scale" scales your implementation parameters r_* and
"measurement_scale" scales q_odom_* and q_lm_*.

Example:
    python3 scripts/evaluation/exp2_qr_tuning.py \
        --manifest runs/qr/manifest.csv \
        --out results/exp2_qr \
        --sources kf ekf pf

Alternative auto-scan:
    python3 scripts/evaluation/exp2_qr_tuning.py --root runs/qr

Auto-scan parses folder/file names containing p<scale>_m<scale>, for example:
    runs/qr/p0.5_m2.0/trajectory_log.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from eval_common import compute_metrics, ensure_outdir, load_csv, plot_bar, plot_heatmap

PATTERN = re.compile(r"p(?P<p>[0-9]+(?:\.[0-9]+)?)_m(?P<m>[0-9]+(?:\.[0-9]+)?)")


def read_manifest(manifest: str | Path | None, root: str | Path | None) -> pd.DataFrame:
    if manifest:
        mf = pd.read_csv(manifest)
        needed = {"csv", "process_scale", "measurement_scale"}
        missing = needed - set(mf.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        if "setting" not in mf.columns:
            mf["setting"] = [f"p{p}_m{m}" for p, m in zip(mf["process_scale"], mf["measurement_scale"])]
        return mf

    if not root:
        raise ValueError("Use either --manifest or --root")
    rows = []
    for csv_path in Path(root).rglob("*.csv"):
        text = str(csv_path)
        match = PATTERN.search(text)
        if not match:
            continue
        rows.append({
            "csv": str(csv_path),
            "process_scale": float(match.group("p")),
            "measurement_scale": float(match.group("m")),
            "setting": f"p{match.group('p')}_m{match.group('m')}",
        })
    if not rows:
        raise ValueError("No CSV files matching pattern p<scale>_m<scale> found under --root")
    return pd.DataFrame(rows).sort_values(["process_scale", "measurement_scale"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="CSV manifest with columns csv,process_scale,measurement_scale")
    ap.add_argument("--root", help="Root directory for auto-scan, e.g. runs/qr")
    ap.add_argument("--out", default="results/exp2_qr", help="Output directory")
    ap.add_argument("--ref", default="amcl", help="Reference source")
    ap.add_argument("--sources", nargs="+", default=["kf", "ekf", "pf"], help="Sources to evaluate")
    ap.add_argument("--max-time-diff", type=float, default=0.5)
    args = ap.parse_args()

    out = ensure_outdir(args.out)
    manifest = read_manifest(args.manifest, args.root)
    manifest.to_csv(out / "used_manifest_qr.csv", index=False)

    all_metrics = []
    for _, run in manifest.iterrows():
        df = load_csv(run["csv"])
        for src in args.sources:
            try:
                m, _ = compute_metrics(df, src, ref_source=args.ref, max_time_diff=args.max_time_diff)
            except ValueError as exc:
                print(f"WARNING {run['csv']} {src}: {exc}")
                continue
            m["csv"] = run["csv"]
            m["setting"] = run["setting"]
            m["process_scale"] = float(run["process_scale"])
            m["measurement_scale"] = float(run["measurement_scale"])
            all_metrics.append(m)

    table = pd.DataFrame(all_metrics)
    if table.empty:
        raise SystemExit("No metrics computed. Check source names and manifest paths.")
    table.to_csv(out / "summary_qr_all.csv", index=False)
    print(table.to_string(index=False))

    # Heatmaps per filter/source.
    for src in sorted(table["source"].unique()):
        sub = table[table["source"] == src]
        plot_heatmap(sub, "rmse_m", out / f"qr_heatmap_rmse_{src}.png", f"{src.upper()} RMSE for Q/R tuning", fmt=".3f")
        if sub["coverage_2sigma_pct"].notna().any():
            plot_heatmap(sub, "coverage_2sigma_pct", out / f"qr_heatmap_coverage_{src}.png", f"{src.upper()} 2-sigma coverage", fmt=".1f")
        if sub["mean_cov_trace_xy"].notna().any():
            plot_heatmap(sub, "mean_cov_trace_xy", out / f"qr_heatmap_covtrace_{src}.png", f"{src.upper()} mean covariance trace", fmt=".3f")
        if sub["mean_runtime_ms"].notna().any():
            plot_heatmap(sub, "mean_runtime_ms", out / f"qr_heatmap_runtime_{src}.png", f"{src.upper()} mean runtime", fmt=".3f")

    # One compact bar chart for best/worst setting by RMSE per source.
    best_rows = table.loc[table.groupby("source")["rmse_m"].idxmin()].copy()
    best_rows["best_setting"] = best_rows["source"].str.upper() + " " + best_rows["setting"].astype(str)
    plot_bar(best_rows, "best_setting", "rmse_m", "Best Q/R setting per filter", "RMSE [m]", out / "qr_best_settings_rmse.png", sort=True)

    print(f"\nSaved Q/R tuning plots and CSV files to: {out}")


if __name__ == "__main__":
    main()
