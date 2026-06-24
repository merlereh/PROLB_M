#!/usr/bin/env python3
"""
Experiment 2 – Q/R noise tuning: covariance heatmaps

For each filter, shows how the mean covariance trace changes across different
process-noise (Q) and measurement-noise (R) scale combinations.

Usage (with manifest CSV):
    python3 scripts/exp2_qr_tuning.py \
        --manifest runs/qr/manifest.csv \
        --out results/exp2_qr

    Manifest columns: csv, process_scale, measurement_scale  [, setting]

Usage (auto-scan – run folders must be named  p<Q>_m<R>):
    python3 scripts/exp2_qr_tuning.py --root runs/qr

Outputs (all in --out):
    summary_qr.csv              – all per-run metrics
    qr_covtrace_<filter>.png    – covariance heatmap per filter
"""

import argparse

import matplotlib.pyplot as plt
import re
from pathlib import Path

import pandas as pd
from eval_common import load_csv, ensure_outdir, compute_metrics, plot_heatmap

_PATTERN = re.compile(r"p(?P<p>[0-9]+(?:\.[0-9]+)?)_m(?P<m>[0-9]+(?:\.[0-9]+)?)")


def read_manifest(manifest, root) -> pd.DataFrame:
    """Load manifest CSV or auto-discover run folders matching p<Q>_m<R>."""
    if manifest:
        mf = pd.read_csv(manifest)
        if missing := {"csv", "process_scale", "measurement_scale"} - set(mf.columns):
            raise ValueError(f"Manifest missing: {sorted(missing)}")
        return mf

    rows = []
    for csv_path in Path(root).rglob("*.csv"):
        m = _PATTERN.search(str(csv_path))
        if not m:
            continue
        rows.append({
            "csv":               str(csv_path),
            "process_scale":     float(m.group("p")),
            "measurement_scale": float(m.group("m")),
        })
    if not rows:
        raise ValueError(f"No CSVs matching p<Q>_m<R> found under '{root}'")
    return pd.DataFrame(rows).sort_values(["process_scale", "measurement_scale"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--root",    help="Auto-scan root, e.g. runs/qr")
    ap.add_argument("--out",     default="results/exp2_qr")
    ap.add_argument("--ref",     default="amcl")
    ap.add_argument("--sources", nargs="+", default=["kf", "ekf", "pf"])
    ap.add_argument("--max-dt",  type=float, default=0.5)
    args = ap.parse_args()

    if not args.manifest and not args.root:
        raise SystemExit("Provide either --manifest or --root")

    out      = ensure_outdir(args.out)
    manifest = read_manifest(args.manifest, args.root)

    # ── Compute metrics for every run × filter combination ────────────────────
    rows = []
    for _, run in manifest.iterrows():
        df = load_csv(run["csv"])
        for src in args.sources:
            try:
                m, _ = compute_metrics(df, src, ref_source=args.ref, max_dt=args.max_dt)
            except ValueError as e:
                print(f"  skip {run['csv']} [{src}]: {e}")
                continue
            m["process_scale"]     = float(run["process_scale"])
            m["measurement_scale"] = float(run["measurement_scale"])
            rows.append(m)

    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("No metrics computed – check --sources and run paths.")
    table.to_csv(out / "summary_qr.csv", index=False)

    # ── One covariance heatmap per filter ─────────────────────────────────────
    # Rows = Q scale, columns = R scale, cell colour = mean covariance trace
    for src in sorted(table["source"].unique()):
        sub = table[table["source"] == src]
        if sub["mean_cov_trace"].notna().any():
            plot_heatmap(sub, "mean_cov_trace",
                         title=f"{src.upper()} – Mean Covariance Trace [m²]",
                         out_file=out / f"qr_covtrace_{src}.png")

    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()