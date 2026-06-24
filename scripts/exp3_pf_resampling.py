#!/usr/bin/env python3
"""
Experiment 3 – PF resampling threshold comparison

Evaluates how the resampling threshold affects PF accuracy and particle diversity.

Usage (with manifest CSV):
    python3 scripts/exp3_pf_resampling.py \
        --manifest runs/pf_resampling/manifest.csv \
        --out results/exp3_pf_resampling

    Manifest columns: csv, method, threshold  [, setting]

Usage (auto-scan – folder names must contain  t<threshold>):
    python3 scripts/exp3_pf_resampling.py --root runs/pf_resampling

Outputs (all in --out):
    summary_pf.csv                  – per-run metrics
    pf_ess_over_time.png            – ESS time series for every threshold setting
    pf_ess_vs_threshold.png         – mean ESS per threshold (summary)
    pf_max_error_vs_threshold.png   – worst-case position error vs threshold
    pf_rmse_vs_threshold.png        – RMSE vs threshold
"""

import argparse

import matplotlib.pyplot as plt
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from eval_common import load_csv, ensure_outdir, compute_metrics

_PATTERN = re.compile(r"(?:^|[_-])t(?P<t>[0-9]+(?:\.[0-9]+)?)")


def read_manifest(manifest, root) -> pd.DataFrame:
    """Load manifest CSV or auto-discover run folders matching *_t<threshold>."""
    if manifest:
        mf = pd.read_csv(manifest)
        if missing := {"csv", "method", "threshold"} - set(mf.columns):
            raise ValueError(f"Manifest missing: {sorted(missing)}")
        if "setting" not in mf.columns:
            mf["setting"] = mf["method"] + "_t" + mf["threshold"].astype(str)
        return mf

    rows = []
    for csv_path in Path(root).rglob("*.csv"):
        m = _PATTERN.search(str(csv_path))
        if not m:
            continue
        method = csv_path.parent.name.split("_t")[0] or "pf"
        t = float(m.group("t"))
        rows.append({
            "csv":       str(csv_path),
            "method":    method,
            "threshold": t,
            "setting":   f"{method}_t{t}",
        })
    if not rows:
        raise ValueError(f"No CSVs with t<threshold> pattern found under '{root}'")
    return pd.DataFrame(rows).sort_values(["method", "threshold"])


_FS = dict(title=25, label=20, tick=20, legend=20)

def _plot_ess_time(aligned_runs: dict, out_file):
    """ESS over time – one line per threshold setting."""
    has_ess = any("ess" in a.columns and a["ess"].notna().any()
                  for a in aligned_runs.values())
    if not has_ess:
        print(f"  skip {out_file.name}: no ESS column in data")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Effective sample size over time",  fontsize=_FS["title"])
    ax.set_xlabel("time from start [s]",             fontsize=_FS["label"])
    ax.set_ylabel("ESS",                             fontsize=_FS["label"])
    ax.tick_params(labelsize=_FS["tick"])
    ax.grid(alpha=0.3)
    t0 = min(float(a["time"].min()) for a in aligned_runs.values() if not a.empty)
    for setting, a in aligned_runs.items():
        if "ess" in a.columns and a["ess"].notna().any():
            ax.plot(a["time"] - t0, a["ess"], lw=2.5, label=setting)
    ax.legend(fontsize=_FS["legend"])
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def _line_plot(table: pd.DataFrame, y_col: str, ylabel: str, title: str, out_file):
    """
    Line plot of y_col vs resampling threshold, one line per resampling method.
    Silently skipped if no valid data is available for y_col.
    """
    data = table.dropna(subset=[y_col])
    if data.empty:
        print(f"  skip {out_file.name}: no data for '{y_col}'")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title,                              fontsize=_FS["title"])
    ax.set_xlabel("Resampling Threshold",              fontsize=_FS["label"])
    ax.set_ylabel(ylabel,                            fontsize=_FS["label"])
    ax.tick_params(labelsize=_FS["tick"])
    ax.grid(alpha=0.3)
    for method, sub in data.groupby("method"):
        sub = sub.sort_values("threshold")
        ax.plot(sub["threshold"], sub[y_col], marker="o", lw=2.5, ms=8, label=str(method))
    ax.legend(fontsize=_FS["legend"])
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--root",   help="Auto-scan root, e.g. runs/pf_resampling")
    ap.add_argument("--out",    default="results/exp3_pf_resampling")
    ap.add_argument("--ref",    default="amcl")
    ap.add_argument("--source", default="pf", help="PF source name in the trajectory CSV")
    ap.add_argument("--max-dt", type=float, default=0.5)
    args = ap.parse_args()

    if not args.manifest and not args.root:
        raise SystemExit("Provide either --manifest or --root")

    out      = ensure_outdir(args.out)
    manifest = read_manifest(args.manifest, args.root)

    # Compute metrics for each threshold run; keep aligned dfs for time series
    rows, aligned_runs = [], {}
    for _, run in manifest.iterrows():
        df = load_csv(run["csv"])
        try:
            m, a = compute_metrics(df, args.source, ref_source=args.ref, max_dt=args.max_dt)
        except ValueError as e:
            print(f"  skip {run['csv']}: {e}")
            continue
        setting = str(run["setting"])
        m["method"]    = run["method"]
        m["threshold"] = float(run["threshold"])
        m["setting"]   = setting
        rows.append(m)
        aligned_runs[setting] = a

    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("No PF metrics computed – check --source and manifest.")
    table.to_csv(out / "summary_pf.csv", index=False)
    print_cols = ["setting", "rmse_m", "max_error_m"] + (["mean_ess"] if "mean_ess" in table.columns else [])
    print(table[print_cols].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────

    # 1. ESS over time – full time series per threshold setting
    _plot_ess_time(aligned_runs, out / "pf_ess_over_time.png")

    # 2. Mean ESS vs threshold – summary bar
    _line_plot(table, "mean_ess",    "mean ESS",
               "Mean ESS vs resampling threshold",
               out / "pf_ess_vs_threshold.png")

    # 3. Max position error vs threshold – worst-case localisation failure
    _line_plot(table, "max_error_m", "max position error [m]",
               "Worst-case error vs resampling threshold",
               out / "pf_max_error_vs_threshold.png")

    # 4. RMSE vs threshold – overall accuracy
    _line_plot(table, "rmse_m",      "RMSE [m]",
               "RMSE vs resampling threshold",
               out / "pf_rmse_vs_threshold.png")

    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
