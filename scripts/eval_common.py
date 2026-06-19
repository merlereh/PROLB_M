#!/usr/bin/env python3
"""
Common evaluation utilities for PRO Lab KF/EKF/PF experiments.

Expected trajectory CSV columns, minimum:
    time, source, x, y
Optional:
    theta, cov_xx, cov_yy, cov_xy,
    runtime_ms / update_time_ms / compute_time_ms,
    ess, resampling_triggered,
    particle_var_x, particle_var_y, particle_cov_trace

Reference source defaults to AMCL. The assignment asks for same input data,
same coordinate frame, same test trajectories and quantitative metrics, so all
scripts use nearest-timestamp alignment against one reference trajectory.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

CHI2_2D_95 = 5.991464547107979  # 95% confidence region for chi-square with 2 DoF
RUNTIME_COLS = ("runtime_ms", "update_time_ms", "compute_time_ms", "dt_runtime_ms")

FILTER_LABELS = {
    "kf": "KF",
    "ekf": "EKF",
    "pf": "PF",
    "odom": "Odometry",
    "amcl": "AMCL reference",
    "ekf_predict_only": "EKF predict-only",
    "kf_predict_only": "KF predict-only",
}


def label(src: str) -> str:
    return FILTER_LABELS.get(str(src).lower(), str(src))


def ensure_outdir(outdir: str | Path) -> Path:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(csv_file: str | Path) -> pd.DataFrame:
    csv_file = Path(csv_file)
    df = pd.read_csv(csv_file)
    required = {"time", "source", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_file} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["source"] = df["source"].astype(str).str.lower().str.strip()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def nearest_indices(ref_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ref_times, query_times)
    idx = np.clip(idx, 0, len(ref_times) - 1)
    idx_prev = np.maximum(idx - 1, 0)
    use_prev = np.abs(ref_times[idx_prev] - query_times) < np.abs(ref_times[idx] - query_times)
    idx[use_prev] = idx_prev[use_prev]
    return idx


def align_to_reference(
    df: pd.DataFrame,
    source: str,
    ref_source: str = "amcl",
    max_time_diff: float = 0.5,
) -> pd.DataFrame:
    """Return source rows with nearest reference x/y attached."""
    source = source.lower()
    ref_source = ref_source.lower()
    sub = df[df["source"] == source].sort_values("time").reset_index(drop=True)
    ref = df[df["source"] == ref_source].sort_values("time").reset_index(drop=True)

    if sub.empty:
        raise ValueError(f"No rows found for source={source!r}")
    if ref.empty:
        raise ValueError(f"No reference rows found for ref_source={ref_source!r}")

    ref_times = ref["time"].to_numpy(float)
    q_times = sub["time"].to_numpy(float)
    idx = nearest_indices(ref_times, q_times)
    dt = np.abs(ref_times[idx] - q_times)
    valid = dt <= max_time_diff

    out = sub.loc[valid].copy().reset_index(drop=True)
    out["ref_time"] = ref_times[idx][valid]
    out["time_diff_ref"] = dt[valid]
    out["x_ref"] = ref["x"].to_numpy(float)[idx][valid]
    out["y_ref"] = ref["y"].to_numpy(float)[idx][valid]
    out["error_x"] = out["x"].to_numpy(float) - out["x_ref"].to_numpy(float)
    out["error_y"] = out["y"].to_numpy(float) - out["y_ref"].to_numpy(float)
    out["error_pos"] = np.sqrt(out["error_x"] ** 2 + out["error_y"] ** 2)
    return out


def cov_trace_xy(row_or_df: pd.DataFrame | pd.Series) -> pd.Series | float:
    return row_or_df["cov_xx"] + row_or_df["cov_yy"]


def cov_det_xy(row_or_df: pd.DataFrame | pd.Series) -> pd.Series | float:
    return row_or_df["cov_xx"] * row_or_df["cov_yy"] - row_or_df["cov_xy"] ** 2


def add_covariance_consistency(aligned: pd.DataFrame) -> pd.DataFrame:
    """Add trace/determinant/Mahalanobis/2-sigma coverage if covariance exists."""
    out = aligned.copy()
    needed = {"cov_xx", "cov_yy", "cov_xy"}
    if not needed.issubset(out.columns):
        out["cov_trace_xy"] = np.nan
        out["cov_det_xy"] = np.nan
        out["mahalanobis2_xy"] = np.nan
        out["inside_2sigma"] = np.nan
        return out

    cxx = out["cov_xx"].to_numpy(float)
    cyy = out["cov_yy"].to_numpy(float)
    cxy = out["cov_xy"].to_numpy(float)
    ex = out["error_x"].to_numpy(float)
    ey = out["error_y"].to_numpy(float)
    det = cxx * cyy - cxy * cxy
    trace = cxx + cyy

    maha = np.full(len(out), np.nan)
    valid = (det > 1e-12) & np.isfinite(det) & (cxx >= 0) & (cyy >= 0)
    # inverse of [[cxx,cxy],[cxy,cyy]] times error vector
    maha[valid] = (
        cyy[valid] * ex[valid] ** 2
        - 2.0 * cxy[valid] * ex[valid] * ey[valid]
        + cxx[valid] * ey[valid] ** 2
    ) / det[valid]

    out["cov_trace_xy"] = trace
    out["cov_det_xy"] = det
    out["mahalanobis2_xy"] = maha
    inside = np.where(np.isfinite(maha), maha <= CHI2_2D_95, np.nan)
    out["inside_2sigma"] = inside
    return out


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_metrics(
    df: pd.DataFrame,
    source: str,
    ref_source: str = "amcl",
    max_time_diff: float = 0.5,
) -> tuple[pd.Series, pd.DataFrame]:
    aligned = align_to_reference(df, source, ref_source, max_time_diff)
    aligned = add_covariance_consistency(aligned)
    err = aligned["error_pos"].to_numpy(float)

    metrics = {
        "source": source.lower(),
        "n_samples": int(len(aligned)),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
        "mae_m": float(np.mean(np.abs(err))) if len(err) else np.nan,
        "max_error_m": float(np.max(err)) if len(err) else np.nan,
        "final_error_m": float(err[-1]) if len(err) else np.nan,
        "std_error_m": float(np.std(err)) if len(err) else np.nan,
        "mean_cov_trace_xy": float(np.nanmean(aligned["cov_trace_xy"])) if "cov_trace_xy" in aligned else np.nan,
        "mean_cov_det_xy": float(np.nanmean(aligned["cov_det_xy"])) if "cov_det_xy" in aligned else np.nan,
        "coverage_2sigma_pct": float(100.0 * np.nanmean(aligned["inside_2sigma"])) if "inside_2sigma" in aligned else np.nan,
    }

    runtime_col = first_existing_column(aligned, RUNTIME_COLS)
    if runtime_col:
        metrics["mean_runtime_ms"] = float(np.nanmean(aligned[runtime_col]))
        metrics["max_runtime_ms"] = float(np.nanmax(aligned[runtime_col]))
    else:
        metrics["mean_runtime_ms"] = np.nan
        metrics["max_runtime_ms"] = np.nan

    if "time" in aligned and len(aligned) > 1:
        dts = np.diff(aligned["time"].to_numpy(float))
        dts = dts[dts > 0]
        metrics["mean_period_s"] = float(np.mean(dts)) if len(dts) else np.nan
        metrics["mean_rate_hz"] = float(1.0 / np.mean(dts)) if len(dts) else np.nan
    else:
        metrics["mean_period_s"] = np.nan
        metrics["mean_rate_hz"] = np.nan

    # PF-specific optional metrics
    if "ess" in aligned.columns:
        metrics["mean_ess"] = float(np.nanmean(aligned["ess"]))
        metrics["min_ess"] = float(np.nanmin(aligned["ess"]))
    else:
        metrics["mean_ess"] = np.nan
        metrics["min_ess"] = np.nan

    resample_col = first_existing_column(aligned, ("resampling_triggered", "resampled", "did_resample"))
    if resample_col:
        metrics["resampling_count"] = int(np.nansum(aligned[resample_col].astype(float)))
    else:
        metrics["resampling_count"] = np.nan

    if "particle_cov_trace" in aligned.columns:
        metrics["mean_particle_spread"] = float(np.nanmean(aligned["particle_cov_trace"]))
    elif {"particle_var_x", "particle_var_y"}.issubset(aligned.columns):
        metrics["mean_particle_spread"] = float(np.nanmean(aligned["particle_var_x"] + aligned["particle_var_y"]))
    else:
        metrics["mean_particle_spread"] = np.nan

    return pd.Series(metrics), aligned


def save_metrics_table(metrics: list[pd.Series], out_csv: str | Path) -> pd.DataFrame:
    table = pd.DataFrame(metrics)
    table.to_csv(out_csv, index=False)
    return table


def cov_ellipse_patch(x, y, cov_xx, cov_yy, cov_xy, n_std=2.0, **kwargs):
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    if np.all(vals <= 1e-12):
        return None
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width = 2.0 * n_std * math.sqrt(vals[0])
    height = 2.0 * n_std * math.sqrt(vals[1])
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    return Ellipse((x, y), width=width, height=height, angle=angle, **kwargs)


def plot_trajectories(
    df: pd.DataFrame,
    sources: list[str],
    out_file: str | Path,
    ref_source: str = "amcl",
    ellipse_every: int = 100,
    n_std: float = 2.0,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Trajectories and uncertainty ellipses")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plot_sources = [ref_source.lower()] + [s.lower() for s in sources]
    seen = set()
    for src in plot_sources:
        if src in seen:
            continue
        seen.add(src)
        sub = df[df["source"] == src].sort_values("time").reset_index(drop=True)
        if sub.empty:
            continue
        ls = "--" if src == ref_source.lower() else "-"
        lw = 1.2 if src == ref_source.lower() else 1.5
        ax.plot(sub["x"], sub["y"], ls=ls, lw=lw, label=label(src))
        ax.scatter(sub["x"].iloc[0], sub["y"].iloc[0], s=25)

        if src != ref_source.lower() and ellipse_every > 0 and {"cov_xx", "cov_yy", "cov_xy"}.issubset(sub.columns):
            line_color = ax.lines[-1].get_color()
            for i in range(0, len(sub), ellipse_every):
                r = sub.iloc[i]
                ell = cov_ellipse_patch(
                    r["x"], r["y"], r["cov_xx"], r["cov_yy"], r["cov_xy"],
                    n_std=n_std,
                    edgecolor=line_color,
                    facecolor="none",
                    linewidth=0.7,
                    alpha=0.35,
                )
                if ell is not None:
                    ax.add_patch(ell)

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_error_time(aligned_by_source: dict[str, pd.DataFrame], out_file: str | Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Position error over time")
    ax.set_xlabel("time from start [s]")
    ax.set_ylabel("position error [m]")
    ax.grid(True, alpha=0.3)
    t0 = min(float(a["time"].min()) for a in aligned_by_source.values() if not a.empty)
    for src, a in aligned_by_source.items():
        ax.plot(a["time"] - t0, a["error_pos"], lw=1.0, label=label(src))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_cov_trace_time(aligned_by_source: dict[str, pd.DataFrame], out_file: str | Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Position covariance trace over time")
    ax.set_xlabel("time from start [s]")
    ax.set_ylabel("trace(P_xy) [m²]")
    ax.grid(True, alpha=0.3)
    t0 = min(float(a["time"].min()) for a in aligned_by_source.values() if not a.empty)
    for src, a in aligned_by_source.items():
        if "cov_trace_xy" not in a.columns or a["cov_trace_xy"].isna().all():
            continue
        ax.plot(a["time"] - t0, a["cov_trace_xy"], lw=1.0, label=label(src))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_bar(table: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, out_file: str | Path, sort=True):
    data = table.dropna(subset=[y_col]).copy()
    if sort:
        data = data.sort_values(y_col)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    labels = data[x_col].astype(str).tolist()
    vals = data[y_col].astype(float).tolist()
    bars = ax.bar(labels, vals)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_heatmap(table: pd.DataFrame, value_col: str, out_file: str | Path, title: str, fmt: str = ".3f"):
    pivot = table.pivot_table(index="process_scale", columns="measurement_scale", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Measurement noise scale")
    ax.set_ylabel("Process noise scale")
    ax.set_xticks(np.arange(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [str(i) for i in pivot.index])
    fig.colorbar(im, ax=ax, label=value_col)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
