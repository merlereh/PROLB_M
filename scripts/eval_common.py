"""
eval_common.py  –  shared utilities for KF / EKF / PF experiment scripts

Expected CSV columns (minimum):  time, source, x, y
Optional:  cov_xx, cov_yy, cov_xy,
           runtime_ms / update_time_ms / compute_time_ms,
           ess
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ── Constants ──────────────────────────────────────────────────────────────────

# Per-filter colours – same colour in every plot so KF is always orange, etc.
FILTER_COLORS = {
    "amcl": "#1f77b4",   # blue   – reference
    "kf":   "#ff7f0e",   # orange
    "ekf":  "#2ca02c",   # green
    "pf":   "#d62728",   # red
}

FILTER_LABELS = {
    "amcl": "AMCL (reference)",
    "kf":   "KF",
    "ekf":  "EKF",
    "pf":   "PF",
}

# Must match LANDMARK_X / LANDMARK_Y in the C++ filter nodes
LANDMARKS = [(1.8, 0.0)]

# Possible column names for per-update compute time (try all of them)
RUNTIME_COLS = ("runtime_ms", "update_time_ms", "compute_time_ms", "dt_runtime_ms")

# 95 % confidence threshold for 2D chi-square (used for 2σ coverage check)
_CHI2_95 = 5.991


def label(src: str) -> str:
    """Human-readable display name for a filter source ('ekf' → 'EKF')."""
    return FILTER_LABELS.get(src.lower(), src)


def ensure_outdir(path) -> Path:
    """Create output directory (including parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Data loading ───────────────────────────────────────────────────────────────

def load_csv(csv_file) -> pd.DataFrame:
    """
    Load a trajectory log CSV.
    Required columns: time, source, x, y.
    'source' is normalised to lowercase so 'KF' and 'kf' are treated the same.
    """
    df = pd.read_csv(csv_file)
    missing = {"time", "source", "x", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_file}: missing columns {sorted(missing)}")
    df["source"] = df["source"].str.lower().str.strip()
    return df.sort_values("time").reset_index(drop=True)


def _nearest_indices(ref_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """For each query timestamp return the index of the nearest reference timestamp."""
    idx  = np.clip(np.searchsorted(ref_times, query_times), 0, len(ref_times) - 1)
    prev = np.maximum(idx - 1, 0)
    closer_to_prev = np.abs(ref_times[prev] - query_times) < np.abs(ref_times[idx] - query_times)
    idx[closer_to_prev] = prev[closer_to_prev]
    return idx


def align_to_reference(df: pd.DataFrame, source: str,
                       ref_source: str = "amcl", max_dt: float = 0.5) -> pd.DataFrame:
    """
    Match each filter pose to the nearest reference (AMCL) pose in time.

    Adds columns: x_ref, y_ref, error_x, error_y, error_pos.
    Rows where the nearest reference is more than max_dt seconds away are dropped.
    """
    src = df[df["source"] == source.lower()].sort_values("time").reset_index(drop=True)
    ref = df[df["source"] == ref_source.lower()].sort_values("time").reset_index(drop=True)

    if src.empty:
        raise ValueError(f"No data for source='{source}'")
    if ref.empty:
        raise ValueError(f"No data for ref_source='{ref_source}'")

    ref_t = ref["time"].to_numpy(float)
    q_t   = src["time"].to_numpy(float)
    idx   = _nearest_indices(ref_t, q_t)
    dt    = np.abs(ref_t[idx] - q_t)
    mask  = dt <= max_dt

    out = src[mask].copy().reset_index(drop=True)
    out["x_ref"]     = ref["x"].to_numpy()[idx][mask]
    out["y_ref"]     = ref["y"].to_numpy()[idx][mask]
    out["error_x"]   = out["x"] - out["x_ref"]
    out["error_y"]   = out["y"] - out["y_ref"]
    out["error_pos"] = np.sqrt(out["error_x"]**2 + out["error_y"]**2)
    return out


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, source: str,
                    ref_source: str = "amcl", max_dt: float = 0.5) -> tuple[dict, pd.DataFrame]:
    """
    Compute standard localisation metrics for one filter.
    Returns (metrics_dict, aligned_dataframe).

    Metrics always present:
        source, rmse_m, max_error_m, mae_m, n_samples,
        mean_cov_trace, mean_runtime_ms
    Metrics added when data is available:
        coverage_2sigma_pct  – fraction of errors inside the 2σ ellipse
        mean_ess, min_ess    – PF effective sample size
    """
    a   = align_to_reference(df, source, ref_source, max_dt)
    err = a["error_pos"].to_numpy()

    m: dict = {
        "source":      source.lower(),
        "rmse_m":      float(np.sqrt(np.mean(err**2))),
        "max_error_m": float(np.max(err)),
        "mae_m":       float(np.mean(np.abs(err))),
        "n_samples":   int(len(a)),
    }

    # Covariance trace = cov_xx + cov_yy  (total xy position uncertainty)
    if {"cov_xx", "cov_yy"}.issubset(a.columns):
        a["cov_trace"] = a["cov_xx"] + a["cov_yy"]
        m["mean_cov_trace"] = float(np.nanmean(a["cov_trace"]))

        # 2σ coverage: fraction of timesteps where the error lies inside the
        # 95% confidence ellipse of the filter's own covariance estimate
        if "cov_xy" in a.columns:
            cxx, cyy, cxy = a["cov_xx"].to_numpy(), a["cov_yy"].to_numpy(), a["cov_xy"].to_numpy()
            ex,  ey       = a["error_x"].to_numpy(), a["error_y"].to_numpy()
            det  = cxx * cyy - cxy**2
            maha = np.where(det > 1e-12,
                            (cyy*ex**2 - 2*cxy*ex*ey + cxx*ey**2) / np.maximum(det, 1e-30),
                            np.nan)
            m["coverage_2sigma_pct"] = float(100.0 * np.nanmean(maha <= _CHI2_95))
    else:
        a["cov_trace"]      = np.nan
        m["mean_cov_trace"] = np.nan

    # Runtime per update (try several possible column names from different nodes)
    rt_col = next((c for c in RUNTIME_COLS if c in a.columns), None)
    m["mean_runtime_ms"] = float(np.nanmean(a[rt_col])) if rt_col else np.nan

    # PF-specific: effective sample size (ESS = N_eff / N)
    if "ess" in a.columns:
        m["mean_ess"] = float(np.nanmean(a["ess"]))
        m["min_ess"]  = float(np.nanmin(a["ess"]))

    return m, a


# ── Plotting ───────────────────────────────────────────────────────────────────

def _cov_ellipse(x, y, cxx, cyy, cxy, n_std=2.0, **kwargs):
    """Return a matplotlib Ellipse patch for a 2x2 position covariance, or None if degenerate."""
    cov = np.array([[cxx, cxy], [cxy, cyy]], float)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    if np.all(vals < 1e-12):
        return None
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    return Ellipse((x, y),
                   width=2 * n_std * math.sqrt(vals[0]),
                   height=2 * n_std * math.sqrt(vals[1]),
                   angle=angle, **kwargs)


# Font sizes used in all plot functions
_FS = dict(title=25, label=20, tick=20, legend=20, annot=20)


def plot_trajectories(df: pd.DataFrame, sources: list[str], out_file,
                      ref_source: str = "amcl", ellipse_every: int = 30):
    """
    x/y trajectory plot for the AMCL reference (dashed) and all filter sources (solid).
    2-sigma covariance ellipses are drawn every ellipse_every poses.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Trajectories with 2-sigma uncertainty ellipses", fontsize=_FS["title"])
    ax.set_xlabel("x [m]",  fontsize=_FS["label"])
    ax.set_ylabel("y [m]",  fontsize=_FS["label"])
    ax.tick_params(labelsize=_FS["tick"])
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Reference trajectory – dashed, drawn first so it sits behind filter lines
    ref = df[df["source"] == ref_source.lower()].sort_values("time")
    ax.plot(ref["x"], ref["y"], "--", lw=2.0,
            color=FILTER_COLORS.get(ref_source.lower(), "#1f77b4"),
            label=label(ref_source))

    for src in sources:
        sub = df[df["source"] == src.lower()].sort_values("time").reset_index(drop=True)
        if sub.empty:
            continue
        color = FILTER_COLORS.get(src.lower(), "#888")
        ax.plot(sub["x"], sub["y"], lw=2.5, color=color, label=label(src))
        ax.scatter(sub["x"].iloc[0], sub["y"].iloc[0], s=25, color=color, zorder=5)

        if ellipse_every > 0 and {"cov_xx", "cov_yy", "cov_xy"}.issubset(sub.columns):
            for i in range(0, len(sub), ellipse_every * 2):
                r   = sub.iloc[i]
                ell = _cov_ellipse(r["x"], r["y"], r["cov_xx"], r["cov_yy"], r["cov_xy"],
                                   edgecolor=color, facecolor="none", lw=1.8, alpha=0.6)
                if ell:
                    ax.add_patch(ell)

    for i, (lx, ly) in enumerate(LANDMARKS):
        ax.scatter([lx], [ly], marker="*", s=350, color="gold", edgecolor="black",
                   lw=1.2, zorder=6, label="Landmark" if i == 0 else None)

    ax.legend(fontsize=_FS["legend"])
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_cov_trace_time(aligned_by_source: dict[str, pd.DataFrame], out_file):
    """
    Line plot of covariance trace (cov_xx + cov_yy) over time.
    Filters without covariance data are silently skipped.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Position covariance trace over time",        fontsize=_FS["title"])
    ax.set_xlabel("time from start [s]",                       fontsize=_FS["label"])
    ax.set_ylabel("Position Covariance [m²]",                   fontsize=_FS["label"])
    ax.tick_params(labelsize=_FS["tick"])
    ax.grid(alpha=0.3)

    t0 = min(float(a["time"].min()) for a in aligned_by_source.values() if not a.empty)
    for src, a in aligned_by_source.items():
        if "cov_trace" not in a.columns or a["cov_trace"].isna().all():
            continue
        ax.plot(a["time"] - t0, a["cov_trace"],
                lw=2.5, color=FILTER_COLORS.get(src, "#888"), label=label(src))
    ax.legend(fontsize=_FS["legend"])
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_bar(data: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str,
             out_file, sort: bool = True):
    """
    Bar chart with value labels on top of each bar.
    Bars are coloured by FILTER_COLORS when x_col contains filter names.
    """
    d = data.dropna(subset=[y_col]).copy()
    if sort:
        d = d.sort_values(y_col)

    labels = d[x_col].astype(str).tolist()
    vals   = d[y_col].astype(float).tolist()
    colors = [FILTER_COLORS.get(l.lower(), "#7f7f7f") for l in labels]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), 5))
    ax.set_title(title,   fontsize=18)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(labelsize=16)
    ax.grid(axis="y", alpha=0.3)
    bars = ax.bar(labels, vals, color=colors)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3g}", ha="center", va="bottom", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_heatmap(table: pd.DataFrame, value_col: str, title: str, out_file, fmt: str = ".3f"):
    """
    Heatmap of value_col over a process_scale x measurement_scale grid.
    Rows = Q multiplier, columns = R multiplier, cell colour = metric value.
    """
    pivot = (table
             .pivot_table(index="process_scale", columns="measurement_scale",
                          values=value_col, aggfunc="mean")
             .sort_index()
             .sort_index(axis=1))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", alpha=0.65)
    ax.set_title(title,                                    fontsize=_FS["title"])
    ax.set_xlabel("Measurement noise scale", fontsize=_FS["label"])
    ax.set_ylabel("Process noise scale",     fontsize=_FS["label"])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=_FS["tick"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(r) for r in pivot.index], fontsize=_FS["tick"])
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=_FS["tick"])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        fontsize=_FS["annot"], fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                                  alpha=0.75, edgecolor="none"))
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
