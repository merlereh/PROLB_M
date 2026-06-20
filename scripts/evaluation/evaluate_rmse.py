"""
python3 scripts/evaluate_rmse.py

Berechnet RMSE für KF, EKF, PF und Odom gegen AMCL als Ground Truth.
Plottet auch den Fehler über die Zeit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "trajectory_log.csv"

# ---------------------------------------------------------------------------

def find_nearest(ref_times, ref_x, ref_y, query_times):
    """
    Für jeden query_time: finde den nächsten ref_time und gib dessen x/y zurück.
    Beide müssen numpy arrays sein, sortiert nach Zeit.
    """
    idx = np.searchsorted(ref_times, query_times)
    idx = np.clip(idx, 0, len(ref_times) - 1)

    # Vergleiche auch den vorherigen Index — nimm den näheren
    idx_prev = np.maximum(idx - 1, 0)
    diff_curr = np.abs(ref_times[idx]      - query_times)
    diff_prev = np.abs(ref_times[idx_prev] - query_times)
    better = diff_prev < diff_curr
    idx[better] = idx_prev[better]

    return ref_x[idx], ref_y[idx]


def compute_rmse(df, source, ref_times, ref_x, ref_y, max_time_diff=0.5):
    """
    Berechnet RMSE für eine Quelle gegen AMCL.
    Wirft Punkte weg bei denen der nächste AMCL-Wert > max_time_diff Sekunden entfernt ist.
    """
    sub = df[df["source"] == source].reset_index(drop=True)
    if sub.empty:
        return None, None, None

    times  = sub["time"].values
    x_vals = sub["x"].values
    y_vals = sub["y"].values

    x_ref, y_ref = find_nearest(ref_times, ref_x, ref_y, times)

    # Filter out points too far in time from any AMCL measurement
    idx_nn = np.searchsorted(ref_times, times)
    idx_nn = np.clip(idx_nn, 0, len(ref_times) - 1)
    time_diffs = np.abs(ref_times[idx_nn] - times)
    valid = time_diffs < max_time_diff

    if valid.sum() == 0:
        return None, None, None

    errors = np.sqrt((x_vals[valid] - x_ref[valid])**2 +
                     (y_vals[valid] - y_ref[valid])**2)
    rmse = np.sqrt(np.mean(errors**2))

    return rmse, times[valid], errors


def main():
    df = pd.read_csv(CSV_FILE)

    # AMCL als Ground Truth
    amcl = df[df["source"] == "amcl"].sort_values("time").reset_index(drop=True)
    if amcl.empty:
        print("Keine AMCL-Daten gefunden! Ist /amcl_pose in der CSV?")
        return

    ref_times = amcl["time"].values
    ref_x     = amcl["x"].values
    ref_y     = amcl["y"].values

    sources = ["kf", "ekf", "pf"]
    colors  = {"kf": "blue", "ekf": "green", "pf": "orange"}
    labels  = {"kf": "KF", "ekf": "EKF", "pf": "PF"}

    # ── RMSE Tabelle ──────────────────────────────────────────────────────────
    print("\n{'='*45}")
    print(f"{'Source':<12} {'RMSE [m]':>10} {'N samples':>12}")
    print("-" * 45)

    results = {}
    for src in sources:
        rmse, times, errors = compute_rmse(df, src, ref_times, ref_x, ref_y)
        if rmse is None:
            print(f"{labels[src]:<12} {'no data':>10}")
            continue
        results[src] = (rmse, times, errors)
        print(f"{labels[src]:<12} {rmse:>10.4f} {len(errors):>12}")

    print("=" * 45)

    # ── Plot 1: Fehler über Zeit ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax1 = axes[0]
    ax1.set_title("Position error over time vs AMCL", fontsize=12)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Error [m]")
    ax1.grid(True, alpha=0.3)

    t0 = ref_times[0]
    for src, (rmse, times, errors) in results.items():
        ax1.plot(times - t0, errors,
                 color=colors[src], lw=0.8, alpha=0.7,
                 label=f"{labels[src]} (RMSE={rmse:.3f}m)")
        ax1.axhline(rmse, color=colors[src], lw=1.5, ls="--", alpha=0.5)

    ax1.legend(loc="upper right")

    # ── Plot 2: RMSE Bar chart ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("RMSE comparison", fontsize=12)
    ax2.set_ylabel("RMSE [m]")
    ax2.grid(True, alpha=0.3, axis="y")

    src_list  = [s for s in sources if s in results]
    rmse_vals = [results[s][0] for s in src_list]
    bar_colors = [colors[s] for s in src_list]
    bar_labels = [labels[s] for s in src_list]

    bars = ax2.bar(bar_labels, rmse_vals, color=bar_colors, alpha=0.8, width=0.5)

    # Werte über den Balken
    for bar, val in zip(bars, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f"{val:.4f} m",
                 ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("rmse_evaluation.png", dpi=150)
    print("\nSaved: rmse_evaluation.png")
    plt.show()


if __name__ == "__main__":
    main()