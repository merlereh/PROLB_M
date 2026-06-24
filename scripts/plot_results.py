"""
python3 scripts/plot_results.py

Welche Filter anzeigen? Einfach True/False setzen:
"""

SHOW_ODOM             = False
SHOW_KF               = True
SHOW_EKF              = True
SHOW_PF               = True
SHOW_AMCL             = True
SHOW_EKF_PREDICT_ONLY = False

# Ellipse alle N Posen zeichnen (0 = keine Ellipsen)
# -> kleinerer Wert = mehr Ellipsen (dichter), größerer Wert = weniger (sparsamer)
ELLIPSE_EVERY = 30
N_STD         = 2.0   # 1-sigma, 2-sigma, ...

# Landmark(s) einzeichnen — Koordinaten müssen zu LANDMARK_X/LANDMARK_Y in
# kalman_filter_node.cpp / extended_kalman_filter_node.cpp / particle_filter_node.cpp passen
SHOW_LANDMARKS = True
LANDMARKS = [(1.8, 0.0)]   # Liste von (x, y), falls später mehr Landmarks dazukommen

CSV_FILE = "trajectory_log.csv"

# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def cov_ellipse(x, y, cov_xx, cov_yy, cov_xy, n_std=2.0, **kwargs):
    cov = np.array([[cov_xx, cov_xy],
                    [cov_xy, cov_yy]])

    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    vals = np.maximum(vals, 0.0)

    width  = 2.0 * n_std * np.sqrt(vals[0])
    height = 2.0 * n_std * np.sqrt(vals[1])
    angle  = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    return Ellipse(
        xy=(x, y),
        width=width,
        height=height,
        angle=angle,
        **kwargs
    )


def main():
    df = pd.read_csv(CSV_FILE)

    # Which sources to actually plot — driven by the toggles above
    active = {
        "odom":               SHOW_ODOM,
        "kf":                 SHOW_KF,
        "ekf":                SHOW_EKF,
        "pf":                 SHOW_PF,
        "amcl":               SHOW_AMCL,
        "ekf_predict_only":   SHOW_EKF_PREDICT_ONLY,
    }

    style = {
        "odom":             dict(color="gray",    lw=1.2, ls="--", label="Odometry",                        zorder=1),
        "kf":               dict(color="blue",    lw=1.5, ls="-",  label="KF",                              zorder=3),
        "ekf":              dict(color="green",   lw=1.5, ls="-",  label="EKF",                             zorder=3),
        "pf":               dict(color="orange",  lw=1.5, ls="-",  label="PF",                              zorder=3),
        "amcl":             dict(color="red",     lw=1.2, ls="-",  label="AMCL (Ref)",                      zorder=2),
        "ekf_predict_only": dict(color="#e74c3c", lw=1.0, ls=":",  label="EKF predict-only (kein Correct)", zorder=2),
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Filter Trajectories with Uncertainty Ellipses", fontsize=13)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    for src, s in style.items():
        if not active[src]:
            continue
        if src not in df["source"].unique():
            continue

        sub = df[df["source"] == src].reset_index(drop=True)

        ax.plot(sub["x"], sub["y"],
                color=s["color"], lw=s["lw"], ls=s["ls"],
                label=s["label"], zorder=s["zorder"])

        ax.plot(sub["x"].iloc[0], sub["y"].iloc[0],
                "o", color=s["color"], ms=6, zorder=5)

        # Ellipses (skip odom, skip if ELLIPSE_EVERY == 0)
        has_cov = {"cov_xx", "cov_yy", "cov_xy"}.issubset(df.columns)
        if src != "odom" and ELLIPSE_EVERY > 0 and has_cov:
            for i in range(0, len(sub), ELLIPSE_EVERY):
                row = sub.iloc[i]
                cxx, cyy, cxy = row["cov_xx"], row["cov_yy"], row["cov_xy"]
                if cxx == 0.0 and cyy == 0.0:
                    continue
                ell = cov_ellipse(
                    row["x"], row["y"], cxx, cyy, cxy,
                    n_std=N_STD,
                    linewidth=0.8,
                    edgecolor=s["color"],
                    facecolor=s["color"],
                    alpha=0.15,
                    zorder=s["zorder"] - 1
                )
                ax.add_patch(ell)

    if SHOW_LANDMARKS:
        for i, (lx, ly) in enumerate(LANDMARKS):
            ax.scatter(
                [lx], [ly],
                marker="*", s=350,
                color="gold", edgecolor="black", linewidth=1.2,
                zorder=6, label="Landmark" if i == 0 else None
            )

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("trajectories_with_ellipses.png", dpi=150)
    print("Saved: trajectories_with_ellipses.png")
    plt.show()


if __name__ == "__main__":
    main()