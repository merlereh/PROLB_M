#!/usr/bin/env python3
"""
Generate ROS parameter YAML files for Q/R tuning and PF resampling experiments.

Your current implementation uses:
    r_*      = process noise parameters
    q_odom_* and q_lm_* = measurement noise parameters

This script creates one modified YAML per experiment setting. You then run the
same trajectory once per generated config and save the resulting trajectory_log.csv.

Examples:
    python3 scripts/evaluation/generate_experiment_configs.py \
        --base config/filter_params.yaml \
        --out config/experiments

Generated:
    config/experiments/qr/p0.5_m0.5/filter_params.yaml
    config/experiments/qr/p0.5_m1.0/filter_params.yaml
    ...
    config/experiments/pf_resampling/threshold_t0.0/filter_params.yaml
    config/experiments/pf_resampling/threshold_t0.5/filter_params.yaml
    ...
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Install PyYAML first: pip install pyyaml") from exc

PROCESS_KEYS = ["r_x", "r_y", "r_theta", "r_vx", "r_vy", "r_omega"]
MEASUREMENT_KEYS = ["q_odom_x", "q_odom_y", "q_odom_theta", "q_lm_r", "q_lm_phi"]


def load_params(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def ros_params(data: dict) -> dict:
    return data["/**"]["ros__parameters"]


def write_yaml(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def scaled_config(base: dict, process_scale: float, measurement_scale: float) -> dict:
    data = deepcopy(base)
    params = ros_params(data)
    for key in PROCESS_KEYS:
        if key in params:
            params[key] = float(params[key]) * process_scale
    for key in MEASUREMENT_KEYS:
        if key in params:
            params[key] = float(params[key]) * measurement_scale
    return data


def threshold_config(base: dict, threshold: float) -> dict:
    data = deepcopy(base)
    params = ros_params(data)
    params["pf_threshold_factor"] = float(threshold)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="config/filter_params.yaml")
    ap.add_argument("--out", default="config/experiments")
    ap.add_argument("--process-scales", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--measurement-scales", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5])
    args = ap.parse_args()

    base_path = Path(args.base)
    out = Path(args.out)
    base = load_params(base_path)

    qr_manifest = []
    for p in args.process_scales:
        for m in args.measurement_scales:
            name = f"p{p}_m{m}"
            cfg_path = out / "qr" / name / "filter_params.yaml"
            write_yaml(scaled_config(base, p, m), cfg_path)
            qr_manifest.append({"setting": name, "process_scale": p, "measurement_scale": m, "config": str(cfg_path)})

    pf_manifest = []
    for t in args.thresholds:
        name = f"threshold_t{t}"
        cfg_path = out / "pf_resampling" / name / "filter_params.yaml"
        write_yaml(threshold_config(base, t), cfg_path)
        pf_manifest.append({"setting": name, "method": "threshold", "threshold": t, "config": str(cfg_path)})

    # Write manifest templates. After running ROS, add/fill the csv column.
    import pandas as pd
    qr_df = pd.DataFrame(qr_manifest)
    qr_df["csv"] = "TODO/path/to/trajectory_log.csv"
    qr_df[["csv", "process_scale", "measurement_scale", "setting", "config"]].to_csv(out / "qr_manifest_template.csv", index=False)

    pf_df = pd.DataFrame(pf_manifest)
    pf_df["csv"] = "TODO/path/to/trajectory_log.csv"
    pf_df[["csv", "method", "threshold", "setting", "config"]].to_csv(out / "pf_resampling_manifest_template.csv", index=False)

    print(f"Generated configs under: {out}")
    print(f"Q/R manifest template: {out / 'qr_manifest_template.csv'}")
    print(f"PF manifest template:  {out / 'pf_resampling_manifest_template.csv'}")


if __name__ == "__main__":
    main()
