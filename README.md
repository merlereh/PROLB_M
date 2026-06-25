# probl_m — Robot Localization: KF, EKF, and Particle Filter

This ROS 2 package implements three localization filters for a TurtleBot3 running in Gazebo.
All three run in parallel so their estimates can be compared side-by-side in RViz against
the ground truth (from `/odom` and AMCL).

| Filter | Short description |
|--------|------------------|
| **Kalman Filter (KF)** | Linear filter. Fuses odometry with a linearized landmark observation. |
| **Extended Kalman Filter (EKF)** | Nonlinear filter using the proper range/bearing measurement model with Jacobian linearization. |
| **Particle Filter (PF)** | Monte Carlo localization with configurable resampling. |

The robot drives a fixed waypoint loop automatically — no manual teleoperation needed.

---

## Project Assignment

The full assignment description is in [`docs/Project_Description4.pdf`](docs/Project_Description4.pdf).

The specific task for this submission: **systematically analyze the effect of Q/R noise tuning**
on KF, EKF, and PF estimation accuracy — covered by Experiment 2 (`scripts/exp2_qr_tuning.py`).

---

## Requirements

### 1. Install ROS 2 Jazzy

Tested on **Ubuntu 24.04 LTS**. Follow the official installation guide:
<https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html>

Or run these commands directly:

```bash
sudo apt install software-properties-common curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
     http://packages.ros.org/ros2/ubuntu noble main" \
     | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-jazzy-desktop
```

Add the ROS environment to your shell:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Install package dependencies

```bash
sudo apt install \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup \
  ros-jazzy-turtlebot3 \
  ros-jazzy-turtlebot3-simulations \
  ros-jazzy-ros-gz \
  libeigen3-dev
```

### 3. Set the TurtleBot3 model

Add to `~/.bashrc` so it persists across terminals:

```bash
export TURTLEBOT3_MODEL=burger
```

---

## Setup

> **Note:** The commands below use `~/ros2_ws` as the workspace directory.
> Replace it with the actual path to your workspace if it has a different name.

### 1. Create workspace and clone

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/merlereh/PRO_Final.git PROLB_M
```

### 2. Build

```bash
cd ~/ros2_ws
colcon build --packages-select probl_m
source install/setup.bash
```

### 3. Gazebo World Setup

The simulation requires a small landmark pillar added to the TurtleBot3 world.
This is done by replacing the default Nav2 model file with the modified version
from this package.

**What was changed:** A single cylinder (`radius = 0.05 m`, `height = 0.5 m`) was added
at position `(1.8, 0.0)` in the world frame. The regular cylinders in the environment have
`radius = 0.15 m`, so the laser scanner can tell them apart by the arc width of the cluster
they produce in the scan.

Run this once after building (workspace name does not matter):

```bash
sudo cp $(ros2 pkg prefix probl_m)/share/probl_m/worlds/turtlebot3_world/model.sdf \
        /opt/ros/jazzy/share/nav2_minimal_tb3_sim/models/turtlebot3_world/model.sdf
```

The original can be restored at any time with:

```bash
sudo apt reinstall ros-jazzy-nav2-minimal-tb3-sim
```

The navigation map (`map.yaml` / `map.pgm`) does **not** need to be updated — the landmark
is small enough that it doesn't block navigation paths and was not present when the map
was recorded.

---

## Run

```bash
cd ~/ros2_ws/src/PROLB_M
ros2 launch probl_m full_simulation.launch.py
```

**What happens on launch:**
1. Gazebo Harmonic starts (headless) with the modified TurtleBot3 world.
2. Nav2 and AMCL start up.
3. RViz opens with a pre-configured layout.
4. After a 5-second delay (so Nav2 is ready), all filter nodes start.
5. The robot receives its initial pose and begins navigating the waypoint loop automatically.

When the run is finished, `trajectory_log.csv` is written to the directory you launched from.

---

## Nodes

### Filter nodes

| Executable | ROS node name | Output topic | Description |
|---|---|---|---|
| `kf_node` | `kf_node` | `/kf_pose` | Linear Kalman Filter. State: `[x, y, θ, vx, vy, ω]`. Correction from odometry every step, landmark correction when detected. |
| `ekf_node` | `ekf_node` | `/ekf_pose` | Extended Kalman Filter. Same state, but uses the nonlinear `h(x) = [r, φ]` measurement model for the landmark with full Jacobian linearization. |
| `pf_node` | `pf_node` | `/pf_pose`, `/my_particle_cloud` | Particle Filter (500 particles). Gaussian likelihood for landmark corrections and selective resampling. |

All three filter nodes subscribe to:
- `/odom` + `/scan` — synchronized via `message_filters` (ApproximateTime, 100 ms window)
- `/cmd_vel` — cached control input for the prediction step
- `/initialpose` — sets the starting pose (sent by `initial_pose_node` on startup)

### Support nodes

| Executable | ROS node name | Description |
|---|---|---|
| `initial_pose_node` | `initial_pose_node` | Publishes the robot's starting pose to `/initialpose`, then drives it through a fixed sequence of waypoints via Nav2's action server. |
| `evaluator_node` | `evaluator_node` | Subscribes to all filter outputs, AMCL, and `/odom`, and logs everything to `trajectory_log.csv` for offline analysis. Also records per-step filter runtime, ESS (particle filter), and resampling events. |
| `landmark_visualizer_node` | `landmark_visualizer_node` | Shows the known landmark position as a blue cylinder in RViz. Tracks the EKF pose (`/ekf_pose`) to predict where the landmark should appear in the scan. The floor disc turns **green** when the EKF detects the landmark and it passes the association gate — **red** otherwise. Note: this node uses the EKF pose only for visualization; the green indicator specifically reflects EKF landmark detection, independent of what KF or PF see. |

---

## Landmark Detection

The landmark is a thin pillar (`radius = 0.05 m`) placed at a known map position.
Detection happens in `include/landmark_scan_helper.hpp` via `detectLandmark()`:

1. **Predict where to look** — compute the expected range and bearing from the current pose estimate.
2. **Open a search window** — only look at beams within ±0.35 rad of the expected direction.
3. **Collect a cluster** — keep beams whose measured range is within ±0.25 m of the expected range.
4. **Signature check** — the arc width of the cluster must match the pillar's known radius
   (regular cylinders are ~3× wider and get rejected here).
5. **Output** — range `r` = closest cluster point + pillar radius; bearing `φ` = cluster centroid.

After detection, a simple **association gate** (Euclidean distance < 0.6 m) checks that the
projected landmark position is close enough to the known world position before the correction
is actually applied.

---

## Configuration

All noise parameters are in `config/filter_params.yaml`. Editing this file is enough — it
applies to all three filters automatically.

```yaml
# How much uncertainty the motion model adds each step
r_x, r_y, r_theta   # position and heading process noise
r_vx, r_vy, r_omega # velocity process noise

        # How much we distrust the odometry correction
q_x, q_y, q_theta

        # Landmark measurement noise (range / bearing)
q_lm_r, q_lm_phi

        # Artificial Gaussian noise injected into odometry (0 = disabled)
odom_noise_x, odom_noise_y, odom_noise_theta

        # PF resampling: 0.0 = plain multinomial, >0 = drop particles below N × average weight
pf_threshold_factor

        # Subsampling: 1 = use every message, 5 = use every 5th
skip_n
```

---

## Evaluation

After one or more simulation runs, use the experiment scripts to analyse the results.
All scripts write their output into a `results/` folder.

### Experiment 1 — Baseline comparison

Compares KF, EKF, and PF on a single run using AMCL as the ground-truth reference.

```bash
python3 scripts/exp1_baseline.py \
    --csv trajectory_log.csv \
    --out results/exp1_baseline
```

Outputs: trajectory plots with 2σ ellipses, RMSE bar chart, runtime bar chart,
covariance trace over time, and a `summary_baseline.csv` with per-filter metrics.

### Experiment 2 — Q/R noise tuning

Shows how RMSE and covariance trace change as you vary the process-noise (Q) and
measurement-noise (R) scale. Requires multiple runs saved in subfolders named `p<Q>_m<R>`
(e.g. `runs/qr/p0.5_m1.0/trajectory_log.csv`).

```bash
python3 scripts/exp2_qr_tuning.py \
    --root runs/qr \
    --out results/exp2_qr
```

Outputs: per-filter covariance heatmaps and a `summary_qr.csv` with all metrics.
The ready-made config files for each Q/R combination live in `config/experiments/qr/`.

### Experiment 3 — PF resampling threshold

Evaluates how `pf_threshold_factor` affects accuracy and particle diversity.
Requires multiple runs with different `pf_threshold_factor` values.

```bash
python3 scripts/exp3_pf_resampling.py \
    --root runs/pf_resampling \
    --out results/exp3_pf_resampling
```

Outputs: ESS over time, ESS vs. threshold, RMSE vs. threshold, and worst-case error
vs. threshold — all in `results/exp3_pf_resampling/`.

### Quick plot (single run)

```bash
python3 scripts/plot_results.py
```

Toggle which filters appear with the `SHOW_*` flags at the top of the script.
