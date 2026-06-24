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

## Requirements

Tested with **ROS 2 Humble** on Ubuntu 22.04.

```bash
sudo apt install \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-turtlebot3 \
  ros-humble-turtlebot3-simulations \
  libeigen3-dev
```

Set the TurtleBot3 model (add to `~/.bashrc` so it persists):

```bash
export TURTLEBOT3_MODEL=burger
```

---

## Gazebo World

The default TurtleBot3 world needs a small extra pillar to serve as the landmark.
This package ships a pre-modified world file at `worlds/tb3_world_landmark.world`.

**What was changed:** A single cylinder (`radius = 0.05 m`, `height = 0.5 m`) was placed at
`(1.8, 0.0)` in the world frame. The regular cylinders in the environment have `radius = 0.15 m`,
so the laser scanner can tell them apart by the width of the cluster they produce in the scan.

The navigation map (`map.yaml` / `map.pgm`) does **not** need to be updated — the landmark
is small enough that it doesn't visibly block paths and was not present when the map was recorded.

The launch file automatically points Gazebo to this custom world file, so no extra setup is needed.

---

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select probl_m
source install/setup.bash
```

---

## Run

```bash
cd ~/ros2_ws/src/probl_m
ros2 launch probl_m full_simulation.launch.py
```

**What happens on launch:**
1. Gazebo starts (headless) with the landmark world.
2. Nav2 and AMCL start up.
3. RViz opens with a pre-configured layout.
4. After a 5-second delay (so Nav2 is ready), all filter nodes start.
5. The robot receives its initial pose and begins navigating the waypoint loop.

---

## Nodes

### Filter nodes

| Executable | ROS node name | Output topic | Description |
|---|---|---|---|
| `kf_node` | `kf_node` | `/kf_pose` | Linear Kalman Filter. State: `[x, y, θ, vx, vy, ω]`. Correction from odometry every step, landmark correction when detected. |
| `ekf_node` | `ekf_node` | `/ekf_pose` | Extended Kalman Filter. Same state, but uses the nonlinear `h(x) = [r, φ]` measurement model for the landmark. |
| `pf_node` | `pf_node` | `/pf_pose`, `/my_particle_cloud` | Particle Filter (500 particles). Uses a Gaussian likelihood for landmark corrections and selective resampling. |

All three filter nodes subscribe to:
- `/odom` + `/scan` — synchronized via `message_filters` (ApproximateTime, 100 ms window)
- `/cmd_vel` — cached control input for the prediction step
- `/initialpose` — sets the starting pose (sent by `initial_pose_node` on startup)

### Support nodes

| Executable | ROS node name | Description |
|---|---|---|
| `initial_pose_node` | `initial_pose_node` | Publishes the robot's starting pose to `/initialpose`, then drives it through a fixed sequence of waypoints via Nav2's action server. |
| `evaluator_node` | `evaluator_node` | Subscribes to all filter outputs, AMCL, and `/odom`, and logs everything to `trajectory_log.csv` for offline analysis. Also records per-step filter runtime, ESS (particle filter), and resampling events. |
| `landmark_visualizer_node` | `landmark_visualizer_node` | Shows the known landmark position as a blue cylinder in RViz. Runs the same detection logic as the filter nodes and turns a floor indicator green when the landmark is in view and passes the association gate, red otherwise. |

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

## Post-run Analysis

```bash
python3 scripts/plot_results.py
```

Toggle which filters appear in the plot by setting the `SHOW_*` flags at the top of the script.
The CSV file `trajectory_log.csv` is written in whichever directory you ran the launch from.