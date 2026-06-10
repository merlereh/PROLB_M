PROBL_M — Probabilistic Robotics Filters with ROS 2 and Nav2

This project implements and evaluates three probabilistic state estimation methods for mobile robotics in a ROS 2 environment:

* Kalman Filter (KF)
* Extended Kalman Filter (EKF)
* Particle Filter (PF)

The filters run as independent ROS 2 nodes in parallel to a Nav2 TurtleBot simulation. Each filter subscribes to velocity and odometry data, estimates the robot pose, and publishes its own pose topic. An additional evaluator node records the estimated trajectories and transforms them into the map frame for comparison and visualization.

⸻

Project Goal

The goal of this project is to compare different probabilistic state estimation methods in a mobile robotics context.

The implemented filters estimate the robot state:

x = [x, y, theta]

where:

* x is the robot position in x-direction
* y is the robot position in y-direction
* theta is the robot orientation/yaw angle

The filters are evaluated by comparing their estimated trajectories against odometry and AMCL/Nav2 localization.

⸻

System Overview

The system is based on ROS 2 Jazzy and Nav2.

Nav2 is used to run the TurtleBot simulation and generate realistic robot motion. The custom filter nodes run in parallel and do not control Nav2 directly.

Nav2 / TurtleBot Simulation
        |
        | publishes
        v
   /cmd_vel, /odom, /tf, /amcl_pose
        |
        | subscribed by
        v
+---------------------+
|   Kalman Filter     | ---> /kf_pose
+---------------------+
+---------------------+
| Extended Kalman     | ---> /ekf_pose
| Filter              |
+---------------------+
+---------------------+
| Particle Filter     | ---> /pf_pose
+---------------------+
+---------------------+
| Evaluator Node      | ---> trajectory_log.csv
+---------------------+

⸻

ROS Topics

Input Topics

The filter nodes subscribe to:

Topic	Message Type	Description
/cmd_vel	geometry_msgs/msg/Twist	Velocity command input
/odom	nav_msgs/msg/Odometry	Odometry pose measurement

The evaluator node subscribes to:

Topic	Message Type	Description
/odom	nav_msgs/msg/Odometry	Robot odometry
/kf_pose	geometry_msgs/msg/PoseWithCovarianceStamped	Kalman Filter estimate
/ekf_pose	geometry_msgs/msg/PoseWithCovarianceStamped	Extended Kalman Filter estimate
/pf_pose	geometry_msgs/msg/PoseWithCovarianceStamped	Particle Filter estimate
/amcl_pose	geometry_msgs/msg/PoseWithCovarianceStamped	Nav2 AMCL pose estimate
/tf	tf2_msgs/msg/TFMessage	Coordinate frame transformations

Output Topics

Topic	Message Type	Description
/kf_pose	geometry_msgs/msg/PoseWithCovarianceStamped	KF pose estimate
/ekf_pose	geometry_msgs/msg/PoseWithCovarianceStamped	EKF pose estimate
/pf_pose	geometry_msgs/msg/PoseWithCovarianceStamped	PF pose estimate
/my_particle_cloud	geometry_msgs/msg/PoseArray	Particle cloud for visualization
trajectory_log.csv	CSV file	Logged trajectory data

⸻

Package Structure

probl_m/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
├── include/
│   └── probl_m/
│       ├── kalman_filter.hpp
│       ├── extended_kalman_filter.hpp
│       └── particle_filter.hpp
├── launch/
│   └── filters.launch.py
├── scripts/
│   └── plot_results.py
└── src/
    ├── kalman_filter_node.cpp
    ├── extended_kalman_filter_node.cpp
    ├── particle_filter_node.cpp
    └── evaluator_node.cpp

⸻

Implemented Nodes

1. Kalman Filter Node

Executable:

kf_node

The Kalman Filter node uses a simplified motion model and odometry as direct pose measurement.

Input:

/cmd_vel
/odom

Output:

/kf_pose

The state prediction is based on:

x'     = x + v * cos(theta) * dt
y'     = y + v * sin(theta) * dt
theta' = theta + omega * dt

The correction step uses the odometry pose as measurement:

z = [x_odom, y_odom, theta_odom]

⸻

2. Extended Kalman Filter Node

Executable:

ekf_node

The Extended Kalman Filter uses the same nonlinear motion model as the Kalman Filter, but additionally linearizes the motion model using a Jacobian matrix.

Input:

/cmd_vel
/odom

Output:

/ekf_pose

The EKF uses the Jacobian:

G =
[ 1  0  -v * sin(theta) * dt ]
[ 0  1   v * cos(theta) * dt ]
[ 0  0   1                    ]

The covariance prediction is:

Sigma_bar = G * Sigma * G^T + R

⸻

3. Particle Filter Node

Executable:

pf_node

The Particle Filter represents the robot belief using a set of particles. Each particle contains a possible robot pose:

particle = [x, y, theta]

Input:

/cmd_vel
/odom

Output:

/pf_pose
/my_particle_cloud

The Particle Filter performs:

1. Particle initialization
2. Motion update using velocity input
3. Weighting using odometry pose measurement
4. Resampling
5. Pose estimation from particle mean

The topic /my_particle_cloud can be visualized in RViz.

⸻

4. Evaluator Node

Executable:

evaluator_node

The evaluator node records the trajectories of all filters and reference sources.

It subscribes to:

/odom
/kf_pose
/ekf_pose
/pf_pose
/amcl_pose

Before writing the data to CSV, the evaluator transforms all poses into the map frame using TF.

Output file:

trajectory_log.csv

CSV format:

time,source,frame,x,y,theta

Example:

12.34,odom,map,1.23,2.45,0.52
12.35,kf,map,1.24,2.46,0.51
12.36,ekf,map,1.25,2.47,0.50
12.37,pf,map,1.22,2.44,0.53
12.38,amcl,map,1.20,2.40,0.55

⸻

Build Instructions

Go to the workspace root:

cd ~/ros2_ws

Build the package:

colcon build --packages-select probl_m

Source the workspace:

source install/setup.bash

⸻

Running the Simulation and Nodes

Terminal 1: Start Nav2 TurtleBot Simulation

source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch nav2_bringup tb3_simulation_launch.py headless:=False

Wait until Gazebo, RViz and Nav2 are fully started.

⸻

Terminal 2: Start Kalman Filter

source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
source install/setup.bash
ros2 run probl_m kf_node --ros-args -p use_sim_time:=true

⸻

Terminal 3: Start Extended Kalman Filter

source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
source install/setup.bash
ros2 run probl_m ekf_node --ros-args -p use_sim_time:=true

⸻

Terminal 4: Start Particle Filter

source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
source install/setup.bash
ros2 run probl_m pf_node --ros-args -p use_sim_time:=true

⸻

Terminal 5: Start Evaluator Node

source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
source install/setup.bash
ros2 run probl_m evaluator_node --ros-args -p use_sim_time:=true

The evaluator writes the trajectory data into:

~/ros2_ws/trajectory_log.csv

⸻

Plotting Results

The script plot_results.py reads the CSV file and creates a trajectory plot.

Run:

cd ~/ros2_ws
python3 src/probl_m/scripts/plot_results.py

The script saves the plot as:

trajectory_plot.png

Open the plot:

xdg-open trajectory_plot.png

⸻

Coordinate Frames

The filters publish poses based on odometry data. Odometry is usually expressed in the odom frame.

AMCL publishes its pose in the map frame.

To compare all trajectories correctly, the evaluator node transforms every pose into the map frame before writing it to the CSV file.

Typical TF structure:

map → odom → base_footprint

You can check the transform with:

ros2 run tf2_ros tf2_echo map odom

If this transform is available, the evaluator can transform odometry-based filter poses into map coordinates.

⸻

Time Handling

The filter nodes use /cmd_vel as control input and /odom as pose measurement.

Since /cmd_vel does not contain a header timestamp, the nodes store the receive time of the latest velocity command and compare it with the timestamp of the odometry message.

A filter update is only performed if the time difference is below a defined threshold.

This prevents very old velocity commands from being combined with newer odometry measurements.

⸻

Evaluation

The project can be evaluated using:

* Trajectory comparison
* RMSE between filter estimates and reference pose
* Runtime comparison
* Different process noise settings
* Different measurement noise settings
* Different trajectories
* Particle count variation for the Particle Filter
* Visualization in RViz2 and Python plots

Possible reference signals:

Reference	Description
/odom	Local odometry estimate
/amcl_pose	Nav2 localization estimate in map frame
Ground truth	Preferred if available from simulation

⸻

Useful Commands

List topics:

ros2 topic list

Check topic frequency:

ros2 topic hz /odom
ros2 topic hz /kf_pose
ros2 topic hz /ekf_pose
ros2 topic hz /pf_pose

Echo one message:

ros2 topic echo /odom --once
ros2 topic echo /kf_pose --once
ros2 topic echo /ekf_pose --once
ros2 topic echo /pf_pose --once

Check running nodes:

ros2 node list

Check TF transform:

ros2 run tf2_ros tf2_echo map odom

⸻

Notes

The custom filter nodes currently run in parallel to Nav2. Nav2 does not use /kf_pose, /ekf_pose or /pf_pose for navigation.

Nav2 continues to use its own localization stack, usually AMCL and TF.

The purpose of the custom filter nodes is to estimate the robot pose independently and compare the results against odometry, AMCL and possible ground truth data.

⸻

Authors

PRO Lab — Probabilistic Robotics with ROS 2 and TurtleBot

Package name:

probl_m
