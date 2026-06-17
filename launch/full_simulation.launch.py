"""
cd ~/ros2_ws
colcon build --packages-select probl_m
source install/setup.bash
cd ~/ros2_ws/src/probl_m
ros2 launch probl_m full_simulation.launch.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # Start Nav2 TurtleBot simulation.
    # headless:=False avoids a Jazzy shutdown bug with the ROS adapter.
    nav2_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'tb3_simulation_launch.py'
            ])
        ),
        launch_arguments={
            'headless': 'True'
        }.items()
    )

    # Kalman Filter node
    kf_node = Node(
        package='probl_m',
        executable='kf_node',
        name='kf_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Extended Kalman Filter node
    ekf_node = Node(
        package='probl_m',
        executable='ekf_node',
        name='ekf_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Particle Filter node
    pf_node = Node(
        package='probl_m',
        executable='pf_node',
        name='pf_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'resampling_method': 'multinomial'}
        ]
    )

    # Evaluator node
    evaluator_node = Node(
        package='probl_m',
        executable='evaluator_node',
        name='evaluator_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Combined initial pose + waypoint node.
    # Waits for AMCL (/amcl_pose) before sending the initial pose,
    # then navigates through all waypoints one by one via NavigateToPose.
    initial_pose_node = Node(
        package='probl_m',
        executable='initial_pose_node',
        name='initial_pose_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    landmark_visualizer_node = Node(
        package='probl_m',
        executable='landmark_visualizer_node',
        name='landmark_visualizer_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        nav2_simulation,

        TimerAction(
            period=5.0,
            actions=[
                kf_node,
                ekf_node,
                pf_node,
                evaluator_node,
                initial_pose_node,
                landmark_visualizer_node,
            ]
        ),
    ])