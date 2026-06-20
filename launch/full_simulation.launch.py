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

    # Shared params file — edit config/filter_params.yaml to change noise values
    params_file = PathJoinSubstitution([
        FindPackageShare('probl_m'),
        'config',
        'experiments/pf_resampling/threshold_t1.0/filter_params.yaml'
    ])

    nav2_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'tb3_simulation_launch.py'
            ])
        ),
        launch_arguments={
            'headless': 'True',
            'rviz': 'False'
        }.items()
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d',
            PathJoinSubstitution([
                FindPackageShare('probl_m'),
                'config',
                'config_ekf.rviz'
            ])
        ],
        parameters=[{'use_sim_time': True}]
    )

    kf_node = Node(
        package='probl_m',
        executable='kf_node',
        name='kf_node',
        output='screen',
        parameters=[
            params_file,               # <-- loads filter_params.yaml
            {'use_sim_time': True}
        ]
    )

    ekf_node = Node(
        package='probl_m',
        executable='ekf_node',
        name='ekf_node',
        output='screen',
        parameters=[
            params_file,               # <-- loads filter_params.yaml
            {'use_sim_time': True}
        ]
    )

    pf_node = Node(
        package='probl_m',
        executable='pf_node',
        name='pf_node',
        output='screen',
        parameters=[
            params_file,               # <-- loads filter_params.yaml (incl. pf_threshold_factor)
            {'use_sim_time': True}
        ]
    )

    evaluator_node = Node(
        package='probl_m',
        executable='evaluator_node',
        name='evaluator_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

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

    ekf_predict_only_node = Node(
        package='probl_m',
        executable='ekf_predict_only_node',
        name='ekf_predict_only_node',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': True}
        ]
    )

    kf_predict_only_node = Node(
        package='probl_m',
        executable='kf_predict_only_node',
        name='kf_predict_only_node',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': True}
        ]
    )

    return LaunchDescription([
        nav2_simulation,
        rviz_node,
        TimerAction(
            period=5.0,
            actions=[
                kf_node,
                ekf_node,
                pf_node,
                evaluator_node,
                ekf_predict_only_node,
                kf_predict_only_node,
                initial_pose_node,
                landmark_visualizer_node,
            ]
        ),
    ])