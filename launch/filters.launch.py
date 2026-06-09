from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='probl_m',
            executable='kalman_filter_node',
            name='kalman_filter_node',
            output='screen'
        ),

        # wenn andere Filter fertig sind:
        # Node(
        #     package='probl_m',
        #     executable='ekf_node',
        #     name='ekf_node',
        #     output='screen'
        # ),

        # Node(
        #     package='probl_m',
        #     executable='particle_filter_node',
        #     name='particle_filter_node',
        #     output='screen'
        # ),
    ])