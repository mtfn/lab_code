from launch import LaunchDescription
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from geometry_msgs.msg import Quaternion
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from geometry_msgs.msg import Quaternion
import numpy as np
import os

def get_share_file(package_name, file_name):
    return os.path.join(get_package_share_directory(package_name), file_name)

def launch_setup(context, *args, **kwargs):
    robot_name = LaunchConfiguration('namespace')
    mode = LaunchConfiguration('mode')

    # define node to launch and parameters to use
    marvelmind_ros2_node = Node(
        package='marvelmind_ros2',
        name='marvelmind_node',
        namespace=robot_name,
        executable='marvelmind_ros2',
        output='screen',
        arguments=['--ros-args', '--log-level', 'rclcpp:=WARN', '--log-level', 'hedgehog_logger:=INFO'],
        parameters=[LaunchConfiguration('marvelmind_ros2_config_file')],
    )

    # define node to launch and parameters to use
    turtlebot4_ekf_node = Node(
        package='turtlebot4_mekf',
        name='mekf',
        namespace=robot_name,
        executable='Turtleb4EKF',
        output='screen',
    )

    return [
        marvelmind_ros2_node,
        turtlebot4_ekf_node,
    ]


def generate_launch_description():
    # ld = LaunchDescription()
    # define node to launch and parameters to use
    robot_name_launch_arg = DeclareLaunchArgument('namespace',default_value='anotherRandomName',description='namespace of the robot.')
    
    marvelmind_ros2_config_file = get_share_file(
        package_name='turtlebot4_mekf', file_name='config/marvelmind_ros2_config.yaml'
    )
    
    marvelmind_ros2_config = DeclareLaunchArgument(
        'marvelmind_ros2_config_file',
        default_value=marvelmind_ros2_config_file,
        description='Path to config file for marvelmind_ros2_config parameters'
    )

    return LaunchDescription([
        robot_name_launch_arg,
        marvelmind_ros2_config,
        OpaqueFunction(function=launch_setup),
    ])
    
    
