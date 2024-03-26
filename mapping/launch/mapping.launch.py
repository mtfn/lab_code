from launch import LaunchDescription
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from geometry_msgs.msg import Quaternion
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from geometry_msgs.msg import Quaternion
import numpy as np
import os

def get_share_file(package_name, file_name):
    return os.path.join(get_package_share_directory(package_name), file_name)

def launch_setup(context, *args, **kwargs):
    robot_name = LaunchConfiguration('namespace')
    dis = LaunchConfiguration('range')
    spd = LaunchConfiguration('speed')
    fpl = LaunchConfiguration('fpl')
    others = LaunchConfiguration('others')

    # define node to launch and parameters to use
    scan_node = Node(
        package='lidar_to_global',
        name='scanNTran_node',
        namespace=robot_name,
        executable='scanNTran',
        output='screen',
        parameters=[{'speed': spd},{'distance': dis},{'append': True}],
    )

    # define node to launch and parameters to use
    mapping_node = Node(
        package='mapping',
        name='singleMapping',
        namespace=robot_name,
        executable='Mapping',
        output='screen',
        parameters=[{'fpl': fpl}],
    )

    return [
        scan_node,
        mapping_node,
    ]


def generate_launch_description():
    robot_name_launch_arg = DeclareLaunchArgument('namespace',default_value='/',description='namespace of the robot.')
    distance_launch_arg = DeclareLaunchArgument('range',default_value='6.0',description='length of path in ft.')
    speed_launch_arg = DeclareLaunchArgument('speed',default_value='0.07',description='speed of robot in m/s.')
    fpl_launch_arg = DeclareLaunchArgument('fpl',default_value='/home/ubuntu/fpl.pcl',description='Path to fpl.pcl file')
    others_launch_arg = DeclareLaunchArgument('others',default_value='',description='Other robots to read points from.')

    return LaunchDescription([
        robot_name_launch_arg,
        distance_launch_arg,
        speed_launch_arg,
        fpl_launch_arg,
        OpaqueFunction(function=launch_setup),
        IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    FindPackageShare("turtlebot4_mekf"), '/launch', '/turtlebot4_mekf.launch.py'])
        ),
    ])