from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    weights_arg = DeclareLaunchArgument('weights', default_value='best.pt')
    conf_thres_arg = DeclareLaunchArgument('conf_thres', default_value='0.25')
    iou_thres_arg = DeclareLaunchArgument('iou_thres', default_value='0.45')
    device_arg = DeclareLaunchArgument('device', default_value='')
    img_size_arg = DeclareLaunchArgument('img_size', default_value='640')

    object_detect_node = Node(
        package='object_detection',
        executable='object_detection',
        # Pass launch arguments to node parameters
        parameters=[
            {'weights': LaunchConfiguration('weights')},
            {'conf_thres': LaunchConfiguration('conf_thres')},
            {'iou_thres': LaunchConfiguration('iou_thres')},
            {'device': LaunchConfiguration('device')},
            {'img_size': LaunchConfiguration('img_size')},
        ]
    )

    # Create the launch description
    ld = LaunchDescription()

    # Add the launch arguments and nodes to the launch description
    ld.add_action(weights_arg)
    ld.add_action(conf_thres_arg)
    ld.add_action(iou_thres_arg)
    ld.add_action(device_arg)
    ld.add_action(img_size_arg)
    ld.add_action(object_detect_node)

    return ld



