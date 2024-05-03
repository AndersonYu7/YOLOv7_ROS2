from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    weights_arg = DeclareLaunchArgument('weights', default_value='yolov7.pt', description='Weights file')
    conf_thres_arg = DeclareLaunchArgument('conf_thres', default_value='0.25', description='Confidence threshold')
    iou_thres_arg = DeclareLaunchArgument('iou_thres', default_value='0.45', description='IOU threshold')
    device_arg = DeclareLaunchArgument('device', default_value='', description='Name of the device')
    img_size_arg = DeclareLaunchArgument('img_size', default_value='640', description='Image size')
    show_img_arg = DeclareLaunchArgument('show_img', default_value='True', description='Show image or not')

    object_detect_node = Node(
        package='yolov7_obj_detect',
        executable='object_detection',
        # Pass launch arguments to node parameters
        parameters=[
            {'weights': LaunchConfiguration('weights')},
            {'conf_thres': LaunchConfiguration('conf_thres')},
            {'iou_thres': LaunchConfiguration('iou_thres')},
            {'device': LaunchConfiguration('device')},
            {'img_size': LaunchConfiguration('img_size')},
            {'show_img': LaunchConfiguration('show_img')},
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
    ld.add_action(show_img_arg)
    ld.add_action(object_detect_node)

    return ld



