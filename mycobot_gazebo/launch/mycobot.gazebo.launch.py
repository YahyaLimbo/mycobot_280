#!/usr/bin/env python3
"""
Launch Ignition Gazebo Fortress simulation with a myCobot robot.

This launch file sets up a complete ROS 2 simulation environment with Ignition
Gazebo Fortress for a myCobot robot.

Adapted from automaticaddison/mycobot_ros2 for ROS2 Humble + Ignition Fortress.

:author: Original by Addison Sears-Collins, adapted for Ignition Fortress
:date: February 2026
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Generate a launch description for the Ignition Gazebo Fortress simulation.

    Returns:
        LaunchDescription: A complete launch description for the simulation
    """
    # Constants for paths to different files and folders
    package_name_gazebo = 'mycobot_gazebo'
    package_name_description = 'mycobot_description'
    package_name_moveit = 'mycobot_moveit_config'

    default_robot_name = 'mycobot_280'
    gazebo_models_path = 'models'
    default_world_file = 'empty.world'
    gazebo_worlds_path = 'worlds'

    ros_ign_bridge_config_file_path = 'config/ros_ign_bridge.yaml'

    # Set the path to different files and folders
    pkg_ros_ign_gazebo = FindPackageShare(package='ros_ign_gazebo').find('ros_ign_gazebo')
    pkg_share_gazebo = FindPackageShare(package=package_name_gazebo).find(package_name_gazebo)
    pkg_share_description = FindPackageShare(
        package=package_name_description).find(package_name_description)
    pkg_share_moveit = FindPackageShare(package=package_name_moveit).find(package_name_moveit)

    gazebo_models_path_full = os.path.join(pkg_share_gazebo, gazebo_models_path)
    default_ros_ign_bridge_config_file_path = os.path.join(
        pkg_share_gazebo, ros_ign_bridge_config_file_path)

    # Launch configuration variables
    jsp_gui = LaunchConfiguration('jsp_gui')
    load_controllers = LaunchConfiguration('load_controllers')
    robot_name = LaunchConfiguration('robot_name')
    use_rviz = LaunchConfiguration('use_rviz')
    use_camera = LaunchConfiguration('use_camera')
    use_gazebo = LaunchConfiguration('use_gazebo')
    use_gripper = LaunchConfiguration('use_gripper')
    use_robot_state_pub = LaunchConfiguration('use_robot_state_pub')
    use_sim_time = LaunchConfiguration('use_sim_time')
    world_file = LaunchConfiguration('world_file')

    world_path = PathJoinSubstitution([
        pkg_share_gazebo,
        gazebo_worlds_path,
        world_file
    ])

    # Set the pose configuration variables
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    z = LaunchConfiguration('z')
    roll = LaunchConfiguration('roll')
    pitch = LaunchConfiguration('pitch')
    yaw = LaunchConfiguration('yaw')

    # Declare the launch arguments
    declare_robot_name_cmd = DeclareLaunchArgument(
        name='robot_name',
        default_value=default_robot_name,
        description='The name for the robot')

    declare_load_controllers_cmd = DeclareLaunchArgument(
        name='load_controllers',
        default_value='true',
        description='Flag to enable loading of ROS 2 controllers')

    declare_use_robot_state_pub_cmd = DeclareLaunchArgument(
        name='use_robot_state_pub',
        default_value='true',
        description='Flag to enable robot state publisher')

    # GUI and visualization arguments
    declare_jsp_gui_cmd = DeclareLaunchArgument(
        name='jsp_gui',
        default_value='false',
        description='Flag to enable joint_state_publisher_gui')

    declare_use_camera_cmd = DeclareLaunchArgument(
        name='use_camera',
        default_value='false',
        description='Flag to enable the RGBD camera for Gazebo point cloud simulation')

    declare_use_gazebo_cmd = DeclareLaunchArgument(
        name='use_gazebo',
        default_value='true',
        description='Flag to enable Gazebo')

    declare_use_gripper_cmd = DeclareLaunchArgument(
        name='use_gripper',
        default_value='false',
        description='Flag to enable the gripper on the robot')

    declare_use_rviz_cmd = DeclareLaunchArgument(
        name='use_rviz',
        default_value='true',
        description='Flag to enable RViz')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')

    declare_world_cmd = DeclareLaunchArgument(
        name='world_file',
        default_value=default_world_file,
        description='World file name (e.g., empty.world, calibration.world)')

    # Pose arguments
    declare_x_cmd = DeclareLaunchArgument(
        name='x',
        default_value='0.0',
        description='x component of initial position, meters')

    declare_y_cmd = DeclareLaunchArgument(
        name='y',
        default_value='0.0',
        description='y component of initial position, meters')

    declare_z_cmd = DeclareLaunchArgument(
        name='z',
        default_value='0.05',
        description='z component of initial position, meters')

    declare_roll_cmd = DeclareLaunchArgument(
        name='roll',
        default_value='0.0',
        description='roll angle of initial orientation, radians')

    declare_pitch_cmd = DeclareLaunchArgument(
        name='pitch',
        default_value='0.0',
        description='pitch angle of initial orientation, radians')

    declare_yaw_cmd = DeclareLaunchArgument(
        name='yaw',
        default_value='0.0',
        description='yaw angle of initial orientation, radians')

    # Include Robot State Publisher launch file if enabled
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_share_description, 'launch', 'robot_state_publisher.launch.py')
        ]),
        launch_arguments={
            'jsp_gui': jsp_gui,
            'use_camera': use_camera,
            'use_gazebo': use_gazebo,
            'use_gripper': use_gripper,
            'use_rviz': use_rviz,
            'use_sim_time': use_sim_time
        }.items(),
        condition=IfCondition(use_robot_state_pub)
    )

    # Include ROS 2 Controllers launch file if enabled
    load_controllers_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_share_moveit, 'launch', 'load_ros2_controllers.launch.py')
        ]),
        launch_arguments={
            'use_gripper': use_gripper,
            'use_sim_time': use_sim_time
        }.items(),
        condition=IfCondition(load_controllers)
    )

    # Set Ignition Gazebo resource path environment variable
    set_ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            gazebo_models_path_full, ':',
            pkg_share_description, ':',
            os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
        ]
    )

    # Start Ignition Gazebo Fortress
    start_gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_ign_gazebo, 'launch', 'ign_gazebo.launch.py')),
        launch_arguments=[('ign_args', [' -r -v 4 ', world_path])])

    # Bridge ROS topics and Ignition messages for establishing communication
    start_gazebo_ros_bridge_cmd = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': default_ros_ign_bridge_config_file_path,
        }],
        output='screen'
    )

    # Includes optimizations to minimize latency and bandwidth when streaming image data
    start_gazebo_ros_image_bridge_cmd = Node(
        package='ros_ign_image',
        executable='image_bridge',
        arguments=[
            '/camera_head/depth_image',
            '/camera_head/image',
        ],
        remappings=[
            ('/camera_head/depth_image', '/camera_head/depth/image_rect_raw'),
            ('/camera_head/image', '/camera_head/color/image_raw'),
        ],
    )

    # Spawn the robot in Ignition Gazebo
    start_gazebo_ros_spawner_cmd = Node(
        package='ros_ign_gazebo',
        executable='create',
        output='screen',
        arguments=[
            '-topic', '/robot_description',
            '-name', robot_name,
            '-allow_renaming', 'true',
            '-x', x,
            '-y', y,
            '-z', z,
            '-R', roll,
            '-P', pitch,
            '-Y', yaw
        ])

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_robot_name_cmd)
    ld.add_action(declare_jsp_gui_cmd)
    ld.add_action(declare_load_controllers_cmd)
    ld.add_action(declare_use_camera_cmd)
    ld.add_action(declare_use_gazebo_cmd)
    ld.add_action(declare_use_gripper_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_use_robot_state_pub_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_world_cmd)

    # Add pose arguments
    ld.add_action(declare_x_cmd)
    ld.add_action(declare_y_cmd)
    ld.add_action(declare_z_cmd)
    ld.add_action(declare_roll_cmd)
    ld.add_action(declare_pitch_cmd)
    ld.add_action(declare_yaw_cmd)

    # Add the actions to the launch description
    ld.add_action(set_ign_resource_path)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(load_controllers_cmd)
    ld.add_action(start_gazebo_cmd)
    ld.add_action(start_gazebo_ros_bridge_cmd)
    ld.add_action(start_gazebo_ros_image_bridge_cmd)
    ld.add_action(start_gazebo_ros_spawner_cmd)

    return ld
