#!/usr/bin/env python3
"""
Launch ROS 2 controllers for the robot.

This script creates a launch description that starts the necessary controllers
for operating the robotic arm and optionally the gripper in a specific sequence.

Launched Controllers:
    1. Joint State Broadcaster: Publishes joint states to /joint_states
    2. Arm Controller: Controls the robot arm movements via /follow_joint_trajectory
    3. Gripper Action Controller (optional): Controls gripper actions via /gripper_action

Launch Sequence:
    1. Joint State Broadcaster
    2. Arm Controller (starts after Joint State Broadcaster)
    3. Gripper Action Controller (starts after Arm Controller, if use_gripper:=true)

:author: Addison Sears-Collins
:date: November 15, 2024
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate a launch description for sequentially starting robot controllers.

    Returns:
        LaunchDescription: Launch description containing sequenced controller starts
    """
    # Declare launch arguments
    declare_use_gripper_cmd = DeclareLaunchArgument(
        name='use_gripper',
        default_value='false',
        description='Whether to load the gripper controller')

    use_gripper = LaunchConfiguration('use_gripper')

    # Start arm controller
    start_arm_controller_cmd = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'arm_controller'],
        output='screen')

    # Start gripper action controller (conditional)
    start_gripper_action_controller_cmd = ExecuteProcess(
        condition=IfCondition(use_gripper),
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'gripper_action_controller'],
        output='screen')

    # Launch joint state broadcaster
    start_joint_state_broadcaster_cmd = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen')

    # Add delay to joint state broadcaster (if necessary)
    delayed_start = TimerAction(
        period=10.0,
        actions=[start_joint_state_broadcaster_cmd]
    )

    # Register event handlers for sequencing
    # Launch the joint state broadcaster after spawning the robot
    load_joint_state_broadcaster_cmd = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=start_joint_state_broadcaster_cmd,
            on_exit=[start_arm_controller_cmd]))

    # Launch the gripper controller after the arm controller (if enabled)
    load_gripper_controller_cmd = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=start_arm_controller_cmd,
            on_exit=[start_gripper_action_controller_cmd]))

    # Create the launch description and populate
    ld = LaunchDescription()

    # Add the actions to the launch description in sequence
    ld.add_action(declare_use_gripper_cmd)
    ld.add_action(delayed_start)
    ld.add_action(load_joint_state_broadcaster_cmd)
    ld.add_action(load_gripper_controller_cmd)

    return ld
