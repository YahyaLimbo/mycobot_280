#!/usr/bin/env python3
"""Validate a hand-eye calibration by using it on an object of known pose.

Spawns an ArUco-tagged plate (marker id 1) into the running simulation, runs a
detector for it, and checks that the calibrated transform puts it where it
really is -- then optionally reaches for it.

Assumes the simulation is already up and a calibration has been solved:

  ros2 launch mycobot_gazebo mycobot_combined.launch.py \\
      use_camera:=true use_aruco_marker:=true use_gripper:=false \\
      use_gz_gui:=false use_rviz:=false world_file:=calibration.world
  ros2 launch mycobot_calibration handeye_calibration.launch.py
  ros2 run mycobot_calibration handeye_solver

then:

  ros2 launch mycobot_calibration validate_calibration.launch.py

The object's spawn pose is given in Gazebo world coordinates while the check
is done in the robot base frame. base_link sits at world
(robot_base_x, robot_base_y, robot_base_z), so the two differ by that offset
and nothing else; the launch derives one from the other rather than making the
caller keep two sets of numbers consistent. Those three must match the x/y/z
the robot was spawned at in mycobot_combined.launch.py.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declare_args = [
        DeclareLaunchArgument('result_file', default_value='/tmp/handeye_result.yaml',
                              description='Calibration produced by handeye_solver.'),
        DeclareLaunchArgument('marker_id', default_value='1',
                              description='ArUco id on the validation object.'),
        DeclareLaunchArgument('marker_size', default_value='0.08',
                              description='Black square edge of the object marker.'),
        DeclareLaunchArgument('object_x', default_value='0.12'),
        DeclareLaunchArgument('object_y', default_value='0.10'),
        DeclareLaunchArgument('object_z', default_value='0.05',
                              description='Object height above base_link.'),
        DeclareLaunchArgument('object_roll_deg', default_value='60.0',
                              description='Tilt of the plate toward the camera. '
                                          'The printed face is the plate +Z, so '
                                          'a positive roll here turns it to face '
                                          '-Y, which is where the camera stand is.'),
        DeclareLaunchArgument('robot_base_x', default_value='0.0',
                              description='base_link X in Gazebo world '
                                          'coordinates. Must match the x the '
                                          'robot was spawned at, or the object '
                                          'is placed where the arm is not.'),
        DeclareLaunchArgument('robot_base_y', default_value='0.0',
                              description='base_link Y in Gazebo world '
                                          'coordinates.'),
        DeclareLaunchArgument('robot_base_z', default_value='0.425',
                              description='Height of base_link in Gazebo world '
                                          'coordinates, i.e. the calibration '
                                          'table top.'),
        DeclareLaunchArgument('reach', default_value='true',
                              description='Also command the arm to reach for it.'),
        DeclareLaunchArgument('standoff', default_value='0.10'),
        DeclareLaunchArgument('respawn_object', default_value='true',
                              description='Remove any existing aruco_target '
                                          'before spawning, so repeat runs do '
                                          'not stack duplicates.'),
        DeclareLaunchArgument('world_name', default_value='calibration_world'),
        DeclareLaunchArgument('show_detections', default_value='false',
                              description='Open an RViz window showing the '
                                          'camera view with the object\'s '
                                          'detection drawn on it. Off by '
                                          'default because the combined sim '
                                          'launch\'s own RViz already carries '
                                          'that view; this is for the '
                                          'use_rviz:=false path.'),
    ]

    def launch_setup(context):
        pkg_gazebo = get_package_share_directory('mycobot_gazebo')
        model_path = os.path.join(pkg_gazebo, 'models', 'aruco_target', 'model.sdf')

        def value(name):
            return LaunchConfiguration(name).perform(context)

        world = value('world_name')
        base_x = float(value('robot_base_x'))
        base_y = float(value('robot_base_y'))
        base_z = float(value('robot_base_z'))
        object_z_base = float(value('object_z'))
        roll = float(value('object_roll_deg'))

        actions = []

        spawn = Node(
            package='ros_ign_gazebo',
            executable='create',
            output='screen',
            arguments=[
                '-world', world,
                '-file', model_path,
                '-name', 'aruco_target',
                '-x', str(float(value('object_x')) + base_x),
                '-y', str(float(value('object_y')) + base_y),
                '-z', str(object_z_base + base_z),
                '-R', str(roll * 3.14159265358979 / 180.0),
                '-P', '0', '-Y', '0',
            ],
        )

        if value('respawn_object').lower() == 'true':
            # Remove first, then spawn -- and strictly in that order.
            #
            # Launch starts every action it is given at once, so adding these
            # as two siblings is a race: when the spawn wins, the remove that
            # follows deletes the object that was just created and the run sees
            # nothing at all. Chaining the spawn to the removal's exit is what
            # makes repeat runs deterministic.
            #
            # The removal's own result is ignored on purpose: on a first run
            # there is nothing to remove and the service answers false, which
            # is not an error.
            remove = ExecuteProcess(
                cmd=['ign', 'service', '-s', f'/world/{world}/remove',
                     '--reqtype', 'ignition.msgs.Entity',
                     '--reptype', 'ignition.msgs.Boolean',
                     '--timeout', '3000',
                     '--req', 'name: "aruco_target" type: MODEL'],
                output='log',
            )
            actions.append(remove)
            actions.append(RegisterEventHandler(
                event_handler=OnProcessExit(target_action=remove,
                                            on_exit=[spawn]),
            ))
        else:
            actions.append(spawn)

        detector = Node(
            package='mycobot_calibration',
            executable='aruco_detector',
            name='aruco_object_detector',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'marker_id': int(value('marker_id')),
                'marker_size': float(value('marker_size')),
                'camera_frame': 'camera_head_depth_optical_frame',
                'marker_frame': 'aruco_object_detected',
                'intrinsics_source': 'computed',
                # Must match the URDF camera_hfov argument. This is hardcoded
                # rather than read from handeye_calibration.yaml because this
                # launch configures a second detector inline; if the URDF FOV
                # is changed, change it here and in both blocks of that file
                # too. A stale value here does not fail -- it scales every
                # validation translation by the ratio of the two focal
                # lengths, and the report still looks self-consistent.
                'camera_hfov': 1.5184,
                'publish_debug_image': True,
            }],
        )

        validator = Node(
            package='mycobot_calibration',
            executable='validate_calibration',
            name='validate_calibration',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'result_file': value('result_file'),
                'object_xyz': [float(value('object_x')),
                               float(value('object_y')),
                               object_z_base],
                'object_rpy_deg': [roll, 0.0, 0.0],
                'reach': value('reach').lower() == 'true',
                'standoff': float(value('standoff')),
            }],
        )

        actions += [detector, validator]

        if value('show_detections').lower() == 'true':
            actions.append(Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2_validation',
                output='log',
                arguments=['-d', os.path.join(
                    get_package_share_directory('mycobot_calibration'),
                    'rviz', 'validate_calibration.rviz')],
                parameters=[{'use_sim_time': True}],
            ))

        # The validator is one-shot; without this the detector would hold the
        # launch open and the run would look hung after the report is printed.
        actions.append(RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=validator,
                on_exit=EmitEvent(event=Shutdown(reason='validation finished')),
            ),
        ))
        return actions

    return LaunchDescription(declare_args + [OpaqueFunction(function=launch_setup)])
