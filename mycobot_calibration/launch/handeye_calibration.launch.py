#!/usr/bin/env python3
"""Run the eye-to-hand calibration sweep against an already-running simulation.

Expects the simulation to be up with the camera and target enabled:

  ros2 launch mycobot_gazebo mycobot_combined.launch.py \\
      use_camera:=true use_aruco_marker:=true use_gripper:=false \\
      use_gz_gui:=false world_file:=calibration.world

then:

  ros2 launch mycobot_calibration handeye_calibration.launch.py

The detector runs continuously; the collector drives the sweep and exits when
the dataset is written. Solving is a separate step so a dataset can be re-run
through different methods without re-collecting:

  ros2 run mycobot_calibration handeye_solver --dataset /tmp/handeye_dataset.json
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('mycobot_calibration')
    default_config = os.path.join(pkg_share, 'config', 'handeye_calibration.yaml')

    config_file = LaunchConfiguration('config_file')
    dataset_file = LaunchConfiguration('dataset_file')
    run_collector = LaunchConfiguration('run_collector')

    # True in simulation, where Gazebo owns the clock. It must be FALSE on real
    # hardware: with no /clock publisher a node started with use_sim_time stays
    # frozen at time zero, so every TF lookup, settle_time and sample timeout
    # waits forever and the sweep looks hung rather than failing.
    sim_time = ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)

    declare_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Take time from /clock. Leave true for Gazebo; set false '
                    'when running against the physical arm.')

    declare_config = DeclareLaunchArgument(
        'config_file', default_value=default_config,
        description='Parameter file for the detector and collector.')

    # A second parameter file, layered over config_file. ROS 2 merges parameter
    # files in order, so this lets the physical rig change the handful of values
    # that differ -- real intrinsics, the colour FOV and optical frame, the
    # printed marker size -- without duplicating a long annotated config and
    # letting the two drift apart. Defaults to an empty file, so simulation is
    # unaffected.
    overrides_file = LaunchConfiguration('overrides_file')

    declare_overrides = DeclareLaunchArgument(
        'overrides_file',
        default_value=os.path.join(pkg_share, 'config', 'no_overrides.yaml'),
        description='Parameter file layered over config_file. Pass '
                    'config/hardware_overrides.yaml for the physical rig.')

    declare_dataset = DeclareLaunchArgument(
        'dataset_file', default_value='/tmp/handeye_dataset.json',
        description='Where the collector writes the observation pairs.')

    declare_run_collector = DeclareLaunchArgument(
        'run_collector', default_value='true',
        description='Set false to run only the detector, e.g. to check marker '
                    'visibility in RViz before committing to a sweep.')

    declare_show_detections = DeclareLaunchArgument(
        'show_detections', default_value='false',
        description='Open an RViz window showing the camera view with the '
                    'detections and the accumulated coverage drawn on it. Off '
                    'by default because the combined sim launch usually has '
                    'its own RViz already up; the same view is available there '
                    'by adding an Image display on '
                    '/aruco_detector/debug_image.')

    detections_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_detections',
        output='log',
        condition=IfCondition(LaunchConfiguration('show_detections')),
        arguments=['-d', os.path.join(pkg_share, 'rviz',
                                      'handeye_calibration.rviz')],
        parameters=[{'use_sim_time': sim_time}],
    )

    declare_show_coverage = DeclareLaunchArgument(
        'show_coverage', default_value='false',
        description='Draw the accumulated detection footprint on the debug '
                    'image, with the 4x3 cell grid and a running count. Use '
                    'for a run meant to demonstrate that the target reached '
                    'the whole frame; the coverage figures are measured and '
                    'recorded by the collector either way.')

    detector = Node(
        package='mycobot_calibration',
        executable='aruco_detector',
        name='aruco_detector',
        output='screen',
        parameters=[
            config_file,
            overrides_file,
            {'use_sim_time': sim_time,
             'draw_coverage_trail': ParameterValue(
                 LaunchConfiguration('show_coverage'), value_type=bool)},
        ],
    )

    collector = Node(
        package='mycobot_calibration',
        executable='pose_collector',
        name='handeye_pose_collector',
        output='screen',
        condition=IfCondition(run_collector),
        parameters=[
            config_file,
            overrides_file,
            {'use_sim_time': sim_time, 'output_file': dataset_file},
        ],
    )

    # The collector is a one-shot script. Without this the detector would keep
    # the launch alive after the sweep finished, and the run would look hung.
    shutdown_when_done = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=collector,
            on_exit=EmitEvent(event=Shutdown(reason='collection finished')),
        ),
    )

    return LaunchDescription([
        declare_sim_time,
        declare_config,
        declare_overrides,
        declare_dataset,
        declare_run_collector,
        declare_show_detections,
        declare_show_coverage,
        detector,
        detections_rviz,
        collector,
        shutdown_when_done,
    ])
