#!/usr/bin/env python3
"""
Combined launch file for myCobot 280: Gazebo simulation + MoveIt2 + RViz.

Launches the full pipeline in a single command:
  - Robot State Publisher (always)
  - Ignition Gazebo + ROS bridges + robot spawner (when use_sim:=true)
  - ROS 2 controllers (when use_sim:=true; hardware stub placeholder for future)
  - MoveIt2 move_group + RViz (always)

Usage examples:
  # Full simulation (default)
  ros2 launch mycobot_gazebo mycobot_combined.launch.py

  # Planning/visualization only, no Gazebo (hardware stub)
  ros2 launch mycobot_gazebo mycobot_combined.launch.py use_sim:=false

  # Headless simulation (no RViz)
  ros2 launch mycobot_gazebo mycobot_combined.launch.py use_rviz:=false

:author: Combined launcher for myCobot ROS2 Humble + Ignition Fortress
:date: March 2026
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # ---------------------------------------------------------------------------
    # Package names
    # ---------------------------------------------------------------------------
    pkg_gazebo = 'mycobot_gazebo'
    pkg_description = 'mycobot_description'
    pkg_moveit = 'mycobot_moveit_config'

    # ---------------------------------------------------------------------------
    # Resolve package share paths at parse time
    # ---------------------------------------------------------------------------
    pkg_share_gazebo = FindPackageShare(pkg_gazebo).find(pkg_gazebo)
    pkg_share_description = FindPackageShare(pkg_description).find(pkg_description)
    pkg_ros_ign_gazebo = FindPackageShare('ros_ign_gazebo').find('ros_ign_gazebo')

    gazebo_models_path = os.path.join(pkg_share_gazebo, 'models')
    ign_bridge_config = os.path.join(pkg_share_gazebo, 'config', 'ros_ign_bridge.yaml')

    # ---------------------------------------------------------------------------
    # Launch arguments
    # ---------------------------------------------------------------------------
    use_sim = LaunchConfiguration('use_sim')
    use_rviz = LaunchConfiguration('use_rviz')
    use_gripper = LaunchConfiguration('use_gripper')
    use_camera = LaunchConfiguration('use_camera')
    robot_name = LaunchConfiguration('robot_name')
    world_file = LaunchConfiguration('world_file')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    z = LaunchConfiguration('z')
    roll = LaunchConfiguration('roll')
    pitch = LaunchConfiguration('pitch')
    yaw = LaunchConfiguration('yaw')

    world_path = PathJoinSubstitution([pkg_share_gazebo, 'worlds', world_file])

    declare_use_sim = DeclareLaunchArgument(
        name='use_sim',
        default_value='true',
        description='Launch Gazebo simulation. Set false for hardware (stub, not yet implemented).')

    declare_use_rviz = DeclareLaunchArgument(
        name='use_rviz',
        default_value='true',
        description='Launch RViz with MoveIt2.')

    declare_use_gripper = DeclareLaunchArgument(
        name='use_gripper',
        default_value='false',
        description='Include gripper in robot model.')

    declare_use_camera = DeclareLaunchArgument(
        name='use_camera',
        default_value='false',
        description='Enable RGBD camera in simulation.')

    declare_robot_name = DeclareLaunchArgument(
        name='robot_name',
        default_value='mycobot_280',
        description='Robot model name.')

    declare_world_file = DeclareLaunchArgument(
        name='world_file',
        default_value='empty.world',
        description='Gazebo world file name (e.g. empty.world).')

    declare_x = DeclareLaunchArgument(name='x', default_value='0.0')
    declare_y = DeclareLaunchArgument(name='y', default_value='0.0')
    declare_z = DeclareLaunchArgument(name='z', default_value='0.05')
    declare_roll = DeclareLaunchArgument(name='roll', default_value='0.0')
    declare_pitch = DeclareLaunchArgument(name='pitch', default_value='0.0')
    declare_yaw = DeclareLaunchArgument(name='yaw', default_value='0.0')

    # ---------------------------------------------------------------------------
    # Always-on: Robot State Publisher
    # ---------------------------------------------------------------------------
    rsp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share_description, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={
            'jsp_gui': 'false',
            'use_camera': use_camera,
            'use_gazebo': use_sim,      # tells RSP whether to expect sim clock
            'use_gripper': use_gripper,
            'use_rviz': 'false',        # RViz handled below in OpaqueFunction
            'use_sim_time': use_sim,
        }.items(),
    )

    # ---------------------------------------------------------------------------
    # Always-on: MoveIt2 move_group + RViz  (inlined OpaqueFunction)
    #
    # NOTE: We inline the MoveIt config and RViz node creation directly here
    # rather than using IncludeLaunchDescription for move_group.launch.py.
    # This is necessary because nesting an OpaqueFunction inside
    # IncludeLaunchDescription silently prevents the RViz node from being
    # launched (the IfCondition on use_rviz does not fire in the sub-context).
    #
    # NOTE: The RViz exit handler (RegisterEventHandler) is intentionally NOT
    # returned from this OpaqueFunction. Returning a RegisterEventHandler from
    # an OpaqueFunction in ROS2 Humble silently prevents all subsequent
    # LaunchDescription actions (RSP, Gazebo, etc.) from being processed.
    # The exit handler is set up by a separate OpaqueFunction (rviz_exit_setup)
    # added at the end of the LaunchDescription.
    # ---------------------------------------------------------------------------
    _rviz_node_holder = []  # shared mutable container between the two closures

    def configure_moveit(context):
        robot_name_str = LaunchConfiguration('robot_name').perform(context)
        use_sim_str = LaunchConfiguration('use_sim').perform(context)
        use_rviz_str = LaunchConfiguration('use_rviz').perform(context)
        use_sim_bool = use_sim_str.lower() == 'true'

        pkg_share_moveit = FindPackageShare(pkg_moveit).find(pkg_moveit)
        config_path = os.path.join(pkg_share_moveit, 'config', robot_name_str)

        moveit_config = (
            MoveItConfigsBuilder(robot_name_str, package_name=pkg_moveit)
            .trajectory_execution(
                file_path=os.path.join(config_path, 'moveit_controllers.yaml'))
            .robot_description_semantic(
                file_path=os.path.join(config_path, f'{robot_name_str}.srdf'))
            .joint_limits(
                file_path=os.path.join(config_path, 'joint_limits.yaml'))
            .robot_description_kinematics(
                file_path=os.path.join(config_path, 'kinematics.yaml'))
            .planning_pipelines(
                pipelines=['ompl', 'pilz_industrial_motion_planner', 'stomp'],
                default_planning_pipeline='ompl',
            )
            .planning_scene_monitor(
                publish_robot_description=False,
                publish_robot_description_semantic=True,
                publish_planning_scene=True,
            )
            .pilz_cartesian_limits(
                file_path=os.path.join(config_path, 'pilz_cartesian_limits.yaml'))
            .to_moveit_configs()
        )

        move_group_node = Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                moveit_config.to_dict(),
                {'use_sim_time': use_sim_bool},
                {'start_state': {
                    'content': os.path.join(config_path, 'initial_positions.yaml')}},
            ],
        )

        actions = [move_group_node]

        if use_rviz_str.lower() == 'true':
            rviz_node = Node(
                package='rviz2',
                executable='rviz2',
                output='screen',
                arguments=[
                    '-d', os.path.join(pkg_share_moveit, 'rviz', 'move_group.rviz'),
                ],
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.planning_pipelines,
                    moveit_config.robot_description_kinematics,
                    moveit_config.joint_limits,
                    {'use_sim_time': use_sim_bool},
                ],
            )
            _rviz_node_holder.append(rviz_node)
            actions.append(rviz_node)

        return actions

    def setup_rviz_exit_handler(context):
        if not _rviz_node_holder:
            return []
        return [RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=_rviz_node_holder[0],
                on_exit=EmitEvent(event=Shutdown(reason='rviz exited')),
            ),
        )]

    moveit_and_rviz = OpaqueFunction(function=configure_moveit)
    rviz_exit_setup = OpaqueFunction(function=setup_rviz_exit_handler)

    # ---------------------------------------------------------------------------
    # Simulation-only block (use_sim:=true)
    # ---------------------------------------------------------------------------

    set_ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            gazebo_models_path, ':',
            pkg_share_description, ':',
            os.environ.get('IGN_GAZEBO_RESOURCE_PATH', ''),
        ],
        condition=IfCondition(use_sim),
    )

    # ROS 2 controllers (requires sim hardware interface)
    load_controllers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                FindPackageShare(pkg_moveit).find(pkg_moveit),
                'launch', 'load_ros2_controllers.launch.py',
            )
        ),
        launch_arguments={
            'use_gripper': use_gripper,
            'use_sim_time': use_sim,
        }.items(),
        condition=IfCondition(use_sim),
    )

    # Ignition Gazebo server
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_ign_gazebo, 'launch', 'ign_gazebo.launch.py')
        ),
        launch_arguments=[('ign_args', [' -r -v 4 ', world_path])],
        condition=IfCondition(use_sim),
    )

    # ROS <-> Ignition topic bridge
    ign_bridge = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': ign_bridge_config}],
        output='screen',
        condition=IfCondition(use_sim),
    )

    # Camera image bridge (only when camera is enabled)
    ign_image_bridge = Node(
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
        condition=IfCondition(use_sim),
    )

    # Robot spawner
    robot_spawner = Node(
        package='ros_ign_gazebo',
        executable='create',
        output='screen',
        arguments=[
            '-topic', '/robot_description',
            '-name', robot_name,
            '-allow_renaming', 'true',
            '-x', x, '-y', y, '-z', z,
            '-R', roll, '-P', pitch, '-Y', yaw,
        ],
        condition=IfCondition(use_sim),
    )

    # ---------------------------------------------------------------------------
    # Hardware stub (use_sim:=false)
    # ---------------------------------------------------------------------------
    # TODO: Replace this block with the hardware driver launch when available.
    #       e.g. IncludeLaunchDescription for mycobot_hardware or similar.
    hardware_stub_msg = LogInfo(
        msg='[mycobot_combined] use_sim:=false — '
            'hardware interface not yet implemented. '
            'Running move_group + RViz in planning-only mode.',
        condition=UnlessCondition(use_sim),
    )

    # ---------------------------------------------------------------------------
    # Assemble launch description
    # ---------------------------------------------------------------------------
    ld = LaunchDescription()

    # Arguments
    ld.add_action(declare_use_sim)
    ld.add_action(declare_use_rviz)
    ld.add_action(declare_use_gripper)
    ld.add_action(declare_use_camera)
    ld.add_action(declare_robot_name)
    ld.add_action(declare_world_file)
    ld.add_action(declare_x)
    ld.add_action(declare_y)
    ld.add_action(declare_z)
    ld.add_action(declare_roll)
    ld.add_action(declare_pitch)
    ld.add_action(declare_yaw)

    # Always-on
    # NOTE: moveit_and_rviz MUST come before rsp_launch. ROS2 launch shares
    # argument context across IncludeLaunchDescription calls: rsp_launch passes
    # use_rviz='false' to RSP which overwrites the shared LaunchConfiguration,
    # so the OpaqueFunction must read use_rviz before that happens.
    ld.add_action(moveit_and_rviz)
    ld.add_action(rsp_launch)

    # Simulation path
    ld.add_action(set_ign_resource_path)
    ld.add_action(load_controllers_launch)
    ld.add_action(gazebo_server)
    ld.add_action(ign_bridge)
    ld.add_action(ign_image_bridge)
    ld.add_action(robot_spawner)

    # Hardware stub
    ld.add_action(hardware_stub_msg)

    # RViz exit handler — registered after Gazebo so it doesn't block sim launch
    ld.add_action(rviz_exit_setup)

    # ---------------------------------------------------------------------------
    # Cleanup: kill stale bridge processes on shutdown so they don't block
    # subsequent launches.  Uses a plain Python callable rather than
    # ExecuteProcess actions because OnShutdown can fire multiple times during
    # the shutdown cascade and ExecuteProcess objects cannot be re-executed.
    # ---------------------------------------------------------------------------
    def kill_bridges(event, context):
        import subprocess
        subprocess.run(['pkill', '-9', '-f', 'parameter_bridge'], capture_output=True)
        subprocess.run(['pkill', '-9', '-f', 'image_bridge'], capture_output=True)
        subprocess.run(['pkill', '-9', '-f', 'ign_gazebo'], capture_output=True)
        return []

    cleanup_bridges = RegisterEventHandler(
        event_handler=OnShutdown(on_shutdown=kill_bridges)
    )
    ld.add_action(cleanup_bridges)

    return ld
