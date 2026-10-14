"""Spawn the robot at the random start pose of an RRT* plan and drive it to the goal.

  ros2 launch diffbot_sim rrt_nav.launch.py plan_file:=/path/to/plan.json
"""
import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, IncludeLaunchDescription,
                             OpaqueFunction, RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEFAULT_DIR = '/mnt/e/งาน/271411/rrt_star_part3'


def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('diffbot_sim')
    plan_file = LaunchConfiguration('plan_file').perform(context)
    with open(plan_file) as f:
        sx, sy, syaw = json.load(f)['start']
    # forget the result of the previous run so the live window starts clean
    traj = LaunchConfiguration('traj_out').perform(context)
    scan = LaunchConfiguration('scan_csv').perform(context)
    for old in (traj, traj.rsplit('.', 1)[0] + '_result.json', scan):
        if os.path.exists(old):
            os.remove(old)

    # Gazebo + robot spawned exactly on the (random) start pose of the plan
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'sim.launch.py')),
        launch_arguments={
            'gui': LaunchConfiguration('gui'),
            'spawn_x': f'{sx:.4f}', 'spawn_y': f'{sy:.4f}', 'spawn_yaw': f'{syaw:.4f}',
        }.items())

    follower = ExecuteProcess(
        cmd=['python3', LaunchConfiguration('follower_script'), 'drive',
             '--plan', plan_file,
             '--traj-out', LaunchConfiguration('traj_out'),
             '--speed', LaunchConfiguration('speed')],
        output='screen')

    # LiDAR scans + pose written while the robot drives; the live window builds the map from them
    recorder = Node(package='diffbot_sim', executable='lidar_recorder', output='screen',
                    parameters=[{'output_path': scan, 'min_distance': 0.2, 'min_rotation_deg': 4.0}])

    # live window: the occupancy grid map is updated from the LiDAR while the robot drives, next to
    # the RRT* path. Started detached (setsid) so it is not killed when the launch shuts down: the
    # final map stays on the screen and is saved.
    live_plot = ExecuteProcess(
        cmd=['bash', '-c', ['setsid "', LaunchConfiguration('plot_python'), '" "',
                            LaunchConfiguration('plot_script'), '" --live --plan "', plan_file,
                            '" --traj "', LaunchConfiguration('traj_out'),
                            '" --scan "', scan, '" >/dev/null 2>&1 &']],
        condition=IfCondition(LaunchConfiguration('live_plot')))

    # give gzserver time to load the world and spawn the robot (the follower also waits for /odom)
    start = TimerAction(period=20.0, actions=[recorder, follower, live_plot])
    # By default everything stays open when the robot has arrived (the follower prints SUCCESS),
    # press Ctrl+C in this terminal to close it. close_when_done:=true closes Gazebo automatically.
    stop = RegisterEventHandler(
        OnProcessExit(target_action=follower, on_exit=[EmitEvent(event=Shutdown(reason='path finished'))]),
        condition=IfCondition(LaunchConfiguration('close_when_done')))
    return [sim, start, stop]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('plan_file', default_value=os.path.join(DEFAULT_DIR, 'plan.json'),
                              description='plan.json written by:  rrt_coppeliasim_660610829.py plan'),
        DeclareLaunchArgument('follower_script',
                              default_value=os.path.join(DEFAULT_DIR, 'rrt_coppeliasim_660610829.py'),
                              description='rrt_coppeliasim_660610829.py (started with the "drive" command)'),
        DeclareLaunchArgument('traj_out', default_value=os.path.join(DEFAULT_DIR, 'trajectory.csv'),
                              description='where the driven trajectory is saved'),
        DeclareLaunchArgument('scan_csv', default_value=os.path.join(DEFAULT_DIR, 'drive_scan.csv'),
                              description='LiDAR rows recorded while driving (feeds the live map)'),
        DeclareLaunchArgument('speed', default_value='0.7', description='max forward speed [m/s]'),
        DeclareLaunchArgument('gui', default_value='true', description='open the Gazebo window'),
        DeclareLaunchArgument('close_when_done', default_value='false',
                              description='close Gazebo automatically when the robot has arrived'),
        DeclareLaunchArgument('live_plot', default_value='true',
                              description='open the live map window (map updated from the LiDAR while driving)'),
        DeclareLaunchArgument('plot_python', default_value=os.path.expanduser('~/ocgm_venv/bin/python'),
                              description='python with numpy + matplotlib'),
        DeclareLaunchArgument('plot_script',
                              default_value=os.path.join(DEFAULT_DIR, 'plot_result_660610829.py'),
                              description='live window program'),
        OpaqueFunction(function=launch_setup),
    ])
