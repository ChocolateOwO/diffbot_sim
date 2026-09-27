import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, IncludeLaunchDescription,
                             RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('diffbot_sim')

    gui_arg = DeclareLaunchArgument('gui', default_value='true',
                                    description='Start gzclient (false = faster data collection)')
    out_arg = DeclareLaunchArgument('output_path', default_value='/mnt/e/งาน/271411/ocgm_part2/data.csv',
                                    description='CSV file for the recorded LiDAR data')

    live_arg = DeclareLaunchArgument('live_map', default_value='true',
                                     description='Open the live occupancy-grid-map window')
    py_arg = DeclareLaunchArgument('map_python', default_value=os.path.expanduser('~/ocgm_venv/bin/python'),
                                   description='Python with numpy + matplotlib (the system one has a broken matplotlib)')
    script_arg = DeclareLaunchArgument('map_script', default_value='/mnt/e/งาน/271411/ocgm_part2/ocgm_660610829.py',
                                       description='OGM script started with --live')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'sim.launch.py')),
        launch_arguments={'gui': LaunchConfiguration('gui')}.items())

    recorder = Node(package='diffbot_sim', executable='lidar_recorder', output='screen',
                    parameters=[{'output_path': LaunchConfiguration('output_path')}])
    driver = Node(package='diffbot_sim', executable='waypoint_driver', output='screen')

    # Live map window. Started detached (setsid) so it is NOT killed when the launch shuts
    # down at the end of the route: it keeps showing the final map and saves ocgm csv/png.
    # Safe to start at the same time as the recorder: live() just polls data.csv and shows
    # "waiting for data.csv ..." until rows arrive, and resets the map if the file shrinks
    # (a fresh run truncating an old data.csv), so there is no need for an extra delay here.
    live_map = ExecuteProcess(
        cmd=['bash', '-c', ['setsid "', LaunchConfiguration('map_python'), '" "', LaunchConfiguration('map_script'),
                            '" --live --data "', LaunchConfiguration('output_path'), '" >/dev/null 2>&1 &']],
        condition=IfCondition(LaunchConfiguration('live_map')))

    # wait until gzserver has loaded the world and the robot is spawned
    start = TimerAction(period=20.0, actions=[recorder, driver, live_map])

    # route finished -> stop everything (recorder closes data.csv on shutdown)
    stop = RegisterEventHandler(OnProcessExit(
        target_action=driver, on_exit=[EmitEvent(event=Shutdown(reason='route finished'))]))

    return LaunchDescription([gui_arg, out_arg, live_arg, py_arg, script_arg, sim, start, stop])
