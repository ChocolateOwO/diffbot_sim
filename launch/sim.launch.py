import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, DeclareLaunchArgument,
                             TimerAction, SetEnvironmentVariable, ExecuteProcess)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('diffbot_sim')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    world_path = os.path.join(pkg_share, 'worlds', 'office_world.world')
    xacro_path = os.path.join(pkg_share, 'urdf', 'diffbot.urdf.xacro')

    world_arg = DeclareLaunchArgument(
        'world', default_value=world_path,
        description='Full path to the world file to load')
    gui_arg = DeclareLaunchArgument(
        'gui', default_value='true',
        description='Start gzclient (3D viewer)')

    # WSLg's D3D12-translated GL driver races/crashes gzclient's Ogre camera
    # if the client starts at the same time as the server; software GL plus
    # a short startup delay (below) avoids it.
    force_software_gl = SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1')

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': LaunchConfiguration('world')}.items())

    # Launched as a raw ExecuteProcess (not gazebo_ros's gzclient.launch.py):
    # going through that wrapper's ExecuteProcess(additional_env=...) reliably
    # crashed gzclient on an Ogre Camera null-pointer assertion under WSLg,
    # even with software GL and a startup delay; a plain subprocess does not.
    gzclient = ExecuteProcess(
        cmd=['gzclient', '--gui-client-plugin=libgazebo_ros_eol_gui.so'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('gui')))

    # this world has 140+ models (walls/furniture); gzserver needs real time
    # to finish loading them before gzclient can attach its camera, otherwise
    # gzclient crashes on an empty scene (Ogre Camera assertion)
    delayed_gzclient = TimerAction(period=15.0, actions=[gzclient])

    robot_description = Command(['xacro ', xacro_path])

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}])

    # base_z = wheel_radius + ground_clearance + body_height/2 = 0.15+0.02+0.20 = 0.37
    # spawn just inside the south entrance door (gap at x in [-0.8, 0.8], y=-15),
    # facing north (+y, yaw=90deg) into the building
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'diffbot',
                   '-x', '0.0', '-y', '-13.8', '-z', '0.37',
                   '-Y', '1.5708'],
        output='screen')

    return LaunchDescription([
        world_arg,
        gui_arg,
        force_software_gl,
        gzserver,
        delayed_gzclient,
        robot_state_publisher,
        spawn_entity,
    ])
