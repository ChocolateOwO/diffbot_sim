import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'diffbot_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chaem',
    maintainer_email='chaem@todo.todo',
    description='4-wheel differential drive robot with 2D LiDAR in a 30x30m multi-room Gazebo world',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'move_robot = diffbot_sim.move_robot:main',
            'wasd_teleop = diffbot_sim.wasd_teleop:main',
            'lidar_recorder = diffbot_sim.lidar_recorder:main',
            'waypoint_driver = diffbot_sim.waypoint_driver:main',
        ],
    },
)
