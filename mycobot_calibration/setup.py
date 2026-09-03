import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'mycobot_calibration'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'scripts'),
         glob('scripts/*.py') + glob('scripts/*.sh')),
        # Also into lib/, which is the only place `ros2 run` looks. Shell
        # scripts cannot be console_scripts entry points, so without this the
        # demo is only reachable by full path.
        (os.path.join('lib', package_name), ['scripts/run_demo.sh']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yahya',
    maintainer_email='m.yahya@hotmail.it',
    description='Eye-to-hand calibration for the myCobot 280.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_detector = mycobot_calibration.aruco_detector_node:main',
            'pose_collector = mycobot_calibration.pose_collector_node:main',
            'handeye_solver = mycobot_calibration.handeye_solver:main',
            'calibration_publisher = '
            'mycobot_calibration.calibration_publisher_node:main',
            'validate_calibration = '
            'mycobot_calibration.validate_calibration_node:main',
            'present_target = '
            'mycobot_calibration.present_target_node:main',
            'jetbot_camera_bridge = '
            'mycobot_calibration.jetbot_camera_bridge:main',
        ],
    },
)
