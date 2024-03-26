import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'lidar_to_global'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
        glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), 
        glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Saimai Lau',
    author_email='s7lau@ucsd.edu',
    maintainer='MURO LAB',
    maintainer_email='murolab101@gmail.com',
    description='Transform lidar data to global frame.',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scanNTran = lidar_to_global.lidar2MM:main',
            'plotPoints = lidar_to_global.plot_path:main'
        ],
    },
)
