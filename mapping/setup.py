import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mapping'

setup(
    name=package_name,
    version='1.0.0',
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
    description='Mapping with Lidar on Turtlebot, localization data required.',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'Mapping = mapping.Mapping:main',
            'multiMapping = mapping.multiMapping:main',
            'plotData = mapping.getData:main',
            'fullMultiMapping = mapping.fullMultiMapping:main',
            'fullMultiMappingStrCal = mapping.fullMultiMappingStrCal:main',
            'fullMultiMappingImg = mapping.fullMultiMappingImg:main',
        ],
    },
)
