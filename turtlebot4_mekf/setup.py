import os
from glob import glob
from setuptools import setup

package_name = 'turtlebot4_mekf'

setup(
    name=package_name,
    version='0.1',
    packages=[package_name],
    # packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
        glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), 
        glob('config/*.yaml')),
        # ('lib/' + package_name, [package_name+'/util.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Brandon Bao',
    author_email='bjbao@ucsd.edu',
    maintainer='MURO LAB',
    maintainer_email='murolab101@gmail.com',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: MIT License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Beacon localization for turtlebot4',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'Turtlebot4MEKF = turtlebot4_mekf.Turtlebot4MEKF:main',
            'fake_beacon = turtlebot4_mekf.fake_beacon:main',
            'fake_imu = turtlebot4_mekf.fake_imu:main',
            'plot_path = turtlebot4_mekf.plot_path:main',
            'Bernoulli_EKF = turtlebot4_mekf.bernoulli_EKF:main',
            'Turtlebot2EKF = turtlebot4_mekf.Turtlebot2EKF:main',
            'Turtleb4EKF = turtlebot4_mekf.Turtlebot4EKF_V2:main',
        ],
    },
)

# cmake_minimum_required(VERSION 3.8)
# project(turtlebot4_mekf)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# # find dependencies
# find_package(ament_cmake 		REQUIRED)
# find_package(rosidl_default_generators 	REQUIRED)
# find_package(std_msgs 			REQUIRED)
# find_package(geometry_msgs  REQUIRED)
# find_package(sensor_msgs REQUIRED)
# # uncomment the following section in order to fill in
# # further dependencies manually.
# # find_package(<dependency> REQUIRED)

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # uncomment the line when a copyright and license is not present in all source files
#   #set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # uncomment the line when this package is not in a git repo
#   #set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

# ament_package()
