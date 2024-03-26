from setuptools import setup

package_name = 'followInCirclePkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='saimai',
    maintainer_email='louis33338888@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'followInCircle = followInCirclePkg.followInCircle:main',
            'follower = followInCirclePkg.follower:main',
            'follower2 = followInCirclePkg.follower2:main',
            'justGo = followInCirclePkg.followInCircle:gogogo'
        ],
    },
)
