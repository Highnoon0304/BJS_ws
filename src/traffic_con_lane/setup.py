from setuptools import setup

package_name = 'traffic_con_lane'

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
    author='Lee',
    author_email='lcm20010304@gmail.com',
    description='...',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'lane_detection_node = traffic_con_lane.lane_detection_node:main'
        ],
    },
)

