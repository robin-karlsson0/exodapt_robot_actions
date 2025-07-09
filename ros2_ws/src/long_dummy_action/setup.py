import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'long_dummy_action'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['tests']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=[
        'setuptools',
        'exodapt_robot_interfaces',
    ],
    zip_safe=True,
    maintainer='Robin Karlsson',
    maintainer_email='robin@exodapt.ai',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'long_dummy_action = ' + package_name + '.long_dummy_action:main',
        ],
    },
)
