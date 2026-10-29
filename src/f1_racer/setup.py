from glob import glob

from setuptools import find_packages, setup

package_name = 'f1_racer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/racelines', glob('racelines/*.csv')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='ameen',
    maintainer_email='ameensadiya12345@gmail.com',
    description='CIE-KFUPM F1TENTH competition controller (gap follower + pure pursuit)',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'controller = f1_racer.controller_node:main',
        ],
    },
)
