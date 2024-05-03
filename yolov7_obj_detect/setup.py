from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'yolov7_obj_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name,'launch'), glob(os.path.join('launch','*launch.[pxy][yma]'))),
        (os.path.join('share',package_name,'weights'), glob(os.path.join('weights/*.pt'))),
        ('lib/' + package_name + '/models/',['models/experimental.py', 'models/common.py',
                                             'models/yolo.py']),
        ('lib/' + package_name + '/utils/',['utils/general.py', 'utils/torch_utils.py',
                                             'utils/plots.py', 'utils/datasets.py',
                                             'utils/google_utils.py', 'utils/activations.py',
                                             'utils/add_nms.py', 'utils/autoanchor.py',
                                             'utils/loss.py', 'utils/metrics.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anderson',
    maintainer_email='anderson@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection = yolov7_obj_detect.object_detection:main'
        ],
    },
)
