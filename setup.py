from setuptools import setup, find_packages
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6

setup(
    name='RlAlgoZoo',
    packages=[package for package in find_packages()
              if package.startswith('rlalgo')],
    install_requires=[
        'gym>=0.15.4, <0.16.0',
        'scipy',
        'tqdm',
        'joblib',
        'cloudpickle',
        'click',
        'opencv-python',
        'torch>=1.4',
        'pyyaml',
    ],
    description="Convenient tools for me to learn RL.",
    author="Jinhua Zhu",
)