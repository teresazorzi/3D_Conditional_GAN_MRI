# setup.py
from setuptools import setup, find_packages

setup(
    name='mrisyngan',
    version='0.1.0',
    description='Conditional WGAN-GP for synthetic 3D MRI generation.',
    author='Teresa Zorzi', 
    packages=find_packages(),
    install_requires=[
        # Main dependencies will be read from requirements.txt
    ],
)