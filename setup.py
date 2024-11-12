from setuptools import setup, find_packages

setup(
    name='robot_controller',
    version='0.0.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
)