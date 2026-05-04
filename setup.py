from setuptools import setup

setup(
    name="moirain",
    packages=[
        'data',
        'models',
        'experiments'
    ],
    package_dir={
        'data': './data',
        'models': './models',
        'experiments': './experiments'
    },
)