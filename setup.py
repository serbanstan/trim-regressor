import os
from setuptools import setup, find_packages

PACKAGE_NAME = 'trim-regressor'


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    # module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    module_path = '__init__.py'
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert False, "'{0}' not found in '{1}'".format(key, module_path)


setup(
    name=PACKAGE_NAME,
    version=read_package_variable('__version__'),
    description='A de-confounding primitive using TRIM',
    license="AGPL-3.0",
    author=read_package_variable('__author__'),
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'd3m',
        'scikit-learn',
    ],
    url='https://github.com/serbanstan/trim-regressor',
    entry_points = {
        'd3m.primitives': [
            'regression.trim_regressor.TrimRegressor = trim_regressor:TrimRegressor',
        ],
    },
)
