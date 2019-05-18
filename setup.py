#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'bokeh>=1.0.4',
    'dask[complete]>=1.1.5',
    'numpy>=1.14',
    'pyarrow==0.12.1',
    'pyyaml>=4.2b1',
    'xarray>=0.12.0',
    'zarr>=2.2.0',
    's3fs>=0.2.1',
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Sarah Bird",
    author_email='fx-data-dev@mozilla.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    description="Utilities to build the dye-score metric from OpenWPM javascript call data.",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='dye_score',
    name='dye_score',
    packages=find_packages(include=['dye_score']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mozilla/dye-score',
    version='0.9.0',
    zip_safe=False,
)
