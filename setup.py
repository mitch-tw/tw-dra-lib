#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


with open('requirements/test.txt') as f:
    test_requirements = f.read().splitlines()


setup(
    author='Mitchell Lisle',
    author_email='mitchell.lisle@thoughtworks.com',
    description='A helper library for DRA training',
    install_requires=requirements,
    include_package_data=True,
    keywords='dra',
    name='dra',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    test_suite='tests',
    license='Not open source',
    tests_require=test_requirements,
    url='https://github.com/mitch-tw/dra',
    version='0.13.0',
    zip_safe=False,
)
