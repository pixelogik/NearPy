#!/usr/bin/env python
import sys
from setuptools import setup, find_packages

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner>=2.0,<3.0'] if needs_pytest else []

setup(
    name='NearPy',
    version='1.0.0',
    author='Ole Krause-Sparmann',
    author_email='ole@pixelogik.de',
    packages=find_packages(exclude=["tests.*"]),
    url='https://github.com/pixelogik/NearPy',
    license='LICENSE.txt',
    description='Framework for fast approximated nearest neighbour search.',
    keywords='nearpy approximate nearest neighbour',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy==1.24.0",
        "scipy==1.9.3",
        "bitarray==2.6.1",
        "future==0.18.2",
    ],
    setup_requires=setup_requires,
    tests_require=[
        "pytest==7.2.0",
        "redis==4.4.0",
        "mockredispy==2.9.3",
        "mongomock==4.1.2",
    ]
)
