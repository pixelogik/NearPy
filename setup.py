#!/usr/bin/env python
import sys
from setuptools import setup

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner>=2.0,<3.0'] if needs_pytest else []

setup(
    name='NearPy',
    version='0.2.2',
    author='Ole Krause-Sparmann',
    author_email='ole@pixelogik.de',
    packages=[
        'nearpy',
        'nearpy.distances',
        'nearpy.experiments',
        'nearpy.filters',
        'nearpy.hashes',
        'nearpy.hashes.permutation',
        'nearpy.storage',
        'nearpy.utils'
    ],
    url='http://pypi.python.org/pypi/NearPy/',
    license='LICENSE.txt',
    description='Framework for fast approximated nearest neighbour search.',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy",
        "scipy",
        "bitarray",
        "future",
    ],
    setup_requires=setup_requires,
    tests_require=[
        "pytest",
        "redis",
        "mockredispy",
    ]
)
