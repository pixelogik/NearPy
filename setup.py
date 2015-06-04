#!/usr/bin/env python

from setuptools import setup

setup(
    name='NearPy',
    version='0.2.1',
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
        "redis",
        "bitarray"
    ],
)
