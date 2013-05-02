#!/usr/bin/env python

from distutils.core import setup

setup(
    name='NearPy',
    version='0.1',
    author='Ole Krause-Sparmann',
    author_email='ole@pixelogik.de',
    packages=['nearpy'],
    url='http://pypi.python.org/pypi/NearPy/',
    license='LICENSE.txt',
    description='Framework for fast approximated nearest neighbour search.',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy",
        "scipy",
        "redis"
    ],
)