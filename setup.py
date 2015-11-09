#!/usr/bin/env python
from setuptools import setup, Command


class RunTests(Command):

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        import sys

        errno = subprocess.call(['py.test'])
        raise SystemExit(errno)


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
    tests_require=[
        "pytest",
        "redis",
        "mockredispy",
    ],
    cmdclass = {
        'test': RunTests
    },
)
