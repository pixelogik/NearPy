# -*- coding: utf-8 -*-

# Copyright (c) 2013 Ole Krause-Sparmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import unittest
import functools
import numpy

from nearpy import Engine
from nearpy.filters import NearestFilter
from nearpy.hashes import RandomBinaryProjectionTree

from mockredis import MockRedis as Redis

from nearpy.storage import MemoryStorage, RedisStorage


RandomBinaryProjectionTree = functools.partial(RandomBinaryProjectionTree,
                                               rand_seed=12)


class TestRandomBinaryProjectionTree(unittest.TestCase):

    def setUp(self):
        self.memory = MemoryStorage()
        self.redis_object = Redis()
        self.redis_storage = RedisStorage(self.redis_object)
        numpy.random.seed(16)

    def test_retrieval(self):
        # We want 12 projections, 20 results at least
        rbpt = RandomBinaryProjectionTree('testHash', 12, 20)

        # Create engine for 100 dimensional feature space, do not forget to set
        # nearest filter to 20, because default is 10
        self.engine = Engine(100, lshashes=[rbpt], vector_filters=[NearestFilter(20)])

        # First insert 200000 random vectors
        for k in range(200000):
            x = numpy.random.randn(100)
            x_data = 'data {}'.format(k)
            self.engine.store_vector(x, x_data)

        # Now do random queries and check result set size
        for k in range(10):
            x = numpy.random.randn(100)
            n = self.engine.neighbours(x)
            self.assertEqual(len(n), 20)

    def test_storage_memory(self):
        # We want 10 projections, 20 results at least
        rbpt = RandomBinaryProjectionTree('testHash', 10, 20)

        # Create engine for 100 dimensional feature space
        self.engine = Engine(100, lshashes=[rbpt], vector_filters=[NearestFilter(20)])

        # First insert 2000 random vectors
        for k in range(2000):
            x = numpy.random.randn(100)
            x_data = 'data'
            self.engine.store_vector(x, x_data)

        self.memory.store_hash_configuration(rbpt)

        rbpt2 = RandomBinaryProjectionTree(None, None, None)
        rbpt2.apply_config(self.memory.load_hash_configuration('testHash'))

        self.assertEqual(rbpt.dim, rbpt2.dim)
        self.assertEqual(rbpt.hash_name, rbpt2.hash_name)
        self.assertEqual(rbpt.projection_count, rbpt2.projection_count)

        for i in range(rbpt.normals.shape[0]):
            for j in range(rbpt.normals.shape[1]):
                self.assertEqual(rbpt.normals[i, j], rbpt2.normals[i, j])

        # Now do random queries and check result set size
        for k in range(10):
            x = numpy.random.randn(100)
            keys1 = rbpt.hash_vector(x, querying=True)
            keys2 = rbpt2.hash_vector(x, querying=True)
            self.assertEqual(len(keys1), len(keys2))
            for k in range(len(keys1)):
                self.assertEqual(keys1[k], keys2[k])

    def test_storage_redis(self):
        # We want 10 projections, 20 results at least
        rbpt = RandomBinaryProjectionTree('testHash', 10, 20)

        # Create engine for 100 dimensional feature space
        self.engine = Engine(100, lshashes=[rbpt], vector_filters=[NearestFilter(20)])

        # First insert 2000 random vectors
        for k in range(2000):
            x = numpy.random.randn(100)
            x_data = 'data'
            self.engine.store_vector(x, x_data)

        self.redis_storage.store_hash_configuration(rbpt)

        rbpt2 = RandomBinaryProjectionTree(None, None, None)
        rbpt2.apply_config(self.redis_storage.load_hash_configuration('testHash'))

        self.assertEqual(rbpt.dim, rbpt2.dim)
        self.assertEqual(rbpt.hash_name, rbpt2.hash_name)
        self.assertEqual(rbpt.projection_count, rbpt2.projection_count)

        for i in range(rbpt.normals.shape[0]):
            for j in range(rbpt.normals.shape[1]):
                self.assertEqual(rbpt.normals[i, j], rbpt2.normals[i, j])

        # Now do random queries and check result set size
        for k in range(10):
            x = numpy.random.randn(100)
            keys1 = rbpt.hash_vector(x, querying=True)
            keys2 = rbpt2.hash_vector(x, querying=True)
            self.assertEqual(len(keys1), len(keys2))
            for k in range(len(keys1)):
                self.assertEqual(keys1[k], keys2[k])

if __name__ == '__main__':
    unittest.main()
