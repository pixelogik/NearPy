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

import numpy
import unittest

from mockredis import MockRedis as Redis

from nearpy.storage import MemoryStorage, RedisStorage

from nearpy.hashes import RandomBinaryProjections, \
    RandomDiscretizedProjections, \
    PCABinaryProjections, PCADiscretizedProjections


class TestHashStorage(unittest.TestCase):

    def setUp(self):
        self.memory = MemoryStorage()
        self.redis_object = Redis()
        self.redis_storage = RedisStorage(self.redis_object)

    def test_hash_memory_storage_none_config(self):
        conf = self.memory.load_hash_configuration('nonexistentHash')

        self.assertIsNone(conf)

    def test_hash_memory_storage_rbp(self):
        hash1 = RandomBinaryProjections('testRBPHash', 10)
        hash1.reset(100)

        self.memory.store_hash_configuration(hash1)

        hash2 = RandomBinaryProjections(None, None)
        hash2.apply_config(self.memory.load_hash_configuration('testRBPHash'))

        self.assertEqual(hash1.dim, hash2.dim)
        self.assertEqual(hash1.hash_name, hash2.hash_name)
        self.assertEqual(hash1.projection_count, hash2.projection_count)

        for i in range(hash1.normals.shape[0]):
            for j in range(hash1.normals.shape[1]):
                self.assertEqual(hash1.normals[i, j], hash2.normals[i, j])

    def test_hash_memory_storage_rdp(self):
        hash1 = RandomDiscretizedProjections('testRDPHash', 10, 0.1)
        hash1.reset(100)

        self.memory.store_hash_configuration(hash1)

        hash2 = RandomDiscretizedProjections(None, None, None)
        hash2.apply_config(self.memory.load_hash_configuration('testRDPHash'))

        self.assertEqual(hash1.dim, hash2.dim)
        self.assertEqual(hash1.hash_name, hash2.hash_name)
        self.assertEqual(hash1.bin_width, hash2.bin_width)
        self.assertEqual(hash1.projection_count, hash2.projection_count)

        for i in range(hash1.normals.shape[0]):
            for j in range(hash1.normals.shape[1]):
                self.assertEqual(hash1.normals[i, j], hash2.normals[i, j])

    def test_hash_memory_storage_pcabp(self):
        train_vectors = numpy.random.randn(10, 100)
        hash1 = PCABinaryProjections('testPCABPHash', 4, train_vectors)

        self.memory.store_hash_configuration(hash1)

        hash2 = PCABinaryProjections(None, None, None)
        hash2.apply_config(self.memory.load_hash_configuration('testPCABPHash'))

        self.assertEqual(hash1.dim, hash2.dim)
        self.assertEqual(hash1.hash_name, hash2.hash_name)
        self.assertEqual(hash1.projection_count, hash2.projection_count)

        for i in range(hash1.components.shape[0]):
            for j in range(hash1.components.shape[1]):
                self.assertEqual(hash1.components[i, j], hash2.components[i, j])

    def test_hash_memory_storage_pcadp(self):
        train_vectors = numpy.random.randn(10, 100)
        hash1 = PCADiscretizedProjections('testPCADPHash', 4, train_vectors, 0.1)

        self.memory.store_hash_configuration(hash1)

        hash2 = PCADiscretizedProjections(None, None, None, None)
        hash2.apply_config(self.memory.load_hash_configuration('testPCADPHash'))

        self.assertEqual(hash1.dim, hash2.dim)
        self.assertEqual(hash1.hash_name, hash2.hash_name)
        self.assertEqual(hash1.bin_width, hash2.bin_width)
        self.assertEqual(hash1.projection_count, hash2.projection_count)

        for i in range(hash1.components.shape[0]):
            for j in range(hash1.components.shape[1]):
                self.assertEqual(hash1.components[i, j], hash2.components[i, j])

    def test_hash_redis_storage_none_config(self):
        conf = self.redis_storage.load_hash_configuration('nonexistentHash')

        self.assertIsNone(conf)

    def test_hash_redis_storage_rbp(self):
        hash1 = RandomBinaryProjections('testRBPHash', 10)
        hash1.reset(100)

        self.redis_storage.store_hash_configuration(hash1)

        hash2 = RandomBinaryProjections(None, None)
        hash2.apply_config(self.redis_storage.load_hash_configuration('testRBPHash'))

        self.assertEqual(hash1.dim, hash2.dim)
        self.assertEqual(hash1.hash_name, hash2.hash_name)
        self.assertEqual(hash1.projection_count, hash2.projection_count)

        for i in range(hash1.normals.shape[0]):
            for j in range(hash1.normals.shape[1]):
                self.assertEqual(hash1.normals[i, j], hash2.normals[i, j])

    def test_hash_redis_storage_rdp(self):
        hash1 = RandomDiscretizedProjections('testRDPHash', 10, 0.1)
        hash1.reset(100)

        self.redis_storage.store_hash_configuration(hash1)

        hash2 = RandomDiscretizedProjections(None, None, None)
        hash2.apply_config(self.redis_storage.load_hash_configuration('testRDPHash'))

        self.assertEqual(hash1.dim, hash2.dim)
        self.assertEqual(hash1.hash_name, hash2.hash_name)
        self.assertEqual(hash1.bin_width, hash2.bin_width)
        self.assertEqual(hash1.projection_count, hash2.projection_count)

        for i in range(hash1.normals.shape[0]):
            for j in range(hash1.normals.shape[1]):
                self.assertEqual(hash1.normals[i, j], hash2.normals[i, j])

    def test_hash_redis_storage_pcabp(self):
        train_vectors = numpy.random.randn(10, 100)
        hash1 = PCABinaryProjections('testPCABPHash', 4, train_vectors)

        self.redis_storage.store_hash_configuration(hash1)

        hash2 = PCABinaryProjections(None, None, None)
        hash2.apply_config(self.redis_storage.load_hash_configuration('testPCABPHash'))

        self.assertEqual(hash1.dim, hash2.dim)
        self.assertEqual(hash1.hash_name, hash2.hash_name)
        self.assertEqual(hash1.projection_count, hash2.projection_count)

        for i in range(hash1.components.shape[0]):
            for j in range(hash1.components.shape[1]):
                self.assertEqual(hash1.components[i, j], hash2.components[i, j])

    def test_hash_redis_storage_pcadp(self):
        train_vectors = numpy.random.randn(10, 100)
        hash1 = PCADiscretizedProjections('testPCADPHash', 4, train_vectors, 0.1)

        self.redis_storage.store_hash_configuration(hash1)

        hash2 = PCADiscretizedProjections(None, None, None, None)
        hash2.apply_config(self.redis_storage.load_hash_configuration('testPCADPHash'))

        self.assertEqual(hash1.dim, hash2.dim)
        self.assertEqual(hash1.hash_name, hash2.hash_name)
        self.assertEqual(hash1.bin_width, hash2.bin_width)
        self.assertEqual(hash1.projection_count, hash2.projection_count)

        for i in range(hash1.components.shape[0]):
            for j in range(hash1.components.shape[1]):
                self.assertEqual(hash1.components[i, j], hash2.components[i, j])


if __name__ == '__main__':
    unittest.main()
