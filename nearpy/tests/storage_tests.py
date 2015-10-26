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
import scipy
import unittest

from mockredis import MockRedis as Redis

from nearpy.storage import MemoryStorage, RedisStorage


class TestStorage(unittest.TestCase):

    def setUp(self):
        self.memory = MemoryStorage()
        self.redis_object = Redis()
        self.redis_storage = RedisStorage(self.redis_object)

    def test_memory_storage(self):
        x = numpy.random.randn(100, 1)
        bucket_key = '23749283743928748'
        x_data = ['one', 'two', 'three']
        self.memory.store_vector('testHash', bucket_key, x, x_data)
        X = self.memory.get_bucket('testHash', bucket_key)
        self.assertEqual(len(X), 1)
        y = X[0][0]
        y_data = X[0][1]
        self.assertEqual(len(y), len(x))
        self.assertEqual(type(x), type(y))
        for k in range(100):
            self.assertEqual(y[k], x[k])
        self.assertEqual(type(y_data), type(x_data))
        self.assertEqual(len(y_data), len(x_data))
        for k in range(3):
            self.assertEqual(y_data[k], x_data[k])
        self.memory.clean_all_buckets()
        self.assertEqual(self.memory.get_bucket('testHash', bucket_key), [])

    def test_redis_storage(self):
        self.redis_storage.clean_all_buckets()
        x = numpy.random.randn(100, 1)
        bucket_key = '23749283743928748'
        x_data = ['one', 'two', 'three']
        self.redis_storage.store_vector('testHash', bucket_key, x, x_data)
        X = self.redis_storage.get_bucket('testHash', bucket_key)
        self.assertEqual(len(X), 1)
        y = X[0][0]
        y_data = X[0][1]
        self.assertEqual(len(y), len(x))
        self.assertEqual(type(x), type(y))
        for k in range(100):
            self.assertEqual(y[k], x[k])
        self.assertEqual(type(y_data), type(x_data))
        self.assertEqual(len(y_data), len(x_data))
        for k in range(3):
            self.assertEqual(y_data[k], x_data[k])
        self.redis_storage.clean_all_buckets()
        self.assertEqual(self.redis_storage.get_bucket('testHash',
                                                       bucket_key), [])

    def test_redis_storage_sparse(self):
        self.redis_storage.clean_all_buckets()
        x = scipy.sparse.rand(100, 1, density=0.1)
        bucket_key = '23749283743928748'
        x_data = ['one', 'two', 'three']
        self.redis_storage.store_vector('testHash', bucket_key, x, x_data)
        X = self.redis_storage.get_bucket('testHash', bucket_key)
        self.assertEqual(len(X), 1)
        y = X[0][0]
        y_data = X[0][1]
        self.assertEqual(type(x), type(y))
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertEqual(x.shape[1], y.shape[1])
        self.assertTrue((y - x).sum() == 0.0)
        self.assertEqual(type(y_data), type(x_data))
        self.assertEqual(len(y_data), len(x_data))
        for k in range(3):
            self.assertEqual(y_data[k], x_data[k])
        self.redis_storage.clean_all_buckets()
        self.assertEqual(self.redis_storage.get_bucket('testHash',
                                                       bucket_key), [])

if __name__ == '__main__':
    unittest.main()
