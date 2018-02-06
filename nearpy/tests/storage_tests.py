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
import numpy
import scipy

from mockredis import MockRedis as Redis

from future.builtins import range

from nearpy.storage import MemoryStorage, RedisStorage


class StorageTest(unittest.TestCase):

    """
    Base class for storage tests.
    """

    def setUp(self):
        self.storage.clean_all_buckets()
        numpy.random.seed(4)

    def check_store_vector(self, x):
        bucket_key = '23749283743928748'
        x_data = ['one', 'two', 'three']
        self.storage.store_vector('testHash', bucket_key, x, x_data)
        bucket = self.storage.get_bucket('testHash', bucket_key)
        self.assertEqual(len(bucket), 1)
        y, y_data = bucket[0]
        self.assertEqual(type(y), type(x))
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(max(abs(y - x)), 0)
        self.assertEqual(y_data, x_data)
        self.storage.clean_all_buckets()
        self.assertEqual(self.storage.get_bucket('testHash', bucket_key), [])

    def check_get_all_bucket_keys(self):
        x, x_data = numpy.ones(100), "data"
        hash_config = [
            ("firstHash", ["1", "2", "3", "4"]),
            ("secondHash", ["10", "20", "3", "4", "50"]),
        ]
        for hash_name, bucket_keys in hash_config:
            for bucket_key in bucket_keys:
                self.storage.store_vector(hash_name, bucket_key, x, x_data)
        for hash_name, bucket_keys in hash_config:
            self.assertSequenceEqual(
                sorted(self.storage.get_all_bucket_keys(hash_name)),
                sorted(bucket_keys)
            )

    def check_delete_vector(self, x):
        hash_name, bucket_name = "tastHash", "testBucket"
        samples = list(range(10))
        for sample in samples:
            self.storage.store_vector(hash_name, bucket_name, x, sample)

        def get_bucket_items():
            return [data for v, data
                    in self.storage.get_bucket(hash_name, bucket_name)]
        self.assertEqual(get_bucket_items(), samples)
        deleted_sample = 4
        self.storage.delete_vector(hash_name, [bucket_name], deleted_sample)
        samples.remove(deleted_sample)
        self.assertEqual(get_bucket_items(), samples)


class MemoryStorageTest(StorageTest):

    def setUp(self):
        self.storage = MemoryStorage()
        super(MemoryStorageTest, self).setUp()

    def test_store_vector(self):
        x = numpy.random.randn(100, 1)
        self.check_store_vector(x)

    def test_store_sparse_vector(self):
        x = scipy.sparse.rand(100, 1, density=0.1)
        self.check_store_vector(x)

    def test_get_all_bucket_keys(self):
        self.check_get_all_bucket_keys()

    def test_delete_vector(self):
        self.check_delete_vector(numpy.ones(100))


class RedisStorageTest(StorageTest):

    def setUp(self):
        self.storage = RedisStorage(Redis())
        super(RedisStorageTest, self).setUp()

    def test_store_vector(self):
        x = numpy.random.randn(100, 1).ravel()
        self.check_store_vector(x)

    def test_store_sparse_vector(self):
        x = scipy.sparse.rand(100, 1, density=0.1)
        self.check_store_vector(x)

    def test_get_all_bucket_keys(self):
        self.check_get_all_bucket_keys()

    def test_delete_vector(self):
        self.check_delete_vector(numpy.ones(100))

    def test_store_zero(self):
        x = numpy.ones(100)
        hash_name, bucket_name = "tastHash", "testBucket"
        self.storage.store_vector(hash_name, bucket_name, x, 0)
        bucket = self.storage.get_bucket(hash_name, bucket_name)
        _, data = bucket[0]
        self.assertEqual(data, 0)

if __name__ == '__main__':
    unittest.main()
