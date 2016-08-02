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
import itertools
    

from nearpy import Engine
from nearpy.utils.utils import unitvec
from nearpy.hashes import UniBucket

class TestEngine(unittest.TestCase):

    def setUp(self):
        self.engine = Engine(1000)
        numpy.random.seed(4)

    def test_storage_issue(self):
        engine1 = Engine(100)
        engine2 = Engine(100)

        for k in range(1000):
            x = numpy.random.randn(100)
            x_data = 'data'
            engine1.store_vector(x, x_data)

        # Each engine should have its own default storage
        self.assertTrue(len(engine2.storage.buckets)==0)

    def test_retrieval(self):
        for k in range(100):
            self.engine.clean_all_buckets()
            x = numpy.random.randn(1000)
            x_data = 'data'
            self.engine.store_vector(x, x_data)
            n = self.engine.neighbours(x)
            y, y_data, y_distance  = n[0]
            normalized_x = unitvec(x)
            delta = 0.000000001
            self.assertAlmostEqual(numpy.abs((normalized_x - y)).max(), 0, delta=delta)
            self.assertEqual(y_data, x_data)
            self.assertAlmostEqual(y_distance, 0.0, delta=delta)

    def test_retrieval_sparse(self):
        for k in range(100):
            self.engine.clean_all_buckets()
            x = scipy.sparse.rand(1000, 1, density=0.05)
            x_data = 'data'
            self.engine.store_vector(x, x_data)
            n = self.engine.neighbours(x)
            y, y_data, y_distance = n[0]
            normalized_x = unitvec(x)
            delta = 0.000000001
            self.assertAlmostEqual(numpy.abs((normalized_x - y)).max(), 0, delta=delta)
            self.assertEqual(y_data, x_data)
            self.assertAlmostEqual(y_distance, 0.0, delta=delta)

    def get_keys(self, engine):
        def get_bucket_keys(lshash):
            bucket = engine.storage.buckets[lshash.hash_name][lshash.hash_name]
            return (i for v, i in bucket)

        return set(itertools.chain.from_iterable(
            get_bucket_keys(lshash) for lshash in engine.lshashes))

    def test_delete_vector_single_hash(self):
        dim = 5
        engine = Engine(dim, lshashes=[UniBucket('testHash')])
        # Index 20 random vectors (set their data to a unique string)
        for index in range(20):
            v = numpy.random.randn(dim)
            engine.store_vector(v, index)

        keys = self.get_keys(engine)

        engine.delete_vector(15)  #delete the vector with key = 15

        new_keys = self.get_keys(engine)

        self.assertGreater(len(keys), len(new_keys))  #new keys has 19 elements instead of 20
        self.assertIn(15, keys)
        self.assertNotIn(15, new_keys)  # the key 15 is the one missing

    def test_delete_vector_multiple_hash(self):
        dim = 5
        hashes = [UniBucket('name_hash_%d' % k) for k in range(10)]
        engine = Engine(dim, lshashes=hashes)
        # Index 20 random vectors (set their data to a unique string)
        for index in range(20):
            v = numpy.random.randn(dim)
            engine.store_vector(v, index)

        keys = self.get_keys(engine)

        engine.delete_vector(15)  #delete the vector with key = 15

        new_keys = self.get_keys(engine)

        self.assertGreater(len(keys), len(new_keys))  #new keys has 19 elements instead of 20
        self.assertIn(15, keys)
        self.assertNotIn(15, new_keys)  # the key 15 is the one missing


if __name__ == '__main__':
    unittest.main()
