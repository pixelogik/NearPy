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

from nearpy import Engine
from nearpy.utils.utils import unitvec


class TestEngine(unittest.TestCase):

    def setUp(self):
        self.engine = Engine(1000)

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

if __name__ == '__main__':
    unittest.main()
