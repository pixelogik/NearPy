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

from nearpy.filters import NearestFilter, DistanceThresholdFilter, UniqueFilter


class TestVectorFilters(unittest.TestCase):

    def setUp(self):
        self.V = []
        self.V.append((numpy.array([0]), 'data1', 0.4))
        self.V.append((numpy.array([1]), 'data2', 0.9))
        self.V.append((numpy.array([2]), 'data3', 1.4))
        self.V.append((numpy.array([3]), 'data4', 2.1))
        self.V.append((numpy.array([4]), 'data5', 0.1))
        self.V.append((numpy.array([5]), 'data6', 8.7))
        self.V.append((numpy.array([6]), 'data7', 3.4))
        self.V.append((numpy.array([7]), 'data8', 2.8))

        self.threshold_filter = DistanceThresholdFilter(1.0)
        self.nearest_filter = NearestFilter(5)
        self.unique = UniqueFilter()

    def test_thresholding(self):
        result = self.threshold_filter.filter_vectors(self.V)
        self.assertEqual(len(result), 3)
        self.assertIn(self.V[0], result)
        self.assertIn(self.V[1], result)
        self.assertIn(self.V[4], result)

    def test_nearest(self):
        result = self.nearest_filter.filter_vectors(self.V)
        self.assertEqual(len(result), 5)
        self.assertIn(self.V[0], result)
        self.assertIn(self.V[1], result)
        self.assertIn(self.V[4], result)
        self.assertIn(self.V[2], result)
        self.assertIn(self.V[3], result)

    def test_unique(self):
        W = self.V
        W.append((numpy.array([7]), 'data8', 2.8))
        W.append((numpy.array([0]), 'data1', 2.8))
        W.append((numpy.array([1]), 'data2', 2.8))
        W.append((numpy.array([6]), 'data7', 2.8))

        result = self.unique.filter_vectors(W)
        self.assertEqual(len(result), 8)


if __name__ == '__main__':
    unittest.main()
