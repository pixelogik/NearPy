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

from nearpy.distances import EuclideanDistance, CosineDistance, ManhattanDistance

########################################################################

# Helper functions


def check_distance_symmetry(test_obj, distance):
    for k in range(100):
        x = numpy.random.randn(10)
        y = numpy.random.randn(10)
        d_xy = distance.distance(x, y)
        d_yx = distance.distance(y, x)

        # I had precision issues with a local install. This test is more tolerant to that.
        test_obj.assertAlmostEqual(d_xy, d_yx, delta=0.00000000000001)

    for k in range(100):
        x = scipy.sparse.rand(30, 1, density=0.3)
        y = scipy.sparse.rand(30, 1, density=0.3)
        d_xy = distance.distance(x, y)
        d_yx = distance.distance(y, x)

        # I had precision issues with a local install. This test is more tolerant to that.
        test_obj.assertAlmostEqual(d_xy, d_yx, delta=0.00000000000001)


def check_distance_triangle_inequality(test_obj, distance):
    for k in range(100):
        x = numpy.random.randn(10)
        y = numpy.random.randn(10)
        z = numpy.random.randn(10)

        d_xy = distance.distance(x, y)
        d_xz = distance.distance(x, z)
        d_yz = distance.distance(y, z)

        test_obj.assertLessEqual(d_xy, d_xz + d_yz)

    for k in range(100):
        x = scipy.sparse.rand(30, 1, density=0.3)
        y = scipy.sparse.rand(30, 1, density=0.3)
        z = scipy.sparse.rand(30, 1, density=0.3)

        d_xy = distance.distance(x, y)
        d_xz = distance.distance(x, z)
        d_yz = distance.distance(y, z)

        test_obj.assertTrue(d_xy <= d_xz + d_yz)

########################################################################


class TestEuclideanDistance(unittest.TestCase):

    def setUp(self):
        self.euclidean = EuclideanDistance()

    def test_triangle_inequality(self):
        check_distance_triangle_inequality(self, self.euclidean)

    def test_symmetry(self):
        check_distance_symmetry(self, self.euclidean)

class TestCosineDistance(unittest.TestCase):

    def setUp(self):
        self.cosine = CosineDistance()

    def test_symmetry(self):
        check_distance_symmetry(self, self.cosine)

class TestManhattanDistance(unittest.TestCase):

    def setUp(self):
        self.manhattan = ManhattanDistance()

    def test_triangle_inequality(self):
        check_distance_triangle_inequality(self, self.manhattan)

    def test_symmetry(self):
        check_distance_symmetry(self, self.manhattan)

if __name__ == '__main__':
    unittest.main()
