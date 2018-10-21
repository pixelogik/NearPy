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

from nearpy.hashes import RandomBinaryProjections, \
    RandomDiscretizedProjections, \
    PCABinaryProjections, PCADiscretizedProjections


class TestRandomBinaryProjections(unittest.TestCase):

    def setUp(self):
        self.rbp = RandomBinaryProjections('testHash', 10)
        self.rbp.reset(100)

    def test_hash_format(self):
        h = self.rbp.hash_vector(numpy.random.randn(100))
        self.assertEqual(len(h), 1)
        self.assertEqual(type(h[0]), type(''))
        self.assertEqual(len(h[0]), 10)
        for c in h[0]:
            self.assertTrue(c == '1' or c == '0')

    def test_hash_deterministic(self):
        x = numpy.random.randn(100)
        first_hash = self.rbp.hash_vector(x)[0]
        for k in range(100):
            self.assertEqual(first_hash, self.rbp.hash_vector(x)[0])

    def test_hash_format_sparse(self):
        h = self.rbp.hash_vector(scipy.sparse.rand(100, 1, density=0.1))
        self.assertEqual(len(h), 1)
        self.assertEqual(type(h[0]), type(''))
        self.assertEqual(len(h[0]), 10)
        for c in h[0]:
            self.assertTrue(c == '1' or c == '0')

    def test_hash_deterministic_sparse(self):
        x = scipy.sparse.rand(100, 1, density=0.1)
        first_hash = self.rbp.hash_vector(x)[0]
        for k in range(100):
            self.assertEqual(first_hash, self.rbp.hash_vector(x)[0])

class TestRandomDiscretizedProjections(unittest.TestCase):

    def setUp(self):
        self.rbp = RandomDiscretizedProjections('testHash', 10, 0.1)
        self.rbp.reset(100)

    def test_hash_format(self):
        h = self.rbp.hash_vector(numpy.random.randn(100))
        self.assertEqual(len(h), 1)
        self.assertEqual(type(h[0]), type(''))

    def test_hash_deterministic(self):
        x = numpy.random.randn(100)
        first_hash = self.rbp.hash_vector(x)[0]
        for k in range(100):
            self.assertEqual(first_hash, self.rbp.hash_vector(x)[0])

    def test_hash_format_sparse(self):
        h = self.rbp.hash_vector(scipy.sparse.rand(100, 1, density=0.1))
        self.assertEqual(len(h), 1)
        self.assertEqual(type(h[0]), type(''))

    def test_hash_deterministic_sparse(self):
        x = scipy.sparse.rand(100, 1, density=0.1)
        first_hash = self.rbp.hash_vector(x)[0]
        for k in range(100):
            self.assertEqual(first_hash, self.rbp.hash_vector(x)[0])

class TestPCABinaryProjections(unittest.TestCase):

    def setUp(self):
        self.vectors = numpy.random.randn(10, 100)
        self.pbp = PCABinaryProjections('pbp', 4, self.vectors)

    def test_hash_format(self):
        h = self.pbp.hash_vector(numpy.random.randn(10))
        self.assertEqual(len(h), 1)
        self.assertEqual(type(h[0]), type(''))
        self.assertEqual(len(h[0]), 4)
        for c in h[0]:
            self.assertTrue(c == '1' or c == '0')

    def test_hash_deterministic(self):
        x = numpy.random.randn(10)
        first_hash = self.pbp.hash_vector(x)[0]
        for k in range(100):
            self.assertEqual(first_hash, self.pbp.hash_vector(x)[0])

    def test_hash_format_sparse(self):
        h = self.pbp.hash_vector(scipy.sparse.rand(10, 1, density=0.6))
        self.assertEqual(len(h), 1)
        self.assertEqual(type(h[0]), type(''))
        self.assertEqual(len(h[0]), 4)
        for c in h[0]:
            self.assertTrue(c == '1' or c == '0')

    def test_hash_deterministic_sparse(self):
        x = scipy.sparse.rand(10, 1, density=0.6)
        first_hash = self.pbp.hash_vector(x)[0]
        for k in range(100):
            self.assertEqual(first_hash, self.pbp.hash_vector(x)[0])


class TestPCADiscretizedProjections(unittest.TestCase):

    def setUp(self):
        self.vectors = numpy.random.randn(10, 100)
        self.pdp = PCADiscretizedProjections('pdp', 4, self.vectors, 0.1)

    def test_hash_format(self):
        h = self.pdp.hash_vector(numpy.random.randn(10))
        self.assertEqual(len(h), 1)
        self.assertEqual(type(h[0]), type(''))

    def test_hash_deterministic(self):
        x = numpy.random.randn(10)
        first_hash = self.pdp.hash_vector(x)[0]
        for k in range(100):
            self.assertEqual(first_hash, self.pdp.hash_vector(x)[0])

    def test_hash_format_sparse(self):
        h = self.pdp.hash_vector(scipy.sparse.rand(10, 1, density=0.6))
        self.assertEqual(len(h), 1)
        self.assertEqual(type(h[0]), type(''))

    def test_hash_deterministic_sparse(self):
        x = scipy.sparse.rand(10, 1, density=0.6)
        first_hash = self.pdp.hash_vector(x)[0]
        for k in range(100):
            self.assertEqual(first_hash, self.pdp.hash_vector(x)[0])


if __name__ == '__main__':
    unittest.main()
