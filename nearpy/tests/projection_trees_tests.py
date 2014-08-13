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
import json

from nearpy import Engine
from nearpy.filters import NearestFilter
from nearpy.hashes import RandomBinaryProjectionTree


class TestRandomBinaryProjectionTree(unittest.TestCase):

    def test_retrieval(self):
        # We want 10 projections, 20 results at least
        rbpt = RandomBinaryProjectionTree('testHash', 12, 20)

        # Create engine for 100 dimensional feature space, do not forget to set
        # nearest filter to 20, because default is 10
        self.engine = Engine(100, lshashes=[rbpt], vector_filters=[NearestFilter(20)])

        # First insert 200000 random vectors
        print 'Indexing...'
        for k in range(200000):
            x = numpy.random.randn(100)
            x_data = 'data'
            self.engine.store_vector(x, x_data)

        print rbpt.tree_root

        # Now do random queries and check result set size
        print 'Querying...'
        for k in range(10):
            x = numpy.random.randn(100)
            n = self.engine.neighbours(x)
            print "Candidate count = %d" % self.engine.candidate_count(x)
            print "Result size = %d" % len(n)
            self.assertEqual(len(n), 20)

if __name__ == '__main__':
    unittest.main()
