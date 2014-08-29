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
import logging

from nearpy import Engine
from nearpy.distances import CosineDistance

from nearpy.hashes import HashPermutations
from nearpy.hashes import RandomBinaryProjections

class TestPermutation(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.WARNING)

        # Create permutations meta-hash
        self.permutations = HashPermutations('permut')

        # Create binary hash as child hash
        rbp = RandomBinaryProjections('rbp1', 4)
        rbp_conf = {'num_permutation':50,'beam_size':10,'num_neighbour':100}

        # Add rbp as child hash of permutations hash
        self.permutations.add_child_hash(rbp, rbp_conf)

        # Create engine with meta hash and cosine distance
        self.engine_perm = Engine(200, lshashes=[self.permutations], distance=CosineDistance())

        # Create engine without permutation meta-hash
        self.engine = Engine(200, lshashes=[rbp], distance=CosineDistance())

    def test_runnable(self):

        # First index some random vectors
        matrix = numpy.zeros((1000,200))
        for i in xrange(1000):
            v = numpy.random.randn(200)
            matrix[i] = v
            self.engine.store_vector(v)
            self.engine_perm.store_vector(v)

        # Then update permuted index
        self.permutations.build_permuted_index()

        # Do random query on engine with permutations meta-hash
        print '\nNeighbour distances with permuted index:'
        query = numpy.random.randn(200)
        results = self.engine_perm.neighbours(query)
        dists = [x[2] for x in results]
        print dists

        # Do random query on engine without permutations meta-hash
        print '\nNeighbour distances without permuted index (distances should be larger):'
        results = self.engine.neighbours(query)
        dists = [x[2] for x in results]
        print dists

        # Real neighbours
        print '\nReal neighbour distances:'
        query = query.reshape((1,200))
        dists = CosineDistance().distance_matrix(matrix,query)
        dists = dists.reshape((-1,))
        dists = sorted(dists)
        print dists[:10]

if __name__ == '__main__':
    unittest.main()
