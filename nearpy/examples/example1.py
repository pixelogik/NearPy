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
import nearpy.utils.utils
from nearpy import Engine
from nearpy.distances import CosineDistance

from nearpy.hashes import RandomBinaryProjections, RandomBinaryProjectionTree, HashPermutations, HashPermutationMapper

def print_results(results):
    print('  Data \t| Distance')
    for r in results:
        data = r[1]
        dist = r[2]
        print('  {} \t| {:.4f}'.format(data, dist))

def example1():

    # Dimension of feature space
    DIM = 100

    # Number of data points (dont do too much because of exact search)
    POINTS = 10000

    print('Creating engines')

    # We want 12 projections, 20 results at least
    rbpt = RandomBinaryProjectionTree('rbpt', 20, 20)

    # Create engine 1
    engine_rbpt = Engine(DIM, lshashes=[rbpt], distance=CosineDistance())

    # Create binary hash as child hash
    rbp = RandomBinaryProjections('rbp1', 20)

    # Create engine 2
    engine = Engine(DIM, lshashes=[rbp], distance=CosineDistance())

    # Create permutations meta-hash
    permutations = HashPermutations('permut')

    # Create binary hash as child hash
    rbp_perm = RandomBinaryProjections('rbp_perm', 20)
    rbp_conf = {'num_permutation':50,'beam_size':10,'num_neighbour':100}

    # Add rbp as child hash of permutations hash
    permutations.add_child_hash(rbp_perm, rbp_conf)

    # Create engine 3
    engine_perm = Engine(DIM, lshashes=[permutations], distance=CosineDistance())

    # Create permutations meta-hash
    permutations2 = HashPermutationMapper('permut2')

    # Create binary hash as child hash
    rbp_perm2 = RandomBinaryProjections('rbp_perm2', 12)

    # Add rbp as child hash of permutations hash
    permutations2.add_child_hash(rbp_perm2)

    # Create engine 3
    engine_perm2 = Engine(DIM, lshashes=[permutations2], distance=CosineDistance())

    print('Indexing %d random vectors of dimension %d' % (POINTS, DIM))

    # First index some random vectors
    matrix = numpy.zeros((POINTS,DIM))
    for i in xrange(POINTS):
        v = numpy.random.randn(DIM)
        matrix[i, :] = nearpy.utils.utils.unitvec(v)
        engine.store_vector(v, i)
        engine_rbpt.store_vector(v, i)
        engine_perm.store_vector(v, i)
        engine_perm2.store_vector(v, i)

    print('Buckets 1 = %d' % len(engine.storage.buckets['rbp1'].keys()))
    print('Buckets 2 = %d' % len(engine_rbpt.storage.buckets['rbpt'].keys()))

    print('Building permuted index for HashPermutations')

    # Then update permuted index
    permutations.build_permuted_index()

    print('Generate random data')

    # Get random query vector
    query = numpy.random.randn(DIM)

    # Do random query on engine 1
    print('\nNeighbour distances with RandomBinaryProjectionTree:')
    print('  -> Candidate count is %d' % engine_rbpt.candidate_count(query))
    results = engine_rbpt.neighbours(query)
    print_results(results)

    # Do random query on engine 2
    print('\nNeighbour distances with RandomBinaryProjections:')
    print('  -> Candidate count is %d' % engine.candidate_count(query))
    results = engine.neighbours(query)
    print_results(results)

    # Do random query on engine 3
    print('\nNeighbour distances with HashPermutations:')
    print('  -> Candidate count is %d' % engine_perm.candidate_count(query))
    results = engine_perm.neighbours(query)
    print_results(results)

    # Do random query on engine 4
    print('\nNeighbour distances with HashPermutations2:')
    print('  -> Candidate count is %d' % engine_perm2.candidate_count(query))
    results = engine_perm2.neighbours(query)
    print_results(results)

    # Real neighbours
    print('\nReal neighbour distances:')
    query = nearpy.utils.utils.unitvec(query)
    query = query.reshape((DIM, 1))
    dists = CosineDistance().distance(matrix,query)
    dists = dists.reshape((-1,))
    # dists = sorted(dists)

    dists_argsort = numpy.argsort(dists)

    results = [(None, d, dists[d]) for d in dists_argsort[:10]]
    print_results(results)



