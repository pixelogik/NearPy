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
import time

from nearpy import Engine
from nearpy.distances import CosineDistance

from nearpy.hashes import RandomBinaryProjections, HashPermutations, HashPermutationMapper

def example2():

    # Dimension of feature space
    DIM = 100

    # Number of data points (dont do too much because of exact search)
    POINTS = 20000

    ##########################################################

    print('Performing indexing with HashPermutations...')
    t0 = time.time()

    # Create permutations meta-hash
    permutations = HashPermutations('permut')

    # Create binary hash as child hash
    rbp_perm = RandomBinaryProjections('rbp_perm', 14)
    rbp_conf = {'num_permutation':50,'beam_size':10,'num_neighbour':100}

    # Add rbp as child hash of permutations hash
    permutations.add_child_hash(rbp_perm, rbp_conf)

    # Create engine
    engine_perm = Engine(DIM, lshashes=[permutations], distance=CosineDistance())

    # First index some random vectors
    matrix = numpy.zeros((POINTS,DIM))
    for i in range(POINTS):
        v = numpy.random.randn(DIM)
        matrix[i] = v
        engine_perm.store_vector(v)

    # Then update permuted index
    permutations.build_permuted_index()

    t1 = time.time()
    print('Indexing took %f seconds' % (t1-t0))

    # Get random query vector
    query = numpy.random.randn(DIM)

    # Do random query on engine 3
    print('\nNeighbour distances with HashPermutations:')
    print('  -> Candidate count is %d' % engine_perm.candidate_count(query))
    results = engine_perm.neighbours(query)
    dists = [x[2] for x in results]
    print(dists)

    # Real neighbours
    print('\nReal neighbour distances:')
    query = query.reshape((DIM))
    dists = CosineDistance().distance(matrix, query)
    dists = dists.reshape((-1,))
    dists = sorted(dists)
    print(dists[:10])

    ##########################################################

    print('\nPerforming indexing with HashPermutationMapper...')
    t0 = time.time()

    # Create permutations meta-hash
    permutations2 = HashPermutationMapper('permut2')

    # Create binary hash as child hash
    rbp_perm2 = RandomBinaryProjections('rbp_perm2', 14)

    # Add rbp as child hash of permutations hash
    permutations2.add_child_hash(rbp_perm2)

    # Create engine
    engine_perm2 = Engine(DIM, lshashes=[permutations2], distance=CosineDistance())

    # First index some random vectors
    matrix = numpy.zeros((POINTS,DIM))
    for i in range(POINTS):
        v = numpy.random.randn(DIM)
        matrix[i] = v
        engine_perm2.store_vector(v)

    t1 = time.time()
    print('Indexing took %f seconds' % (t1-t0))

    # Get random query vector
    query = numpy.random.randn(DIM)

    # Do random query on engine 4
    print('\nNeighbour distances with HashPermutationMapper:')
    print('  -> Candidate count is %d' % engine_perm2.candidate_count(query))
    results = engine_perm2.neighbours(query)
    dists = [x[2] for x in results]
    print(dists)

    # Real neighbours
    print('\nReal neighbour distances:')
    query = query.reshape((DIM))
    dists = CosineDistance().distance(matrix,query)
    dists = dists.reshape((-1,))
    dists = sorted(dists)
    print(dists[:10])

    ##########################################################

    print('\nPerforming indexing with multiple binary hashes...')
    t0 = time.time()

    hashes = []
    for k in range(20):
        hashes.append(RandomBinaryProjections('rbp_%d' % k, 10))

    # Create engine
    engine_rbps = Engine(DIM, lshashes=hashes, distance=CosineDistance())

    # First index some random vectors
    matrix = numpy.zeros((POINTS,DIM))
    for i in range(POINTS):
        v = numpy.random.randn(DIM)
        matrix[i] = v
        engine_rbps.store_vector(v)

    t1 = time.time()
    print('Indexing took %f seconds' % (t1-t0))

    # Get random query vector
    query = numpy.random.randn(DIM)

    # Do random query on engine 4
    print('\nNeighbour distances with multiple binary hashes:')
    print('  -> Candidate count is %d' % engine_rbps.candidate_count(query))
    results = engine_rbps.neighbours(query)
    dists = [x[2] for x in results]
    print(dists)

    # Real neighbours
    print('\nReal neighbour distances:')
    query = query.reshape((DIM))
    dists = CosineDistance().distance(matrix,query)
    dists = dists.reshape((-1,))
    dists = sorted(dists)
    print(dists[:10])

    ##########################################################
