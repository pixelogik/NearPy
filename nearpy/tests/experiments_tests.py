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
from __future__ import print_function

import numpy
import unittest

from nearpy.experiments import RecallPrecisionExperiment
from nearpy.hashes import UniBucket, RandomDiscretizedProjections, \
    RandomBinaryProjections, PCABinaryProjections
from nearpy.filters import NearestFilter, UniqueFilter
from nearpy.distances import CosineDistance, EuclideanDistance

from nearpy import Engine


class TestRecallExperiment(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(4)

    def test_experiment_with_unibucket_1(self):
        dim = 50
        vector_count = 100
        vectors = numpy.random.randn(dim, vector_count)
        unibucket = UniBucket('testHash')
        nearest = NearestFilter(10 + 1)
        engine = Engine(dim, lshashes=[unibucket],
                        vector_filters=[nearest],
                        distance=EuclideanDistance())
        exp = RecallPrecisionExperiment(10, vectors)
        result = exp.perform_experiment([engine])

        # Both recall and precision must be one in this case
        self.assertEqual(result[0][0], 1.0)
        self.assertEqual(result[0][1], 1.0)

    def test_experiment_with_unibucket_2(self):
        dim = 50
        vector_count = 100
        vectors = numpy.random.randn(dim, vector_count)
        unibucket = UniBucket('testHash')
        nearest = NearestFilter(10 + 1)
        engine = Engine(dim, lshashes=[unibucket],
                        vector_filters=[nearest])
        exp = RecallPrecisionExperiment(5, vectors)
        result = exp.perform_experiment([engine])

        # In this case precision is only 0.5
        # because the engine returns 10 nearest, but
        # the experiment only looks for 5 nearest.
        self.assertEqual(result[0][0], 1.0)
        self.assertEqual(result[0][1], 0.5)

    def test_experiment_with_unibucket_3(self):
        dim = 50
        vector_count = 100
        vectors = numpy.random.randn(dim, vector_count)
        unibucket = UniBucket('testHash')
        nearest = NearestFilter(5 + 1)
        engine = Engine(dim, lshashes=[unibucket],
                        vector_filters=[nearest])
        exp = RecallPrecisionExperiment(10, vectors)
        result = exp.perform_experiment([engine])

        # In this case recall is only 0.5
        # because the engine returns 5 nearest, but
        # the experiment looks for 10 nearest.
        self.assertEqual(result[0][0], 0.5)
        self.assertEqual(result[0][1], 1.0)

    def test_experiment_with_list_1(self):
        dim = 50
        vector_count = 100
        vectors = []
        for index in range(vector_count):
            vectors.append(numpy.random.randn(dim))
        unibucket = UniBucket('testHash')
        nearest = NearestFilter(10 + 1)
        engine = Engine(dim, lshashes=[unibucket],
                        vector_filters=[nearest])
        exp = RecallPrecisionExperiment(10, vectors)
        result = exp.perform_experiment([engine])

        # Both recall and precision must be one in this case
        self.assertEqual(result[0][0], 1.0)
        self.assertEqual(result[0][1], 1.0)

    def test_experiment_with_list_2(self):
        dim = 50
        vector_count = 100
        vectors = []
        for index in range(vector_count):
            vectors.append(numpy.random.randn(dim))
        unibucket = UniBucket('testHash')
        nearest = NearestFilter(10 + 1)
        engine = Engine(dim, lshashes=[unibucket],
                        vector_filters=[nearest])
        exp = RecallPrecisionExperiment(5, vectors)
        result = exp.perform_experiment([engine])

        # In this case precision is only 0.5
        # because the engine returns 10 nearest, but
        # the experiment only looks for 5 nearest.
        self.assertEqual(result[0][0], 1.0)
        self.assertEqual(result[0][1], 0.5)

    def test_experiment_with_list_3(self):
        dim = 50
        vector_count = 100
        vectors = []
        for index in range(vector_count):
            vectors.append(numpy.random.randn(dim))
        unibucket = UniBucket('testHash')
        nearest = NearestFilter(5 + 1)
        engine = Engine(dim, lshashes=[unibucket],
                        vector_filters=[nearest])
        exp = RecallPrecisionExperiment(10, vectors)
        result = exp.perform_experiment([engine])

        # In this case recall is only 0.5
        # because the engine returns 5 nearest, but
        # the experiment looks for 10 nearest.
        self.assertEqual(result[0][0], 0.5)
        self.assertEqual(result[0][1], 1.0)

    def test_random_discretized_projections(self):
        dim = 4
        vector_count = 5000
        vectors = numpy.random.randn(dim, vector_count)

        # First get recall and precision for one 1-dim random hash
        rdp = RandomDiscretizedProjections('rdp', 1, 0.01)
        nearest = NearestFilter(10 + 1)
        engine = Engine(dim, lshashes=[rdp],
                        vector_filters=[nearest])
        exp = RecallPrecisionExperiment(10, vectors)
        result = exp.perform_experiment([engine])

        recall1 = result[0][0]
        precision1 = result[0][1]
        searchtime1 = result[0][2]

        print('\nRecall RDP: %f, Precision RDP: %f, SearchTime RDP: %f\n' % \
            (recall1, precision1, searchtime1))

        # Then get recall and precision for one 4-dim random hash
        rdp = RandomDiscretizedProjections('rdp', 2, 0.2)
        engine = Engine(dim, lshashes=[rdp],
                        vector_filters=[nearest])
        result = exp.perform_experiment([engine])

        recall2 = result[0][0]
        precision2 = result[0][1]
        searchtime2 = result[0][2]

        print('\nRecall RDP: %f, Precision RDP: %f, SearchTime RDP: %f\n' % \
            (recall2, precision2, searchtime2))

        # Many things are random here, but the precision should increase
        # with dimension
        self.assertTrue(precision2 > precision1)

    def test_random_binary_projections(self):
        dim = 4
        vector_count = 5000
        vectors = numpy.random.randn(dim, vector_count)

        # First get recall and precision for one 1-dim random hash
        rbp = RandomBinaryProjections('rbp', 32)
        nearest = NearestFilter(10 + 1)
        engine = Engine(dim, lshashes=[rbp],
                        vector_filters=[nearest])
        exp = RecallPrecisionExperiment(10, vectors)
        result = exp.perform_experiment([engine])

        recall1 = result[0][0]
        precision1 = result[0][1]
        searchtime1 = result[0][2]

        print('\nRecall RBP: %f, Precision RBP: %f, SearchTime RBP: %f\n' % \
            (recall1, precision1, searchtime1))

if __name__ == '__main__':
    unittest.main()
