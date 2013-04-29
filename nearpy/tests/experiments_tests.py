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

from nearpy.experiments import RecallPrecisionExperiment
from nearpy.hashes import UniBucket
from nearpy.filters import NearestFilter
from nearpy.distances import AngularDistance

from nearpy import Engine


class TestRecallExperiment(unittest.TestCase):

    def setUp(self):
        self.dim = 1
        self.engines = []
        self.engines.append(Engine(self.dim))
        self.engines.append(Engine(self.dim, distance=AngularDistance()))
        self.vectors = numpy.random.randn(self.dim, 100)

    def test_experiment_with_unibucket_1(self):
        dim = 50
        vector_count = 100
        vectors = numpy.random.randn(dim, vector_count)
        unibucket = UniBucket('testHash')
        nearest = NearestFilter(10)
        engine = Engine(self.dim, lshashes=[unibucket],
                        vector_filters=[nearest])
        exp = RecallPrecisionExperiment(10, self.vectors, [engine])

        # Both recall and precision must be one in this case
        self.assertEqual(exp.result[0][0], 1.0)
        self.assertEqual(exp.result[0][1], 1.0)

    def test_experiment_with_unibucket_2(self):
        dim = 50
        vector_count = 100
        vectors = numpy.random.randn(dim, vector_count)
        unibucket = UniBucket('testHash')
        nearest = NearestFilter(10)
        engine = Engine(self.dim, lshashes=[unibucket],
                        vector_filters=[nearest])
        exp = RecallPrecisionExperiment(5, self.vectors, [engine])

        # In this case precision is only 0.5
        # because the engine returns 10 nearest, but
        # the experiment only looks for 5 nearest.
        self.assertEqual(exp.result[0][0], 1.0)
        self.assertEqual(exp.result[0][1], 0.5)

    def test_experiment_with_unibucket_3(self):
        dim = 50
        vector_count = 100
        vectors = numpy.random.randn(dim, vector_count)
        unibucket = UniBucket('testHash')
        nearest = NearestFilter(5)
        engine = Engine(self.dim, lshashes=[unibucket],
                        vector_filters=[nearest])
        exp = RecallPrecisionExperiment(10, self.vectors, [engine])

        # In this case recall is only 0.5
        # because the engine returns 5 nearest, but
        # the experiment looks for 10 nearest.
        self.assertEqual(exp.result[0][0], 0.5)
        self.assertEqual(exp.result[0][1], 1.0)

    def test_experiment_with_list(self):
        # TODO!!!
        pass

    def test_random_discretized_projections(self):
        pass

if __name__ == '__main__':
    unittest.main()
