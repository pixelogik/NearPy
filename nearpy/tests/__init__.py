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
from __future__ import absolute_import

from nearpy.tests.hashes_tests import TestRandomBinaryProjections, \
    TestRandomDiscretizedProjections, TestPCABinaryProjections, TestPCADiscretizedProjections
from nearpy.tests.engine_tests import TestEngine
from nearpy.tests.storage_tests import TestStorage
from nearpy.tests.distances_tests import TestEuclideanDistance, TestCosineDistance, TestManhattanDistance
from nearpy.tests.filters_tests import TestVectorFilters
from nearpy.tests.experiments_tests import TestRecallExperiment
from nearpy.tests.hash_storage_tests import TestHashStorage
from nearpy.tests.projection_trees_tests import TestRandomBinaryProjectionTree
from nearpy.tests.permutation_tests import TestPermutation
