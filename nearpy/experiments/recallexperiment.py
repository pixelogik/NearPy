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
from scipy.spatial.distance import cdist


class RecallExperiment(object):
    """
    Performs nearest neighbour recall experiments with custom vector data
    for all engines in the specified list.

    Returns for each engine a floating point recall value. It measures how
    many of the closest 10 neighbours (according to an exact linear search)
    are in the search result returned by the individual engine.

    Because this experiment performs an exact search to first determine the
    real 10 nearest neighbours for each vector, the data set should not be
    too large if you don't want to wait too long for the result. Also
    the exact search computes a dense distance matrix that is quadratic in
    the data set size, so there are also memory limitations.
    """

    def __init__(self, vectors, engine_list):
        """
        Performs exact nearest neighbour search on the data set.

        vectors can either be a numpy matrix with all the vectors
        as columns OR a python array containing the individual
        numpy vectors.
        """
        # If vectors is NOT a numpy matrix, create one
        if not isinstance(vectors, numpy.ndarray):
            V = numpy.zeros((len(vectors), len(vectors)))
            for index in range(len(vectors)):
                vector = vectors[index]
                V[:, index] = vector
            vectors = V

        # Get transposed version of vector matrix, so that the rows
        # are the vectors (needed by cdist)
        vectors_t = numpy.transpose(vectors)

        # Compute distance matrix
        D = cdist(vectors_t, vectors_t, 'euclidean')
