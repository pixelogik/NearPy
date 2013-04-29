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

from scipy.spatial.distance import cdist


class RecallPrecisionExperiment(object):
    """
    Performs nearest neighbour recall experiments with custom vector data
    for all engines in the specified list.

    self.result contains list of (recall, precision) tuple.

    Because this experiment performs an exact search to first determine the
    real N nearest neighbours for each vector, the data set should not be
    too large if you don't want to wait too long for the result. Also
    the exact search computes a distance matrix that is quadratic in
    the data set size, so there are also memory limitations.
    """

    def __init__(self, N, vectors, engine_list):
        """
        Performs exact nearest neighbour search on the data set.

        vectors can either be a numpy matrix with all the vectors
        as columns OR a python array containing the individual
        numpy vectors.
        """
        self.N = N
        # We need a dict from vector string representation to index
        self.vector_dict = {}
        # Array for result tuples
        self.result = []

        # If vectors is not a numpy matrix, create one
        if not isinstance(vectors, numpy.ndarray):
            V = numpy.zeros((vectors[0].shape[0], len(vectors)))
            for index in range(len(vectors)):
                vector = vectors[index]
                V[:, index] = vector
            vectors = V

        for index in range(vectors.shape[1]):
            self.vector_dict[self.__vector_to_string(
                vectors[:, index])] = index

        # Get transposed version of vector matrix, so that the rows
        # are the vectors (needed by cdist)
        vectors_t = numpy.transpose(vectors)

        # Compute distance matrix
        D = cdist(vectors_t, vectors_t, 'euclidean')

        # For each vector get the closest N neigbbours from distance matrix
        self.closest = []
        for index in range(D.shape[1]):
            # This includes the vector itself, which is what we want
            self.closest.append(scipy.argsort(D[:, index])[:N])
            #print '\nClosest for %d:' % index
            #print self.closest[index]
            #print 'Vector:'
            #print vectors[:, index]

        # For each engine, first index vectors and then retrieve neighbours
        for engine in engine_list:
            # Clean storage
            engine.clean_all_buckets()
            # Use this to compute average recall
            avg_recall = 0.0
            # Use this to compute average precision
            avg_precision = 0.0

            # Index vectors and store them
            for index in range(vectors.shape[1]):
                #print 'Storing vector %d' % index
                #print vectors[:, index]
                engine.store_vector(vectors[:, index], 'testData')

            # Look for N nearest neighbours
            for index in range(vectors.shape[1]):
                # Get indices of the real nearest as set
                real_nearest = set(self.closest[index])
                #print 'Real nearest for %d:' % index
                #print real_nearest

                # Get nearest N according to engine
                nearest = engine.neighbours(vectors[:, index])

                #print 'Nearest according to engine (len=%d):' % len(nearest)
                #print [x[0] for x in nearest]

                # For comparance we need their indices (as set)
                nearest = set([self.__index_of_vector(x[0]) for x in nearest])

                # Get intersection count
                inter_count = float(len(real_nearest.intersection(nearest)))

                # Normalize recall for this vector
                recall = inter_count/float(len(real_nearest))
                #print 'Recall = %f' % recall

                # Normalize precision for this vector
                precision = inter_count/float(len(nearest))
                #print 'Precision = %f' % precision

                # Add to accumulator
                avg_recall += recall

                # Add to accumulator
                avg_precision += precision

            # Normalize recall over data set
            avg_recall = avg_recall / float(vectors.shape[1])

            # Normalize precision over data set
            avg_precision = avg_precision / float(vectors.shape[1])

            #print '\n AVG RECALL = %f\n' % avg_recall
            #print '\n AVG PRECISION = %f\n' % avg_precision
            self.result.append((avg_recall, avg_precision))

    def __vector_to_string(self, vector):
        """ Returns string representation of vector. """
        return numpy.array_str(vector)

    def __index_of_vector(self, vector):
        """ Returns index of specified vector from test data set. """
        return self.vector_dict[self.__vector_to_string(vector)]
