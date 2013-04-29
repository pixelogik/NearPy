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
import time

from scipy.spatial.distance import cdist

from nearpy.utils import numpy_array_from_list_or_numpy_array


class RecallPrecisionExperiment(object):
    """
    Performs nearest neighbour recall experiments with custom vector data
    for all engines in the specified list.

    perform_experiment() returns list of (recall, precision, search_time)
    tuple. All are the averaged values over all request vectors. search_time
    is the average retrieval/search time compared to the average exact search
    time.

    Because this experiment performs an exact search to first determine the
    real N nearest neighbours for each vector, the data set should not be
    too large if you don't want to wait too long for the result. Also
    the exact search computes a distance matrix that is quadratic in
    the data set size, so there are also memory limitations.
    """

    def __init__(self, N, vectors):
        """
        Performs exact nearest neighbour search on the data set.

        vectors can either be a numpy matrix with all the vectors
        as columns OR a python array containing the individual
        numpy vectors.
        """
        self.N = N
        # We need a dict from vector string representation to index
        self.vector_dict = {}

        # Get numpy array representation of input
        self.vectors = numpy_array_from_list_or_numpy_array(vectors)

        # Build map from vector string representation to vector
        for index in range(self.vectors.shape[1]):
            self.vector_dict[self.__vector_to_string(
                self.vectors[:, index])] = index

        # Get transposed version of vector matrix, so that the rows
        # are the vectors (needed by cdist)
        vectors_t = numpy.transpose(self.vectors)

        # We have to time the exact search
        exact_search_start_time = time.time()

        print '\nStarting exact search...\n'
        # Compute distance matrix
        D = cdist(vectors_t, vectors_t, 'euclidean')

        # For each vector get the closest N neigbbours from distance matrix
        self.closest = []
        for index in range(D.shape[1]):
            # This includes the vector itself, which is what we want
            self.closest.append(scipy.argsort(D[:, index])[:N])

        print '\Done with exact search...\n'

        # Save time needed for exact search
        exact_search_time = time.time() - exact_search_start_time
        # We are interested in the search time per vector
        self.exact_search_time_per_vector = exact_search_time / D.shape[1]

    def perform_experiment(self, engine_list):
        """
        Performs nearest neighbour recall experiments with custom vector data
        for all engines in the specified list.

        Returns self.result contains list of (recall, precision, search_time)
        tuple. All are the averaged values over all request vectors.
        search_time is the average retrieval/search time compared to the
        average exact search time.
        """
        # We will fill this array with measures for all the engines.
        result = []

        # For each engine, first index vectors and then retrieve neighbours
        for engine in engine_list:
            # Clean storage
            engine.clean_all_buckets()
            # Use this to compute average recall
            avg_recall = 0.0
            # Use this to compute average precision
            avg_precision = 0.0
            # Use this to compute average search time
            avg_search_time = 0.0

            # Index vectors and store them
            for index in range(self.vectors.shape[1]):
                #print 'Storing vector %d' % index
                #print vectors[:, index]
                engine.store_vector(self.vectors[:, index], 'testData')

            # Look for N nearest neighbours
            for index in range(self.vectors.shape[1]):
                # Get indices of the real nearest as set
                real_nearest = set(self.closest[index])
                #print 'Real nearest for %d:' % index
                #print real_nearest

                # We have to time the search
                search_time_start = time.time()

                # Get nearest N according to engine
                nearest = engine.neighbours(self.vectors[:, index])

                # Get search time
                search_time = time.time() - search_time_start

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

                # Add to accumulator
                avg_search_time += search_time

            # Normalize recall over data set
            avg_recall = avg_recall / float(self.vectors.shape[1])

            # Normalize precision over data set
            avg_precision = avg_precision / float(self.vectors.shape[1])

            # Normalize search time over data set
            avg_search_time = avg_search_time / float(self.vectors.shape[1])

            # Normalize search time with respect to exact search
            avg_search_time /= self.exact_search_time_per_vector

            #print '\n AVG RECALL = %f\n' % avg_recall
            #print '\n AVG PRECISION = %f\n' % avg_precision
            result.append((avg_recall, avg_precision, avg_search_time))

        # Return (recall, precision, search_time) tuple
        return result

    def __vector_to_string(self, vector):
        """ Returns string representation of vector. """
        return numpy.array_str(vector)

    def __index_of_vector(self, vector):
        """ Returns index of specified vector from test data set. """
        return self.vector_dict[self.__vector_to_string(vector)]
