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
import scipy
import time
import sys

from scipy.spatial.distance import cdist

from nearpy.utils import numpy_array_from_list_or_numpy_array


class RecallPrecisionExperiment(object):
    """
    Performs nearest neighbour recall experiments with custom vector data
    for all engines in the specified list.

    perform_experiment() returns list of (recall, precision, search_time)
    tuple. These are the averaged values over all request vectors. search_time
    is the average retrieval/search time compared to the average exact search
    time.

    coverage_ratio determines how many of the vectors are used as query
    vectors for exact andapproximated search. Because the search comparance
    overhead is quite large, it is best with large data sets (>10000) to
    use a low coverage_ratio (like 0.1) to make the experiment fast. A
    coverage_ratio of 0.1 makes the experiment use 10% of all the vectors
    for querying, that is, it looks for 10% of all vectors for the nearest
    neighbours.
    """

    def __init__(self, N, vectors, coverage_ratio=0.2):
        """
        Performs exact nearest neighbour search on the data set.

        vectors can either be a numpy matrix with all the vectors
        as columns OR a python array containing the individual
        numpy vectors.
        """
        # We need a dict from vector string representation to index
        self.vector_dict = {}
        self.N = N
        self.coverage_ratio = coverage_ratio

        # Get numpy array representation of input
        self.vectors = numpy_array_from_list_or_numpy_array(vectors)

        # Build map from vector string representation to vector
        for index in range(self.vectors.shape[1]):
            self.vector_dict[self.__vector_to_string(
                self.vectors[:, index])] = index

        # Get transposed version of vector matrix, so that the rows
        # are the vectors (needed by cdist)
        vectors_t = numpy.transpose(self.vectors)

        # Determine the indices of query vectors used for comparance
        # with approximated search.
        query_count = numpy.floor(self.coverage_ratio *
                                  self.vectors.shape[1])
        self.query_indices = []
        for k in range(int(query_count)):
            index = numpy.floor(k*(self.vectors.shape[1]/query_count))
            index = min(index, self.vectors.shape[1]-1)
            self.query_indices.append(int(index))

        print('\nStarting exact search (query set size=%d)...\n' % query_count)

        # For each query vector get the closest N neighbours
        self.closest = {}
        self.exact_search_time_per_vector = 0.0

        for index in self.query_indices:

            v = vectors_t[index, :].reshape(1, self.vectors.shape[0])
            exact_search_start_time = time.time()
            D = cdist(v, vectors_t, 'euclidean')
            self.closest[index] = scipy.argsort(D)[0, 1:N+1]

            # Save time needed for exact search
            exact_search_time = time.time() - exact_search_start_time
            self.exact_search_time_per_vector += exact_search_time

        print('\Done with exact search...\n')

        # Normalize search time
        self.exact_search_time_per_vector /= float(len(self.query_indices))

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
            print('Engine %d / %d' % (engine_list.index(engine),
                                      len(engine_list)))

            # Clean storage
            engine.clean_all_buckets()
            # Use this to compute average recall
            avg_recall = 0.0
            # Use this to compute average precision
            avg_precision = 0.0
            # Use this to compute average search time
            avg_search_time = 0.0

            # Index all vectors and store them
            for index in range(self.vectors.shape[1]):
                engine.store_vector(self.vectors[:, index],
                                    'data_%d' % index)

            # Look for N nearest neighbours for query vectors
            for index in self.query_indices:
                # Get indices of the real nearest as set
                real_nearest = set(self.closest[index])

                # We have to time the search
                search_time_start = time.time()

                # Get nearest N according to engine
                nearest = engine.neighbours(self.vectors[:, index])

                # Get search time
                search_time = time.time() - search_time_start

                # For comparance we need their indices (as set)
                nearest = set([self.__index_of_vector(x[0]) for x in nearest])

                # Remove query index from search result to make sure that
                # recall and precision make sense in terms of "neighbours".
                # If ONLY the query vector is retrieved, we want recall to be
                # zero!
                nearest.remove(index)

                # If the result list is empty, recall and precision are 0.0
                if len(nearest) == 0:
                    recall = 0.0
                    precision = 0.0
                else:
                    # Get intersection count
                    inter_count = float(len(real_nearest.intersection(
                        nearest)))

                    # Normalize recall for this vector
                    recall = inter_count/float(len(real_nearest))

                    # Normalize precision for this vector
                    precision = inter_count/float(len(nearest))

                # Add to accumulator
                avg_recall += recall

                # Add to accumulator
                avg_precision += precision

                # Add to accumulator
                avg_search_time += search_time

            # Normalize recall over query set
            avg_recall = avg_recall / float(len(self.query_indices))

            # Normalize precision over query set
            avg_precision = avg_precision / float(len(self.query_indices))

            # Normalize search time over query set
            avg_search_time = avg_search_time / float(len(self.query_indices))

            # Normalize search time with respect to exact search
            avg_search_time /= self.exact_search_time_per_vector

            print('  recall=%f, precision=%f, time=%f' % (avg_recall,
                                                          avg_precision,
                                                          avg_search_time))

            result.append((avg_recall, avg_precision, avg_search_time))

        # Return (recall, precision, search_time) tuple
        return result

    def __vector_to_string(self, vector):
        """ Returns string representation of vector. """
        return numpy.array_str(vector)

    def __index_of_vector(self, vector):
        """ Returns index of specified vector from test data set. """
        return self.vector_dict[self.__vector_to_string(vector)]
