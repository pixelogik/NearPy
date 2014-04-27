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


class DistanceRatioExperiment(object):
    """
    Performs nearest neighbour experiments with custom vector data
    for all engines in the specified list.

    Distance ratio is the average distance of retrieved approximated
    neighbours, that are outside the radius of the real nearest N
    neighbours,  with respect to this radius.

    Let R be the radius of the real N nearest neighbours around the
    query vector. Then a distance ratio of 1.0 means, that the average
    approximated nearest neighbour is 2*R away from the query point.
    A distance_ratio of 0.0 means, all approximated neighbours are
    within the radius.

    This is a much better performance measure for ANN than recall or precision,
    because in ANN we are interested in spatial relations between query vector
    and the results.

    perform_experiment() returns list of (distance_ratio, result_size,
    search_time) tuple. These are the averaged values over all request
    vectors. search_time is the average retrieval/search time compared
    to the average exact search time. result_size is the size of the
    retrieved set of approximated neighbours.

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

        # For each query vector get radius of closest N neighbours
        self.nearest_radius = {}
        self.exact_search_time_per_vector = 0.0

        for index in self.query_indices:

            v = vectors_t[index, :].reshape(1, self.vectors.shape[0])
            exact_search_start_time = time.time()
            D = cdist(v, vectors_t, 'euclidean')

            # Get radius of closest N neighbours
            self.nearest_radius[index] = scipy.sort(D)[0, N]

            # Save time needed for exact search
            exact_search_time = time.time() - exact_search_start_time
            self.exact_search_time_per_vector += exact_search_time

        print('\Done with exact search...\n')

        # Normalize search time
        self.exact_search_time_per_vector /= float(len(self.query_indices))

    def perform_experiment(self, engine_list):
        """
        Performs nearest neighbour experiments with custom vector data
        for all engines in the specified list.

        Returns self.result contains list of (distance_ratio, search_time)
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
            # Use this to compute average distance_ratio
            avg_distance_ratio = 0.0
            # Use this to compute average result set size
            avg_result_size = 0.0
            # Use this to compute average search time
            avg_search_time = 0.0

            # Index all vectors and store them
            for index in range(self.vectors.shape[1]):
                engine.store_vector(self.vectors[:, index],
                                    'data_%d' % index)

            # Look for N nearest neighbours for query vectors
            for index in self.query_indices:
                # We have to time the search
                search_time_start = time.time()

                # Get nearest N according to engine
                nearest = engine.neighbours(self.vectors[:, index])

                # Get search time
                search_time = time.time() - search_time_start

                # Get average distance ratio (with respect to radius
                # of real N closest neighbours)
                distance_ratio = 0.0
                for n in nearest:
                    # If the vector is outside the real neighbour radius
                    if n[2] > self.nearest_radius[index]:
                        # Compute distance to real neighbour radius
                        d = (n[2] - self.nearest_radius[index])
                        # And normalize it. 1.0 means: distance to
                        # real neighbour radius is identical to radius
                        d /= self.nearest_radius[index]
                        # If all neighbours are in the radius, the
                        # distance ratio is 0.0
                        distance_ratio += d
                # Normalize distance ratio over all neighbours
                distance_ratio /= len(nearest)

                # Add to accumulator
                avg_distance_ratio += distance_ratio

                # Add to accumulator
                avg_result_size += len(nearest)

                # Add to accumulator
                avg_search_time += search_time

            # Normalize distance ratio over query set
            avg_distance_ratio /= float(len(self.query_indices))

            # Normalize avg result size
            avg_result_size /= float(len(self.query_indices))

            # Normalize search time over query set
            avg_search_time = avg_search_time / float(len(self.query_indices))

            # Normalize search time with respect to exact search
            avg_search_time /= self.exact_search_time_per_vector

            print('  distance_ratio=%f, result_size=%f, time=%f' % (avg_distance_ratio,
                                                                    avg_result_size,
                                                                    avg_search_time))

            result.append((avg_distance_ratio, avg_result_size, avg_search_time))

        return result

    def __vector_to_string(self, vector):
        """ Returns string representation of vector. """
        return numpy.array_str(vector)

    def __index_of_vector(self, vector):
        """ Returns index of specified vector from test data set. """
        return self.vector_dict[self.__vector_to_string(vector)]
