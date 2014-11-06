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

import json

import numpy as np

from nearpy.hashes import RandomBinaryProjections
from nearpy.hashes import PCABinaryProjections
from nearpy.hashes import RandomBinaryProjectionTree
from nearpy.filters import NearestFilter
from nearpy.distances import EuclideanDistance
from nearpy.distances import CosineDistance
from nearpy.storage import MemoryStorage


class Engine(object):
    """
    Objects with this type perform the actual ANN search and vector indexing.
    They can be configured by selecting implementations of the Hash, Distance,
    Filter and Storage interfaces.

    There are four different modes of the engine:

        (1) Full configuration - All arguments are defined.
                In this case the distance and vector filters
                are applied to the bucket contents to deliver the
                resulting list of filtered (vector, data, distance) tuples.
        (2) No distance - The distance argument is None.
                In this case only the vector filters are applied to
                the bucket contents and the result is a list of
                filtered (vector, data) tuples.
        (3) No vector filter - The vector_filter argument is None.
                In this case only the distance is applied to
                the bucket contents and the result is a list of
                unsorted/unfiltered (vector, data, distance) tuples.
        (4) No vector filter and no distance - Both arguments are None.
                In this case the result is just the content from the
                buckets as an unsorted/unfiltered list of (vector, data)
                tuples.
    """

    def __init__(self, dim, lshashes=None,
                 distance=None,
                 vector_filters=None,
                 storage=None):
        """ Keeps the configuration. """
        if lshashes is None: lshashes = [RandomBinaryProjections('default', 10)]
        self.lshashes = lshashes
        if distance is None: distance = EuclideanDistance()
        self.distance = distance
        if vector_filters is None: vector_filters = [NearestFilter(10)]
        self.vector_filters = vector_filters
        if storage is None: storage = MemoryStorage()
        self.storage = storage

        # Initialize all hashes for the data space dimension.
        for lshash in self.lshashes:
            lshash.reset(dim)

    def store_vector(self, v, data=None):
        """
        Hashes vector v and stores it in all matching buckets in the storage.
        The data argument must be JSON-serializable. It is stored with the
        vector and will be returned in search results.
        """
        # Store vector in each bucket of all hashes
        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(v):
                #print 'Storying in bucket %s one vector' % bucket_key
                self.storage.store_vector(lshash.hash_name, bucket_key,
                                          v, data)


    def candidate_count(self, v):
        """
        Returns candidate count for nearest neighbour search for specified vector.
        The candidate count is the count of vectors taken from all buckets the
        specified vector is projected onto.

        Use this method to check if your hashes are configured good. High candidate
        counts makes querying slow.

        For example if you always want to retrieve 20 neighbours but the candidate
        count is 1000 or something you have to change the hash so that each bucket
        has less entries (increase projection count for example).
        """
        # Collect candidates from all buckets from all hashes
        candidates = []
        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(v, querying=True):
                bucket_content = self.storage.get_bucket(lshash.hash_name,
                                                         bucket_key)
                #print 'Bucket %s size %d' % (bucket_key, len(bucket_content))
                candidates.extend(bucket_content)

        return len(candidates)

    def neighbours(self, v):
        """
        Hashes vector v, collects all candidate vectors from the matching
        buckets in storage, applys the (optional) distance function and
        finally the (optional) filter function to construct the returned list
        of either (vector, data, distance) tuples or (vector, data) tuples.
        """
        # Collect candidates from all buckets from all hashes
        candidates = []
        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(v, querying=True):
                bucket_content = self.storage.get_bucket(lshash.hash_name,
                                                         bucket_key)
                #print 'Bucket %s size %d' % (bucket_key, len(bucket_content))
                candidates.extend(bucket_content)

        #print 'Candidate count is %d' % len(candidates)

        # Apply distance implementation if specified
        if self.distance:
            if isinstance(self.distance,CosineDistance):
               # Use distance matrix computation
                candidate_matrix = np.zeros((len(candidates),self.lshashes[0].dim))
                for i in xrange(len(candidates)):
                    candidate_matrix[i] = candidates[i][0]
                    npv = np.array(v)
                    npv = npv.reshape((1,len(npv)))
                    cos_dists = self.distance.distance_matrix(candidate_matrix,npv)
                    cos_dists = cos_dists.reshape((-1,))
                candidates = [(candidates[i][0], candidates[i][1], cos_dists[i]) for i in xrange(len(candidates))]
            else:
                candidates = [(x[0], x[1], self.distance.distance(x[0], v)) for x
                              in candidates]


        # Apply vector filters if specified and return filtered list
        if self.vector_filters:
            filter_input = candidates
            for vector_filter in self.vector_filters:
                filter_input = vector_filter.filter_vectors(filter_input)
            # Return output of last filter
            return filter_input

        # If there is no vector filter, just return list of candidates
        return candidates

    def clean_all_buckets(self):
        """ Clears buckets in storage (removes all vectors and their data). """
        self.storage.clean_all_buckets()

    def clean_buckets(self, hash_name):
        """ Clears buckets in storage (removes all vectors and their data). """
        self.storage.clean_buckets(hash_name)
