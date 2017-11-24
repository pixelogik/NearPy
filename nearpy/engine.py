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
from nearpy.filters import NearestFilter, UniqueFilter
from nearpy.distances import EuclideanDistance
from nearpy.distances import CosineDistance
from nearpy.storage import MemoryStorage, MongoStorage
from nearpy.utils.utils import unitvec


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
                 fetch_vector_filters=None,
                 vector_filters=None,
                 storage=None):
        """ Keeps the configuration. """
        if lshashes is None: lshashes = [RandomBinaryProjections('default', 10)]
        self.lshashes = lshashes
        if distance is None: distance = CosineDistance()
        self.distance = distance
        if vector_filters is None: vector_filters = [NearestFilter(10)]
        self.vector_filters = vector_filters
        if fetch_vector_filters is None: fetch_vector_filters = [UniqueFilter()]
        self.fetch_vector_filters = fetch_vector_filters
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
        # We will store the normalized vector (used during retrieval)
        nv = unitvec(v)
        # Store vector in each bucket of all hashes
        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(v):
                #print 'Storying in bucket %s one vector' % bucket_key
                self.storage.store_vector(lshash.hash_name, bucket_key,
                                          nv, data)

    def delete_vector(self, data, v=None):
        """
        Deletes vector v and his id (data) in all matching buckets in the storage.
        The data argument must be JSON-serializable.
        """

        # Delete data id in each hashes
        for lshash in self.lshashes:
            if v is None:
                keys = self.storage.get_all_bucket_keys(lshash.hash_name)
            else:
                keys = lshash.hash_vector(v)
            self.storage.delete_vector(lshash.hash_name, keys, data)

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
        candidates = self._get_candidates(v)
        return len(candidates)

    def neighbours(self, v,
                   distance=None,
                   fetch_vector_filters=None,
                   vector_filters=None,
                   mongo_fetch_vector_filters=None):
        """
        Hashes vector v, collects all candidate vectors from the matching
        buckets in storage, applys the (optional) distance function and
        finally the (optional) filter function to construct the returned list
        of either (vector, data, distance) tuples or (vector, data) tuples.
        """

        # Collect candidates from all buckets from all hashes
        candidates = self._get_candidates(v, mongo_fetch_vector_filters)
        # print 'Candidate count is %d' % len(candidates)

        # Apply fetch vector filters if specified and return filtered list
        if fetch_vector_filters:
            candidates = self._apply_filter(fetch_vector_filters,
                                            candidates)

        # Apply distance implementation if specified
        if not distance:
            distance = self.distance
        candidates = self._append_distances(v, distance, candidates)

        # Apply vector filters if specified and return filtered list
        if not vector_filters:
            vector_filters = self.vector_filters
        candidates = self._apply_filter(vector_filters, candidates)

        # If there is no vector filter, just return list of candidates
        return candidates


    def _get_candidates(self, v, mongo_fetch_vector_filters=None):
        """ Collect candidates from all buckets from all hashes """
        candidates = []
        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(v, querying=True):
                if mongo_fetch_vector_filters:
                    bucket_content = self.storage.get_bucket(
                        lshash.hash_name,
                        bucket_key,
                        mongo_fetch_vector_filters
                    )
                else:
                    bucket_content = self.storage.get_bucket(
                        lshash.hash_name,
                        bucket_key,
                    )
                #print 'Bucket %s size %d' % (bucket_key, len(bucket_content))
                candidates.extend(bucket_content)
        return candidates


    def _apply_filter(self, filters, candidates):
        """ Apply vector filters if specified and return filtered list """
        if filters:
            filter_input = candidates
            for fetch_vector_filter in filters:
                filter_input = fetch_vector_filter.filter_vectors(filter_input)

            return filter_input
        else:
            return candidates

    def _append_distances(self, v, distance, candidates):
        """ Apply distance implementation if specified """
        if distance:
            # Normalize vector (stored vectors are normalized)
            nv = unitvec(v)
            candidates = [(x[0], x[1], self.distance.distance(x[0], nv)) for x
                            in candidates]

        return candidates

    def clean_all_buckets(self):
        """ Clears buckets in storage (removes all vectors and their data). """
        self.storage.clean_all_buckets()

    def clean_buckets(self, hash_name):
        """ Clears buckets in storage (removes all vectors and their data). """
        self.storage.clean_buckets(hash_name)
