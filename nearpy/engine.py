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

from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter
from nearpy.distances import EuclideanDistance
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

    def __init__(self, dim, lshashes=[RandomBinaryProjections('default', 10)],
                 distance=EuclideanDistance(),
                 vector_filters=[NearestFilter(10)],
                 storage=MemoryStorage()):
        """ Keeps the configuration. """
        self.lshashes = lshashes
        self.distance = distance
        self.vector_filters = vector_filters
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
                self.storage.store_vector(lshash.hash_name, bucket_key,
                                          v, data)

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
            for bucket_key in lshash.hash_vector(v):
                bucket_content = self.storage.get_bucket(lshash.hash_name,
                                                         bucket_key)
                candidates.extend(bucket_content)

        # Apply distance implementation if specified
        if self.distance:
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
