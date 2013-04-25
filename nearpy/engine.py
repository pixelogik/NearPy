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

from hashes import RandomBinaryProjections
from filters import VectorFilter
from distances import Distance
from storage import MemoryStorage

class Engine(object):
    """
    Objects with this type perform the actual ANN search and vector indexing.
    They can be configured by selecting implementations of the Hash, Distance,
    Filter and Storage interfaces.
    """

    def __init__(self, dim, lshash=RandomBinaryProjections(10),
                 distance=Distance(), vector_filter=VectorFilter(),
                 storage=MemoryStorage()):
        """ Keeps the configuration. """
        self.lshash = lshash
        self.distance = distance
        self.vector_filter = vector_filter
        self.storage = storage

        # Initialize hash for the data space dimension.
        self.lshash.reset(dim)

    def store_vector(self, v, data=None):
        """
        Hashes vector v and stores it in all matching buckets in the storage.
        The data argument must be JSON-serializable. It is stored with the
        vector and will be returned in search results.
        """
        # Get list of bucket keys from hash
        bucket_keys = self.lshash.hash_vector(v)
        # Stpre vector and data in each bucket
        for bucket_key in bucket_keys:
            self.storage.store_vector(bucket_key, v, data)

    def neighbours(self, v):
        """
        Hashes vector v, collects all candidate vectors from the matching
        buckets in storage, applys the distance function and finally the filter
        function to construct the returned list of (vector, data_key) items.
        """
        pass








