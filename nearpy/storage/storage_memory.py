# -*- coding: utf-8 -*-

# Copyright (c) 2013 Ole Krause-Sparmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
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

from future.utils import viewkeys
from nearpy.storage.storage import Storage


class MemoryStorage(Storage):
    """ Simple implementation using python dicts. """

    def __init__(self):
        self.buckets = {}
        self.hash_configs = {}

    def store_vector(self, hash_name, bucket_key, v, data):
        """
        Stores vector and JSON-serializable data in bucket with specified key.
        """

        if not hash_name in self.buckets:
            self.buckets[hash_name] = {}

        if not bucket_key in self.buckets[hash_name]:
            self.buckets[hash_name][bucket_key] = []
        self.buckets[hash_name][bucket_key].append((v, data))

    def store_many_vectors(self, hash_name, bucket_keys, vs, data):
        """
        Store a batch of vectors.
        Stores vector and JSON-serializable data in bucket with specified key.
        """
        for idx, v in enumerate(vs):
            if data is not None:
                self.store_vector(hash_name, bucket_keys[idx], v, data[idx])
            else:
                self.store_vector(hash_name, bucket_keys[idx], v, data)

    def get_all_bucket_keys(self, hash_name):
        return viewkeys(self.buckets[hash_name])

    def delete_vector(self, hash_name, bucket_keys, data):
        """
        Deletes vector and JSON-serializable data in buckets with specified keys.
        """
        for key in bucket_keys:
            bucket = self.get_bucket(hash_name, key)
            bucket[:] = [(v, id_data) for v, id_data
                         in bucket if id_data != data]

    def get_bucket(self, hash_name, bucket_key):
        """
        Returns bucket content as list of tuples (vector, data).
        """
        if hash_name in self.buckets:
            if bucket_key in self.buckets[hash_name]:
                return self.buckets[hash_name][bucket_key]
        return []

    def clean_buckets(self, hash_name):
        """
        Removes all buckets and their content for specified hash.
        """
        self.buckets[hash_name] = {}

    def clean_all_buckets(self):
        """
        Removes all buckets from all hashes and their content.
        """
        self.buckets = {}

    def store_hash_configuration(self, lshash):
        """
        Stores hash configuration
        """
        self.hash_configs[lshash.hash_name] = lshash.get_config()

    def load_hash_configuration(self, hash_name):
        """
        Loads and returns hash configuration
        """
        return self.hash_configs.get(hash_name)
