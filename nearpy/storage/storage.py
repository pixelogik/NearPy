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


class Storage(object):
    """ Interface for storage adapters. """

    def store_vector(self, hash_name, bucket_key, v, data):
        """
        Stores vector and JSON-serializable data in bucket with specified key.
        """
        raise NotImplementedError

    def store_many_vectors(self, hash_name, bucket_keys, vs, data):
        """
        Store a batch of vectors.
        Stores vector and JSON-serializable data in bucket with specified key.
        """
        raise NotImplementedError

    def get_all_bucket_keys(self, hash_name):
        """
        Returns all bucket keys for the given hash as iterable of strings
        """
        raise NotImplementedError


    def delete_vector(self, hash_name, bucket_keys, data):
        """
        Deletes vector and JSON-serializable data in buckets with specified keys.
        """
        raise NotImplementedError

    def get_bucket(self, hash_name, bucket_key):
        """
        Returns bucket content as list of tuples (vector, data).
        """
        raise NotImplementedError

    def clean_buckets(self, hash_name):
        """
        Removes all buckets and their content.
        """
        raise NotImplementedError

    def clean_all_buckets(self):
        """
        Removes all buckets and their content.
        """
        raise NotImplementedError

    def store_hash_configuration(self, lshash):
        """
        Stores hash configuration
        """
        raise NotImplementedError

    def load_hash_configuration(self, hash_name):
        """
        Loads and returns hash configuration
        """
        raise NotImplementedError
