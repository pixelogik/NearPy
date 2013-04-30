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

import redis

from storage import Storage


class RedisStorage(Storage):
    """ Storage using redis. """

    def __init__(self, redis_object):
        """ Uses specified redis object for storage. """
        self.redis_object = redis_object

    def store_vector(self, hash_name, bucket_key, v, data):
        """
        Stores vector and JSON-serializable data in bucket with specified key.
        """
        redis_key = 'nearpy_%s_%s' % (hash_name, bucket_key)
        self.redis_object.rpush(redis_key, (v, data))

    def get_bucket(self, hash_name, bucket_key):
        """
        Returns bucket content as list of tuples (vector, data).
        """
        redis_key = 'nearpy_%s_%s' % (hash_name, bucket_key)
        return self.redis_object.lrange(0, -1)

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
