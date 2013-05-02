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
import json
import numpy

from nearpy.storage.storage import Storage


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

        # Make sure it is a 1d vector
        v = numpy.reshape(v, v.shape[0])

        val_dict = {'vector': v.tolist()}
        if data:
            val_dict['data'] = data

        self.redis_object.rpush(redis_key, json.dumps(val_dict))

    def get_bucket(self, hash_name, bucket_key):
        """
        Returns bucket content as list of tuples (vector, data).
        """
        redis_key = 'nearpy_%s_%s' % (hash_name, bucket_key)
        items = self.redis_object.lrange(redis_key, 0, -1)
        results = []
        for item_str in items:
            val_dict = json.loads(item_str)
            vector = numpy.fromiter(val_dict['vector'], dtype=numpy.float64)
            if 'data' in val_dict:
                results.append((vector, val_dict['data']))
            else:
                results.append((vector, None))

        return results

    def clean_buckets(self, hash_name):
        """
        Removes all buckets and their content for specified hash.
        """
        bucket_keys = self.redis_object.keys(pattern='nearpy_%s_*' % hash_name)
        for bucket_key in bucket_keys:
            self.redis_object.delete(bucket_key)

    def clean_all_buckets(self):
        """
        Removes all buckets from all hashes and their content.
        """
        bucket_keys = self.redis_object.keys(pattern='nearpy_*')
        for bucket_key in bucket_keys:
            self.redis_object.delete(bucket_key)
