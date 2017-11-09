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

import numpy
import scipy
try:
    import cPickle as pickle
except ImportError:
    import pickle

from future.builtins import bytes
from nearpy.storage.storage import Storage


class MongoStorage(Storage):
    """ Storage using MongoDB. """

    def __init__(self, mongo_object):
        """ Uses specified pymongo object for storage. """
        self.mongo_object = mongo_object

    def store_vector(self, hash_name, bucket_key, v, data):
        """
        Stores vector and JSON-serializable data in MongoDB with specified key.
        """
        mongo_key = self._format_mongo_key(hash_name, bucket_key)

        val_dict = {}
        val_dict['lsh'] = mongo_key

        # Depending on type (sparse or not) fill value dict
        if scipy.sparse.issparse(v):
            # Make sure that we are using COO format (easy to handle)
            if not scipy.sparse.isspmatrix_coo(v):
                v = scipy.sparse.coo_matrix(v)

            # Construct list of [index, value] items,
            # one for each non-zero element of the sparse vector
            encoded_values = []

            for k in range(v.data.size):
                row_index = v.row[k]
                value = v.data[k]
                encoded_values.append([int(row_index), value])

            val_dict['sparse'] = 1
            val_dict['nonzeros'] = encoded_values
            val_dict['dim'] = v.shape[0]
        else:
            # Make sure it is a 1d vector
            v = numpy.reshape(v, v.shape[0])
            val_dict['vector'] = v.tostring()

        val_dict['dtype'] = v.dtype.name

        # Add data if set
        if data is not None:
            if isinstance(data, str):
                val_dict['data'] = data
            elif isinstance(data, dict):
                # Assume using Python 3.5
                val_dict = {**val_dict, **data}

        # Push JSON representation of dict to end of bucket list
        self.mongo_object.insert_one(val_dict)

    def _format_mongo_key(self, hash_name, bucket_key):
        return '{}{}'.format(self._format_hash_prefix(hash_name), bucket_key)

    def _format_hash_prefix(self, hash_name):
        return "nearpy_{}_".format(hash_name)

    def get_all_bucket_keys(self, hash_name):
        prefix_len = len(self._format_hash_prefix(hash_name))
        return list(self.mongo_object.find(
            {'lsh': {'$regex': self._format_hash_prefix(hash_name)}}))

    def _get_bucket_rows(self, hash_name, bucket_key,
                         mongo_fetch_vector_filters=None):
        lsh_key = self._format_mongo_key(hash_name, bucket_key)
        if mongo_fetch_vector_filters:
            return self.mongo_object.find({'lsh': lsh_key,
                                           **mongo_fetch_vector_filters})
        else:
            return self.mongo_object.find({'lsh': lsh_key})

    def delete_vector(self, hash_name, bucket_keys, data):
        """
        Deletes vector and JSON-serializable data in buckets with specified keys.
        """
        lsh_keys = [self._format_mongo_key(hash_name, key)
                    for key in bucket_keys]
        self.mongo_object.remove({'lsh': {'$in': lsh_keys},
                                  'data': data})

    def get_bucket(self, hash_name, bucket_key, mongo_fetch_vector_filters=None):
        """
        Returns bucket content as list of tuples (vector, data).
        """
        results = []
        for row in self._get_bucket_rows(hash_name, bucket_key,
                                         mongo_fetch_vector_filters):
            val_dict = row
            # Depending on type (sparse or not) reconstruct vector
            if 'sparse' in val_dict:

                # Fill these for COO creation
                row  = []
                col  = []
                data = []

                # For each non-zero element, append values
                for e in val_dict['nonzeros']:
                    row.append(e[0]) # Row index
                    data.append(e[1]) # Value
                    col.append(0) # Column index (always 0)

                # Create numpy arrays for COO creation
                coo_row = numpy.array(row, dtype=numpy.int32)
                coo_col = numpy.array(col, dtype=numpy.int32)
                coo_data = numpy.array(data)

                # Create COO sparse vector
                vector = scipy.sparse.coo_matrix( (coo_data,(coo_row,coo_col)), shape=(val_dict['dim'],1) )
                # Add data to result tuple, if present
                [val_dict.pop(k) for k in 
                 ['vector', 'nonzeros', 'sparse', 'dim', 'dtype' '_id']]
                results.append((vector, val_dict))

            else:
                vector = numpy.fromstring(val_dict['vector'],
                                          dtype=val_dict['dtype'])
                [val_dict.pop(k) for k in ['vector', 'dtype', '_id']]
                # Add data to result tuple, if present
                results.append((vector, val_dict))

        return results

    def clean_buckets(self, hash_name):
        """
        Removes all buckets and their content for specified hash.
        """
        self.mongo_object.remove(
            {'lsh': {'$regex': self._format_hash_prefix(hash_name)}})

    def clean_all_buckets(self):
        """
        Removes all buckets from all hashes and their content.
        """
        self.mongo_object.remove(
            {'lsh': {'$regex': 'nearpy_'}})

    def store_hash_configuration(self, lshash):
        """
        Stores hash configuration
        """
        self.mongo_object.insert_one(
            {'hash_conf_name': lshash.hash_name+'_conf',
             'hash_configuration': pickle.dumps(lshash.get_config())
             }
        )

    def load_hash_configuration(self, hash_name):
        """
        Loads and returns hash configuration
        """
        conf = self.mongo_object.find_one(
            {'hash_conf_name': hash_name + '_conf'}
        )
        return pickle.loads(conf['hash_configuration']) if conf is not None\
            else None
