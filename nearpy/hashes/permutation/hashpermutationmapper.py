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

from nearpy.hashes import LSHash, RandomBinaryProjections, PCABinaryProjections, RandomBinaryProjectionTree


class HashPermutationMapper(LSHash):
    """
    This meta-hash performs permutations on binary bucket keys
    and keeps a dictionary for this mapping.

    You use this just like every other LSHash implementation and
    add the actual binary hashes you want to use via the add_child_hash
    method. Each child hash will be used separatly.

    So to use this you have to do the following steps:

    1. Create HashPermutationMapper instance and use it in the Engine constructor
    2. Add your binary hashes as child hashes by calling add_child_hash()
    3. Store your vectors using the Engine
    4. Now when you query the bucket key map is used
    """

    def __init__(self, hash_name):
        """ Just keeps the name. """
        super(HashPermutationMapper, self).__init__(hash_name)
        self.child_hashes = []
        self.dim = None
        self.bucket_key_map = {}

    def reset(self, dim):
        """ Resets / Initializes the hash for the specified dimension. """
        self.dim = dim
        self.bucket_key_map = {}
        # Reset all child hashes
        for child_hash in self.child_hashes:
            child_hash.reset(dim)

    def permuted_keys(self, key):
        result = []
        for j in range(len(key)):
            bits = list(key)
            bits[j] = '1' if key[j]=='0' else '0'
            result.append(''.join(bits))
        return result

    def hash_vector(self, v, querying=False):
        """
        Hashes the vector and returns the bucket key as string.
        """

        bucket_keys = []

        if querying:
            # If we are querying, use the bucket key map
            for lshash in self.child_hashes:
                # Get regular bucket keys from hash
                for bucket_key in lshash.hash_vector(v, querying):
                    prefixed_key = lshash.hash_name+'_'+bucket_key
                    # Get entries from map (bucket keys with hamming distance of 1)
                    if prefixed_key in self.bucket_key_map:
                        bucket_keys.extend(self.bucket_key_map[prefixed_key].keys())
        else:
            # If we are indexing (storing) just use child hashes without permuted index
            for lshash in self.child_hashes:
                # Get regular bucket keys from hash
                for bucket_key in lshash.hash_vector(v, querying):
                    # Get permuted keys
                    perm_keys = self.permuted_keys(bucket_key)
                    # Put extact hit key into list
                    perm_keys.append(bucket_key)

                    # Append key for storage (not the permutations)
                    bucket_keys.append(lshash.hash_name+'_'+bucket_key)

                    # For every permutation register all the variants
                    for perm_key in perm_keys:
                        prefixed_key = lshash.hash_name+'_'+perm_key

                        # Make sure dictionary exists
                        if not prefixed_key in self.bucket_key_map:
                            self.bucket_key_map[prefixed_key] = {}

                        for variant in perm_keys:
                            prefixed_variant = lshash.hash_name+'_'+variant
                            self.bucket_key_map[prefixed_key][prefixed_variant] = 1

        # Return all the bucket keys
        return bucket_keys

    def get_config(self):
        """
        Returns pickle-serializable configuration struct for storage.
        """
        return {
            'hash_name': self.hash_name,
            'dim': self.dim,
            'bucket_key_map': self.bucket_key_map
        }

    def apply_config(self, config):
        """
        Applies config
        """
        self.hash_name = config['hash_name']
        self.dim = config['dim']
        self.bucket_key_map = config['bucket_key_map']

    def add_child_hash(self, child_hash):
        """
        Adds specified child hash.

        The hash must be one of the binary types.
        """

        # Hash must generate binary keys
        if not (isinstance(child_hash,PCABinaryProjections) or isinstance(child_hash,RandomBinaryProjections) or isinstance(child_hash,RandomBinaryProjectionTree)):
            raise ValueError('Child hashes must generate binary keys')

        # Add both hash and config to array of child hashes. Also we are going to
        # accumulate used bucket keys for every hash in order to build the permuted index
        self.child_hashes.append(child_hash)



