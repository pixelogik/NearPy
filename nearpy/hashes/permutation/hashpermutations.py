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
from nearpy.hashes.permutation import Permutation


class HashPermutations(LSHash):
    """
    By: Xing Shi (xingshi@usc.edu)

    This meta-hash performs hash permutations on binary bucket keys.
    You use this just like every other LSHash implementation and
    add the actual binary hashes you want to use via the add_child_hash
    method. Each child hash will be used separatly.

    After all vectors have been indexed you have to call build_permuted_index
    to generate the permuted index.

    So to use this you have to do the following steps:

    1. Create HashPermutations instance and use it in the Engine constructor
    2. Add your binary hashes as child hashes by calling add_child_hash()
    3. Store your vectors using the Engine
    4. Build the permuted index by calling build_permuted_index()
    5. Now when you query the permuted index is used

    If you are adding more vectors afterwards you can update the permuted index
    by calling build_permuted_index() again and again.

    """

    def __init__(self, hash_name):
        """ Just keeps the name. """
        super(HashPermutations, self).__init__(hash_name)
        self.permutation = Permutation()
        self.child_hashes = []
        self.dim = None

    def reset(self, dim):
        """ Resets / Initializes the hash for the specified dimension. """
        self.dim = dim
        # Reset all child hashes
        for child_hash in self.child_hashes:
            child_hash['hash'].reset(dim)
            child_hash['bucket_keys'] = {}

    def hash_vector(self, v, querying=False):
        """
        Hashes the vector and returns the bucket key as string.
        """

        bucket_keys = []

        if querying:
            # If we are querying, use the permuted indexes to get bucket keys
            for child_hash in self.child_hashes:
                lshash = child_hash['hash']
                # Make sure the permuted index for this hash is existing
                if not lshash.hash_name in self.permutation.permutedIndexs:
                    raise AttributeError('Permuted index is not existing for hash with name %s' % lshash.hash_name)

                # Get regular bucket keys from hash
                for bucket_key in lshash.hash_vector(v, querying):
                    #print 'Regular bucket key %s' % bucket_key
                    # Get neighbour keys from permuted index
                    neighbour_keys = self.permutation.get_neighbour_keys(lshash.hash_name,bucket_key)
                    # Add them to result, but prefix with hash name
                    for n in neighbour_keys:
                        bucket_keys.append(lshash.hash_name+'_'+n)

        else:
            # If we are indexing (storing) just use child hashes without permuted index
            for child_hash in self.child_hashes:
                lshash = child_hash['hash']

                # Get regular bucket keys from hash
                for bucket_key in lshash.hash_vector(v, querying):
                    # Register bucket key in child hash dict
                    child_hash['bucket_keys'][bucket_key] = bucket_key
                    # Append bucket key to result prefixed with child hash name
                    bucket_keys.append(lshash.hash_name+'_'+bucket_key)

        # Return all the bucket keys
        return bucket_keys

    def get_config(self):
        """
        Returns pickle-serializable configuration struct for storage.
        """
        return {
            'hash_name': self.hash_name,
            'dim': self.dim
        }

    def apply_config(self, config):
        """
        Applies config
        """
        self.hash_name = config['hash_name']
        self.dim = config['dim']

    def add_child_hash(self, child_hash, permute_config):
        """
        Adds specified child hash with specified configuration.
        The hash must be one of the binary types.

        permute_config is a dict in the following format:
        permute_config = { "num_permutation":50,
                           "beam_size":10,
                           "num_neighbour":100
                         }
        """

        # Hash must generate binary keys
        if not (isinstance(child_hash,PCABinaryProjections) or isinstance(child_hash,RandomBinaryProjections) or isinstance(child_hash,RandomBinaryProjectionTree)):
            raise ValueError('Child hashes must generate binary keys')

        # Add both hash and config to array of child hashes. Also we are going to
        # accumulate used bucket keys for every hash in order to build the permuted index
        self.child_hashes.append({'hash': child_hash, 'config': permute_config, 'bucket_keys': {}})

    def build_permuted_index(self):
        """
        Build PermutedIndex for all your binary hashings.
        PermutedIndex would be used to find the neighbour bucket key
        in terms of Hamming distance. Permute_configs is nested dict
        in the following format:
        permuted_config = {"<hash_name>":
                           { "num_permutation":50,
                             "beam_size":10,
                             "num_neighbour":100 }
                          }
        """

        for child_hash in self.child_hashes:
            # Get config values for child hash
            config = child_hash['config']
            num_permutation = config['num_permutation']
            beam_size = config['beam_size']
            num_neighbour = config['num_neighbour']

            # Get used buckets keys for child hash
            bucket_keys = child_hash['bucket_keys'].keys()

            # Get actual child hash
            lshash = child_hash['hash']

            # Compute permuted index for this hash
            self.permutation.build_permuted_index(lshash,bucket_keys,num_permutation,beam_size,num_neighbour)


