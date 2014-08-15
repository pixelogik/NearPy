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
import scipy.sparse

from nearpy.hashes.lshash import LSHash


class RandomBinaryProjectionTreeNode(object):

    def __init__(self):
        # Count of vectors stored in this subtree
        self.vector_count = 0
        # Dict containing the two childs (one for '0', one for '1')
        self.childs = {}
        self.bucket_key = None

    def insert_entry_for_bucket(self, bucket_key, tree_depth):
        """
        Increases counter for specified bucket_key (leaf of tree) and
        also increases counters along the way from the root to the leaf.
        """

        # First increase vector count of this subtree
        self.vector_count = self.vector_count + 1

        if tree_depth < len(bucket_key):
            hash_char = bucket_key[tree_depth]

            # This is not the leaf so continue down into the tree towards the leafes

            #print 'hash_char=%c' % hash_char

            # If child node for this character ('0' or '1') is not existing yet, create it
            if not hash_char in self.childs:
                #print 'Creating child for %c' % hash_char
                self.childs[hash_char] = RandomBinaryProjectionTreeNode()

            # Continue on the child
            self.childs[hash_char].insert_entry_for_bucket(bucket_key, tree_depth+1)
        else:
            # This is a leaf, so keep the bucket key
            #print 'Inserting leaf %s(%s, %d), count is %d' % (bucket_key, self.bucket_key, tree_depth, self.vector_count)
            if not self.bucket_key is None:
                if self.bucket_key != bucket_key:
                    raise AttributeError
            self.bucket_key = bucket_key

    def collect_all_bucket_keys(self):
        """
        Just collects all buckets keys from subtree
        """
        if len(self.childs) == 0:
            # This is a leaf so just return the bucket key (we reached the bucket leaf)
            #print 'Returning (collect) leaf bucket key %s with %d vectors' % (self.bucket_key, self.vector_count)
            return [self.bucket_key]

        # Not leaf, return results of childs
        result = []
        for child in self.childs.values():
            result = result + child.collect_all_bucket_keys()

        return result

    def bucket_keys_to_guarantee_result_set_size(self, bucket_key, N, tree_depth):
        """
        Returns list of bucket keys based on the specified bucket key
        and minimum result size N.
        """

        if tree_depth == len(bucket_key):
            #print 'Returning leaf bucket key %s with %d vectors' % (self.bucket_key, self.vector_count)
            # This is a leaf so just return the bucket key (we reached the bucket leaf)
            return [self.bucket_key]

        # If not leaf, this is a subtree node.

        hash_char = bucket_key[tree_depth]
        if hash_char == '0':
            other_hash_char = '1'
        else:
            other_hash_char = '0'

        # Check if child has enough results
        if hash_char in self.childs:
            if self.childs[hash_char].vector_count < N:
                # If not combine buckets of both child subtrees
                listA = self.childs[hash_char].collect_all_bucket_keys()
                listB = self.childs[other_hash_char].collect_all_bucket_keys()
                return listA + listB
            else:
                # Child subtree has enough results, so call method on child
                return self.childs[hash_char].bucket_keys_to_guarantee_result_set_size(bucket_key, N, tree_depth+1)
        else:
            # That subtree is not existing, so just follow the other side
            return self.childs[other_hash_char].bucket_keys_to_guarantee_result_set_size(bucket_key, N, tree_depth+1)


class RandomBinaryProjectionTree(LSHash):
    """
    Projects a vector on n random hyperplane normals and assigns
    a binary value to each projection depending on the sign. This
    divides the data set by each hyperplane and generates a binary
    hash value in string form, which is being used as a bucket key
    for storage.

    Almost same as RandomBinaryProjections. Difference is that
    this hash constructs a binary tree in order to guarantee to
    always be able to retrieve N results. Use minimum_result_size
    to set this N.
    """

    def __init__(self, hash_name, projection_count, minimum_result_size, rand_seed=None):
        """
        Creates projection_count random vectors, that are used for projections
        thus working as normals of random hyperplanes. Each random vector /
        hyperplane will result in one bit of hash.

        So if you for example decide to use projection_count=10, the bucket
        keys will have 10 digits and will look like '1010110011'.
        """
        super(RandomBinaryProjectionTree, self).__init__(hash_name)
        self.projection_count = projection_count
        self.dim = None
        self.normals = None
        self.rand = numpy.random.RandomState(rand_seed)
        self.normals_csr = None
        self.tree_root = None
        self.minimum_result_size = minimum_result_size

    def reset(self, dim):
        """ Resets / Initializes the hash for the specified dimension. """
        if self.dim != dim:
            self.dim = dim
            self.normals = self.rand.randn(self.projection_count, dim)
            self.tree_root = RandomBinaryProjectionTreeNode()

    def hash_vector(self, v, querying=False):
        """
        Hashes the vector and returns the binary bucket key as string.
        """
        if scipy.sparse.issparse(v):
            # If vector is sparse, make sure we have the CSR representation
            # of the projection matrix
            if self.normals_csr == None:
                self.normals_csr = scipy.sparse.csr_matrix(self.normals)
            # Make sure that we are using CSR format for multiplication
            if not scipy.sparse.isspmatrix_csr(v):
                v = scipy.sparse.csr_matrix(v)
            # Project vector onto all hyperplane normals
            projection = self.normals_csr.dot(v)
        else:
            # Project vector onto all hyperplane normals
            projection = numpy.dot(self.normals, v)

        # Build binary key
        binary_key = ''.join(['1' if x > 0.0 else '0' for x in projection])

        if querying:
            #print 'Querying...'
            # Make sure returned buckets keys contain at least N results
            return self.tree_root.bucket_keys_to_guarantee_result_set_size(binary_key, self.minimum_result_size, 0)
        else:
            # We are indexing, so adapt bucket key counter in binary tree
            self.tree_root.insert_entry_for_bucket(binary_key, 0)

            # Return binary key
            return [binary_key]

    def get_config(self):
        """
        Returns pickle-serializable configuration struct for storage.
        """
        # Fill this dict with config data
        return {
            'hash_name': self.hash_name,
            'dim': self.dim,
            'projection_count': self.projection_count,
            'normals': self.normals,
            'tree_root': self.tree_root,
            'minimum_result_size': self.minimum_result_size
        }

    def apply_config(self, config):
        """
        Applies config
        """
        self.hash_name = config['hash_name']
        self.dim = config['dim']
        self.projection_count = config['projection_count']
        self.normals = config['normals']
        self.tree_root = config['tree_root']
        self.minimum_result_size = config['minimum_result_size']




