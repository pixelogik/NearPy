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


class RandomBinaryProjections(LSHash):
    """
    Projects a vector on n random hyperplane normals and assigns
    a binary value to each projection depending on the sign. This
    divides the data set by each hyperplane and generates a binary
    hash value in string form, which is being used as a bucket key
    for storage.
    """

    def __init__(self, hash_name, projection_count, rand_seed=None):
        """
        Creates projection_count random vectors, that are used for projections
        thus working as normals of random hyperplanes. Each random vector /
        hyperplane will result in one bit of hash.

        So if you for example decide to use projection_count=10, the bucket
        keys will have 10 digits and will look like '1010110011'.
        """
        super(RandomBinaryProjections, self).__init__(hash_name)
        self.projection_count = projection_count
        self.dim = None
        self.normals = None
        self.rand = numpy.random.RandomState(rand_seed)
        self.normals_csr = None

    def reset(self, dim):
        """ Resets / Initializes the hash for the specified dimension. """
        if self.dim != dim:
            self.dim = dim
            self.normals = self.rand.randn(self.projection_count, dim)

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
        # Return binary key
        return [''.join(['1' if x > 0.0 else '0' for x in projection])]

    def get_config(self):
        """
        Returns pickle-serializable configuration struct for storage.
        """
        # Fill this dict with config data
        return {
            'hash_name': self.hash_name,
            'dim': self.dim,
            'projection_count': self.projection_count,
            'normals': self.normals
        }

    def apply_config(self, config):
        """
        Applies config
        """
        self.hash_name = config['hash_name']
        self.dim = config['dim']
        self.projection_count = config['projection_count']
        self.normals = config['normals']




