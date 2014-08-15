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

from nearpy.utils import numpy_array_from_list_or_numpy_array, perform_pca


class PCABinaryProjections(LSHash):
    """
    Projects a vector on n first principal components and assigns
    a binary value to each projection depending on the sign. This
    divides the data set by each principal component hyperplane and
    generates a binary hash value in string form, which is being
    used as a bucket key for storage.
    """

    def __init__(self, hash_name, projection_count, training_set):
        """
        Computes principal components for training vector set. Uses
        first projection_count principal components for projections.

        Training set must be either a numpy matrix or a list of
        numpy vectors.
        """
        super(PCABinaryProjections, self).__init__(hash_name)
        self.projection_count = projection_count

        # Only do training if training set was specified
        if not training_set is None:
            # Get numpy array representation of input
            training_set = numpy_array_from_list_or_numpy_array(training_set)

            # Get subspace size from training matrix
            self.dim = training_set.shape[0]

            # Get transposed training set matrix for PCA
            training_set_t = numpy.transpose(training_set)

            # Compute principal components
            (eigenvalues, eigenvectors) = perform_pca(training_set_t)

            # Get largest N eigenvalue/eigenvector indices
            largest_eigenvalue_indices = numpy.flipud(
                scipy.argsort(eigenvalues))[:projection_count]

            # Create matrix for first N principal components
            self.components = numpy.zeros((self.dim,
                                           len(largest_eigenvalue_indices)))

            # Put first N principal components into matrix
            for index in range(len(largest_eigenvalue_indices)):
                self.components[:, index] = \
                    eigenvectors[:, largest_eigenvalue_indices[index]]

            # We need the component vectors to be in the rows
            self.components = numpy.transpose(self.components)
        else:
            self.dim = None
            self.components = None

        # This is only used in case we need to process sparse vectors
        self.components_csr = None

    def reset(self, dim):
        """ Resets / Initializes the hash for the specified dimension. """
        if self.dim != dim:
            raise Exception('PCA hash is trained for specific dimension!')

    def hash_vector(self, v, querying=False):
        """
        Hashes the vector and returns the binary bucket key as string.
        """
        if scipy.sparse.issparse(v):
            # If vector is sparse, make sure we have the CSR representation
            # of the projection matrix
            if self.components_csr == None:
                self.components_csr = scipy.sparse.csr_matrix(self.components)
            # Make sure that we are using CSR format for multiplication
            if not scipy.sparse.isspmatrix_csr(v):
                v = scipy.sparse.csr_matrix(v)
            # Project vector onto all hyperplane normals
            projection = self.components_csr.dot(v).toarray()
        else:
            # Project vector onto all hyperplane normals
            projection = numpy.dot(self.components, v)
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
            'components': self.components
        }

    def apply_config(self, config):
        """
        Applies config
        """
        self.hash_name = config['hash_name']
        self.dim = config['dim']
        self.projection_count = config['projection_count']
        self.components = config['components']


