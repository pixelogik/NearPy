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

from nearpy.distances.distance import Distance


class CosineDistance(Distance):
    """  Uses 1-cos(angle(x,y)) as distance measure. """

    def distance(self, x, y):
        """
        Computes distance measure between vectors x and y. Returns float.
        """
        if scipy.sparse.issparse(x):
            x = x.toarray().ravel()
            y = y.toarray().ravel()
        return 1.0 - numpy.dot(x, y) / (numpy.linalg.norm(x) *
                                        numpy.linalg.norm(y))


    def distance_matrix(self,a,b):
        """
        Computes distance measure between matrix x and matrix y. Return Matrix.
        """
        # a,b should be matrix
        # each row is a vector in a, b
        dt = numpy.dot(a,b.T)
        norm_a = numpy.sqrt(numpy.sum(a * a, axis = 1))
        norm_a = norm_a.reshape((len(norm_a),1))
        norm_b = numpy.sqrt(numpy.sum(b * b, axis = 1))
        norm_b = norm_b.reshape((len(norm_b),1))
        cos_matrix = dt / ( numpy.dot( norm_a , norm_b.T))
        return 1.0-cos_matrix
