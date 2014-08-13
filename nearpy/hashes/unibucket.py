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

from nearpy.hashes.lshash import LSHash


class UniBucket(LSHash):
    """
    Puts alls vectors in one bucket. This is used for testing
    the engines and experiments.
    """

    def __init__(self, hash_name):
        """ Just keeps the name. """
        super(UniBucket, self).__init__(hash_name)
        self.dim = None

    def reset(self, dim):
        """ Resets / Initializes the hash for the specified dimension. """
        self.dim = dim

    def hash_vector(self, v, querying=False):
        """
        Hashes the vector and returns the bucket key as string.
        """
        # Return bucket key identical to vector string representation
        return [self.hash_name+'']

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
