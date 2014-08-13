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


class LSHash(object):
    """ Interface for locality-sensitive hashes. """

    def __init__(self, hash_name):
        """
        The hash name is used in storage to store buckets of
        different hashes without collision.
        """
        self.hash_name = hash_name

    def reset(self, dim):
        """ Resets / Initializes the hash for the specified dimension. """
        raise NotImplementedError

    def hash_vector(self, v, querying=False):
        """
        Hashes the vector and returns a list of bucket keys, that match the
        vector. Depending on the hash implementation this list can contain
        one or many bucket keys. Querying is True if this is used for
        retrieval and not indexing.
        """
        raise NotImplementedError

    def get_config(self):
        """
        Returns pickle-serializable configuration struct for storage.
        """
        raise NotImplementedError

    def apply_config(self, config):
        """
        Applies config
        """
        raise NotImplementedError
