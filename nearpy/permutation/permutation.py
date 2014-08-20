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

from bitarray import bitarray

from nearpy.permutation.permute import Permute
from nearpy.permutation.permutedIndex import PermutedIndex


class Permutation:

    """
    The pumutation class
    """

    def __init__(self):
        # self.permutedIndexs' key is the corresponding lshash's hash_name
        self.permutedIndexs = {}

    def build_permuted_index(
            self,
            lshash,
            buckets,
            num_permutation,
            beam_size,
            num_neighbour):
        pi = PermutedIndex(
            lshash,
            buckets,
            num_permutation,
            beam_size,
            num_neighbour)
        hash_name = lshash.hash_name
        self.permutedIndexs[hash_name] = pi

    def get_neighbour_keys(self, hash_name, bucket_key):
        permutedIndex = self.permutedIndexs[hash_name]
        return permutedIndex.get_neighbour_keys(
            bucket_key,
            permutedIndex.num_neighbour)
