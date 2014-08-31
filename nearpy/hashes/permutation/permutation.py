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

#from bitarray import bitarray

from nearpy.hashes.permutation.permute import Permute
from nearpy.hashes.permutation.permutedIndex import PermutedIndex


class Permutation:

    """
    This class 1) stores all the permutedIndex in a dict self.permutedIndexs ({hash_name,permutedIndex}) 
    and 2) provide a method to get the neighbour bucket keys given hash_name and query bucket key.
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
        """
        Build a permutedIndex and store it into the dict self.permutedIndexs.
        lshash: the binary lshash object (nearpy.hashes.lshash).
        buckets: the buckets object corresponding to lshash. It's a dict object 
                 which can get from nearpy.storage.buckets[lshash.hash_name]
        num_permutation: the number of sorted randomly-permuted bucket key lists (SRPBKL).
        beam_size: beam size, details please refer to __init__() in nearpy.hashes.permutation.PermutedIndex 
        num_neighbour: the number of neighbour bucket keys needed to return in self.get_neighbour_keys().
        """
        # Init a PermutedIndex
        pi = PermutedIndex(
            lshash,
            buckets,
            num_permutation,
            beam_size,
            num_neighbour)
        # get hash_name
        hash_name = lshash.hash_name
        self.permutedIndexs[hash_name] = pi

    def get_neighbour_keys(self, hash_name, bucket_key):
        """
        Return the neighbour buckets given hash_name and query bucket key.
        """
        # get the permutedIndex given hash_name
        permutedIndex = self.permutedIndexs[hash_name]
        # return neighbour bucket keys of query bucket key
        return permutedIndex.get_neighbour_keys(
            bucket_key,
            permutedIndex.num_neighbour)
