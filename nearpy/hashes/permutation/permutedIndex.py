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

import logging

from bitarray import bitarray

from nearpy.hashes.permutation.permute import Permute


class PermutedIndex:
    """
    The goal of permutedIndex is to help find the neighbours, 
    in term of Hamming distance, of a query key in a set of binary keys.
    
    PermutedIndex is essentially a number of sorted permuted key lists 
    (self.permuted_lists stores all these lists). Each list correspond 
    to a Permute object, which helps permute a binary key.
    
    For example, a set a binary keys: ['00011','00001','00111','01111','10000'], 
    and we want to find the 1-neighbour of '00001'.
    
    If we just sort the original list, i.e. ['00001','00011','00111','01111','10000'], 
    the 1-neighbour of '00001' in the sorted list, '00011', is not the cloest neighbour 
    in term of Hamming distance. 
    
    Here's an approximate solution:
    1) permute every binary key in the list using a map [1,2,3,4,5]=>[2,3,4,5,1], 
       then the permuted list is ['00110','00010','01110','11110','00001']. 
    2) Sorted the permuted list, => ['00001','00010','00110','01110','11110'].
    3) Given the query key '00001', permute it, => '00010'.
    4) Using the permuted query key to do a binary search in the sorted permuted list, 
       get the neighbours, '00001' and '00110'.
    5) Doing a reversed permutation on the neighbours, => '10000' and '00011'.
    5) The real neighbour in term of Hamming distance is found: '10000'.
    
    Often, only one permuted list is not enough to find all the neighbours. 
    The more permuted lists, the more neighbours we can find. The parameter num_permutation
    specifies how many permuted lists will be created.
    
    In the step 4, after we find the position of permuted query key in the list, 
    it's better to return more neighbours around that place as candidates. 
    The parameter beam_size specifies how many neighbours in the sorted list will be returned. 
    """

    def __init__(
            self,
            lshash,
            buckets,
            num_permutation,
            beam_size,
            num_neighbour):

        self.num_permutation = num_permutation
        self.beam_size = beam_size
        self.lshash = lshash
        self.projection_count = self.lshash.projection_count
        self.num_neighbour = num_neighbour

        # add permutations
        self.permutes = []
        for i in xrange(self.num_permutation):
            p = Permute(self.projection_count)
            self.permutes.append(p)

        # convert current buckets to an array of bitarray
        original_keys = []
        for key in buckets:
            ba = bitarray(key)
            original_keys.append(ba)

        # build permutation lists
        self.permuted_lists = []
        i = 0
        for p in self.permutes:
            logging.info(
                'Creating Permutated Index for {}: #{}/{}'.format(lshash.hash_name, i, len(self.permuted_lists)))
            i += 1
            permuted_list = []
            for ba in original_keys:
                c = ba.copy()
                p.permute(c)
                permuted_list.append((c, ba))
            # sort the list
            permuted_list = sorted(permuted_list)
            self.permuted_lists.append(permuted_list)

    def hamming_distance(self, a, b):
        return int((a ^ b).count())

    def get_neighbour_keys(self, bucket_key, k):
        """
        The computing complexity is O( np*beam*log(np*beam) )
        where,
        np = number of permutations
        beam = self.beam_size
        
        Make sure np*beam is much less than the number of bucket keys, 
        otherwise we could use brute-force to get the neighbours
        """
        # convert query_key into bitarray
        query_key = bitarray(bucket_key)

        topk = set()
        for i in xrange(len(self.permutes)):
            p = self.permutes[i]
            plist = self.permuted_lists[i]
            candidates = p.search_revert(plist, query_key, self.beam_size)
            topk = topk.union(set(candidates))
        topk = list(topk)
        
        # sort the topk neighbour keys according to the Hamming distance to qurey key
        topk = sorted(topk, key=lambda x: self.hamming_distance(x, query_key))
        # return the top k items
        topk_bin = [x.to01() for x in topk[:k]]
        return topk_bin
