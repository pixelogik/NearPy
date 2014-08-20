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

from nearpy.permutation.permute import Permute

class PermutedIndex:

    def __init__(self,lshash,buckets,num_permutation,beam_size,num_neighbour):

        # lshash, buckets are coresponding
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
            logging.info('Creating Permutated Index for {}: #{}/{}'.format(lshash.hash_name,i,len(self.permuted_lists)))
            i+=1
            permuted_list = []
            for ba in original_keys:
                c = ba.copy()
                p.permute(c)
                permuted_list.append((c,ba))
            # sort the list
            permuted_list = sorted(permuted_list)
            self.permuted_lists.append(permuted_list)

       
    def hamming_distance(self,a,b):
        return int((a ^ b).count())


    def get_neighbour_keys(self,bucket_key,k):
        # O( np*beam*log(np*beam) )
        # np = number of permutations
        # beam = self.beam_size
        # np * beam == 200 * 100 Still really fast
        
        query_key = bitarray(bucket_key)
        topk = set()
        for i in xrange(len(self.permutes)):
            p = self.permutes[i]
            plist = self.permuted_lists[i]
            candidates = p.search_revert(plist,query_key,self.beam_size)
            topk = topk.union(set(candidates))
        topk = list(topk)
        topk = sorted(topk, key = lambda x : self.hamming_distance(x,query_key))
        topk_bin = [x.to01() for x in topk[:k]]
        return topk_bin


