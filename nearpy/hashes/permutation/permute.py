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

import random

#from bitarray import bitarray
from bisect import bisect_left


class Permute:

    """
    Permute provide a random [n] -> [n] permute operation.
    e.g, a [3] -> [3] permute operation could be:
    [0,1,2] -> [1,0,2], "010" => "100"
    """

    def __init__(self, n):
        """
        Init a Permute object. Randomly generate a mapping, e.g. [0,1,2] -> [1,0,2]
        """
        m = range(n)
        for end in xrange(n - 1, 0, -1):
            r = random.randint(0, end)
            tmp = m[end]
            m[end] = m[r]
            m[r] = tmp
        self.mapping = m

    def permute(self, ba):
        """
        Permute the bitarray ba inplace.
        """
        c = ba.copy()
        for i in xrange(len(self.mapping)):
            ba[i] = c[self.mapping[i]]
        return ba

    def revert(self, ba):
        """
        Reversely permute the bitarray ba inplace.
        """
        c = ba.copy()
        for i in xrange(len(self.mapping)):
            ba[self.mapping[i]] = c[i]
        return ba

    def search_revert(self, bas, ba, beam_size):
        """
        ba: query bitarray
        bas: a sorted list of tuples of (permuted bitarray, original bitarray)
        return : query bitarray's beam-size neighbours (unpermuted bitarray)
        """
        pba = ba.copy()
        self.permute(pba)
        assert(beam_size % 2 == 0)
        half_beam = beam_size / 2

        # binary search (pba,ba) in bas
        idx = bisect_left(bas, (pba, ba))

        start = max(0, idx - half_beam)
        end = min(len(bas), idx + half_beam)
        res = bas[start:end]

        # return the original(unpermuted) keys
        res = [x[1] for x in res]
        return res
