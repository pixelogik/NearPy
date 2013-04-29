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


class VectorFilter(object):
    """
    Interface for vector-list filters. They get either (vector, data, distance)
    tupes or (vector, data) tuples and return subsets of them.

    Some filters work on lists of (vector, data, distance) tuples, others work
    on lists of (vector, data) tuples and others work on both types.
    Depending on the configuration of the engine, you have to select the right
    filter chain.

    Filter are chained in the engine, if you specify more than one. This way
    you can combine their functionalities.

    The default filtes in the engine (see engine.py) are a UniqueFilter
    followed by a NearestFilter(10). The UniqueFilter makes sure, that the
    candidate list contains each vector only once and the NearestFilter(10)
    returns the 10 closest candidates (using the distance).

    Which kind you need is very simple to determine: If you use a Distance
    implementation, you have to use filters that take
    (vector, data, distance) tuples. If you however decide to not use Distance
    (Engine with distance=None), you have to use a vector filters that
    process lists of (vector, data) tuples.

    However all filters can handle both input types. They will just return the
    input list if their filter mechanism does not apply on the input type.
    """

    def filter_vectors(self, input_list):
        """
        Returns subset of specified input list.
        """
        raise NotImplementedError
