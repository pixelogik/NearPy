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
import unittest

from nearpy import Engine


class TestEngine(unittest.TestCase):

    def setUp(self):
        self.engine = Engine(1000)

    def test_retrieval(self):
        for k in range(100):
            self.engine.clean_all_buckets()
            x = numpy.random.randn(1000)
            x_data = 'data'
            self.engine.store_vector(x, x_data)
            n = self.engine.neighbours(x)
            y = n[0][0]
            y_data = n[0][1]
            y_distance = n[0][2]
            self.assertTrue((y == x).all())
            self.assertEqual(y_data, x_data)
            self.assertEqual(y_distance, 0.0)


if __name__ == '__main__':
    unittest.main()
