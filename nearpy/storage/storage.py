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


class Storage(object):
    """ Interface for storage adapters. """

    def store_vector(bucket_key, v, data):
        """
        Stores vector and JSON-serializable data in bucket with specified key.
        """
        raise NotImplementedError

    def get_bucket(self, bucket_key):
        """
        Returns bucket content as list of tuples (vector, data).
        """
        raise NotImplementedError

    def clean_buckets(self):
        """
        Removes all buckets and their content.
        """
        raise NotImplementedError


#überlegen was im case von trigram space z.b. im storage gespeichert werden muss.

#pro wort: list e von coordinates (int)
#pro



#DISATNZFUNKTION IST KONFIGURIERBAR!!!!

#IM FALLE VON BINÄREN SPARSE MATRIZEN !!!!(trigram) -> aufsteigend, und etc....



#verschiedene arten von vector spaces:

#für alle fällem al durchdenken, wie das speicher und das persitenzmodell aussehen würden

#-> binary sparse (binäre, sparse vektoren, nur 1 und 0)
#-> sparse (nicht-binär, aber sparse)
#-> dense (nicht-binär und dicht)


