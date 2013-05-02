===========
NearPy
===========

NearPy is a Python framework for fast (approximated) nearest neighbour search in high dimensional vector spaces using different locality-sensitive hashing methods.

It allows to experiment and to evaluate new methods but is also production-ready. It comes with a redis storage adapter.

Example usage:

    from nearpy import Engine
    from nearpy.hashes import RandomBinaryProjections

    # Dimension of our vector space
    dimension = 500

    # Create a random binary hash with 10 bits
    rbp = RandomBinaryProjections('rbp', 10)

    # Create engine with pipeline configuration
    engine = Engine(dimension, lshashes=[rbp])

    # Index 1000000 random vectors (set their data to a unique string)
    for index in range(100000):
        v = numpy.random.randn(dimension)
        engine.store_vector(v, 'data_%d' % index)

    # Create random query vector
    query = numpy.random.randn(dimension)

    # Get nearest neighbours
    N = engine.neighbours(query)

Read more here: http://nearpy.io
