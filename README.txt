# NearPy

NearPy is a Python framework for fast (approximated) nearest neighbour search in high dimensional vector spaces using different locality-sensitive hashing methods.

It allows to experiment and to evaluate new methods but is also production-ready. It comes with a redis storage adapter.

To install simply do *pip install NearPy*. It will also install the packages scipy, numpy and redis.

Both dense and sparse (scipy.sparse) vectors are supported right now.

Read more here: http://nearpy.io

## Principle

To find approximated nearest neighbours for a query vector, first the vectors to be stored are indexed. For each vector that should be indexed ('stored') a hash is generated,
that is a string value. This hash is used as the bucket key
of the bucket the vector is then stored into. Buckets are in most cases just lists of vectors, it is the terminology used in these applications.

This would not be of any use for finding nearest neighbours to a query vector. So the 'secret' to this mechanism is to use so called locality sensitive hashes, LSHs.
These hashes take spatiality into account so that they tend to generate identical hash values (bucket keys) for close vectors. This makes it then super fast to get close
vectors given a query vector. Because this is a very rough approach it is called approximated nearest neighbour search, because you might not get the real nearest
neighbours. In many applications this is fine because you just wanna get 20 vectors that are 'equal enough'.

## Engine

When using NearPy you will mostly do this by configuring and using an Engine object. Engines are configurable
pipelines for approximated nearest neighbour search (ANNS) using locality sensitive hashes (LSHs).

![alt text](http://nearpy.io/images/Pipeline.png "Pipeline diagram")

Engines are configured using the constructor that accepts the different components along the pipeline:

```python
def __init__(self, dim, lshashes=[RandomBinaryProjections('default', 10)],
             distance=CosineDistance(),
             vector_filters=[NearestFilter(10)],
             storage=MemoryStorage()):
```

The ANNS pipeline is configured for a fixed dimensionality of the feature space, that is set using the dim parameter of the constructor. This must be an positive integer value.

The engine can use multiple LSHs and takes them from the lshashes parameter, that must be an array of
LSHash objects.

Depending on the kind of filters used during querying a distance measure can be specified. This is only
needed if you use filters that need a distance (like NearestFilter or DistanceThresholdFilter).

Filters are used in a last step during querying nearest neighbours. Existing implementations are NearestFilter, DistanceThresholdFilter and UniqueFilter.

The is another parameter to the engine called fetch_vector_filters, which is performed after fetching candidate vectors from the buckets and before distances
or vector filters are used. This is by default one UniqueFilter and it should always be. I keep this still as a parameter if people need to play around with that.

The engine supports different kinds of ways how the indexed vectors (and the buckets) are stored. Current
storage implementations are MemoryStorage and RedisStorage.

There are two main methods of the engine:

```python
store_vector(self, v, data=None)
neighbours(self, v)
```
store_vector() hashes vector v with all configured LSHs and stores it in all matching buckets in the storage.
The optional data argument must be JSON-serializable. It is stored with the vector and will be returned in search results.
It is best practice to use only database ids as attached 'data' and store the actual data in a database.

neighbours() hashes vector v with all configured LSHs, collects all candidate vectors from the matching
buckets in storage, applies the (optional) distance function and finally the (optional) filter function
to construct the returned list of either (vector, data, distance) tuples or (vector, data) tuples.

To remove indexed vectors and their data from the engine these two methods can be used:

```python
clean_all_buckets(self)
clean_buckets(self, hash_name)
```
## Hashes

All LSH implementatiosn in NearPy do subclass nearpy.hashes.LSHash, which has one main method, besides
constructor and reset methods.

```python
    hash_vector(self, v)
```

hash_vector() hashes the specified vector and returns a list of bucket keys with one or more entries.

The LSH RandomBinaryProjections projects the specified vector on n random
normalized vectors in the feature space and returns a string made from zeros and ones. If v lies on
the positive side of the n-th normal vector the n-th character in the string is a '1', if v lies
on the negative side of it, the n-th character in the string is a '0'. This way this LSH projects
each possible vector of the feature space ('input space') into one of many possible buckets.

The LSH RandomDiscretizedProjections is almost identical to RandomBinaryProjections. The only difference is,
that is divides the projection value by a bin width, and using the bin index in each random projection
as part of the bucket key. Given the same count of random projection vectors as RandomBinaryProjections, this
results in more buckets given the same vector set. The density of buckets on the projections can be controlled
by the bin width, which is part of the constructor.

The LSH PCABinaryProjections is trained with a training set of vectors specified with the constructor. It
performs PCA (principal component analysis) to find the directions of highest variance in the training set.
It then uses the first n principal components as projection vectors (or dimensions of the subspace that is
projected into). The idea was that this makes it more safe to get a good distribution of the vectors among
the buckets. I do not have any tests on this and don't know if this makes sense at all.

The LSH PCADiscretizedProjections is the pca version of RandomDiscretizedProjections, not using random vectors
but the first n principal components of the training set, like PCABinaryProjections does it.

## Hash configurations

To save your index in Redis and re-use it after the indexing process is done you should persist
your hash configurations so that afterwards you can re-create the same engine for more indexing
or querying.

This is done like this:

```python
# Create redis storage adapter
redis_object = Redis(host='localhost', port=6379, db=0)
redis_storage = RedisStorage(redis_object)

# Get hash config from redis
config = redis_storage.load_hash_configuration('MyHash')

if config is None:
    # Config is not existing, create hash from scratch, with 10 projections
    lshash = RandomBinaryProjections('MyHash', 10)
else:
    # Config is existing, create hash with None parameters
    lshash = RandomBinaryProjections(None, None)
    # Apply configuration loaded from redis
    lshash.apply_config(config)

# Create engine for feature space of 100 dimensions and use our hash.
# This will set the dimension of the lshash only the first time, not when
# using the configuration loaded from redis. Use redis storage to store
# buckets.
engine = Engine(100, lshashes=[lshash], storage=redis_storage)

# Do some stuff like indexing or querying with the engine...

# Finally store hash configuration in redis for later use
redis_storage.store_hash_configuration(lshash)
```
===========

Example usage:

```python
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
```

===========








