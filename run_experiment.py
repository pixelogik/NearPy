import time
import numpy

from nearpy import Engine
from nearpy.hashes import RandomDiscretizedProjections, UniBucket
from nearpy.filters import NearestFilter, UniqueFilter
from nearpy.distances import EuclideanDistance
from nearpy.experiments import DistanceRatioExperiment, RecallPrecisionExperiment

# Set dimension and vector count for this experiment
dimension = 100
vector_count = 100000

# Create data set from two clusters
vectors = []

center = numpy.random.randn(dimension)
for index in xrange(vector_count/2):
    vector = center + 0.01 * numpy.random.randn(dimension)
    vectors.append(vector)

center = numpy.random.randn(dimension)
for index in xrange(vector_count/2):
    vector = center + 0.01 * numpy.random.randn(dimension)
    vectors.append(vector)

# We are looking for the N closest neighbours
N = 20
nearest = NearestFilter(N)

# We will fill this array with all the engines we want to test
engines = []

print 'Creating engines...'

# We are going to test these bin widths
bin_widths = [ 0.01 * x for x in range(1,5)]
# Create engines for all configurations
for bin_width in bin_widths:
    # Use four random 1-dim discretized projections
    rdp1 = RandomDiscretizedProjections('rdp1', 4, bin_width)
    rdp2 = RandomDiscretizedProjections('rdp2', 4, bin_width)
    rdp3 = RandomDiscretizedProjections('rdp3', 4, bin_width)
    rdp4 = RandomDiscretizedProjections('rdp4', 4, bin_width)
    #ub1 = UniBucket('uni')

    # Create engine with this configuration
    #engine = Engine(dimension, lshashes=[rdp1, rdp2, rdp3, rdp4],
    #                vector_filters=[unique, nearest])
    engine = Engine(dimension, lshashes=[rdp1, rdp2, rdp3, rdp4],
                    vector_filters=[nearest])

    # Add engine to list of engines to evaluate
    engines.append(engine)

print 'Creating experiment and performing exact search...'

# Create experiment (looking for ten closest neighbours).
# The constructor performs exact search for evaluation.
# So the data set should not be too large for experiments.
exp = DistanceRatioExperiment(N, vectors, coverage_ratio=0.01)

print 'Performing experiment for all engines...'

# Perform experiment for all engines
result = exp.perform_experiment(engines)

