from __future__ import division
from time import time
from tempfile import mkstemp
import multiprocessing as mp

from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
import numpy as np

# This workaround (read: hack) makes pytest and memory_profiler play nice:
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pairwise_distance import mean_pairwise_distance


def get_data(N, seed=10, save_data=True):
    """ This generates some test data that we can use to test our pairwise-
    distance functions, if the specified data doesn't already exist.

    Required arguments:
    N           -- The number of datapoints in the test data.

    Optional arguments:
    seed        -- The seed for NumPy's random module. (default: 10)
    save_data   -- Use temp file if False, .csv if True. (default: False)
    """

    # Simulate some blobby latitude and longitude data.
    # Feature 1: latitude
    # Feature 2: longitude
    # Feature 3: weights (aka counts)
    np.random.seed(seed)
    centers = [[0., 0., 0.],
               [1., 0., 0.],
               [0.5, np.sqrt(0.75), 0.],
               [0.5, np.sqrt(0.75), 0.]]
    cluster_std = [0.3] * len(centers)
    blobs, _ = make_blobs(n_samples=N,
                          n_features=3,
                          centers=centers,
                          cluster_std=cluster_std)

    # The weights should be randomly distributed in [0, 10), so let's fix that.
    blobs[:, 2] = np.random.random_sample((N,)) * 10.

    if save_data:
        f = './test_array.csv'
    else:
        _, f = mkstemp(suffix='.csv')

    # 1 meter corresponds to roughly 1e-5 degrees (or ~1e-7 radians) in
    # latitutde-longitude coordinates, and are always between +/- 180, so we
    # don't need to save more precision than eight significant figures.
    # Be careful! This WILL overwrite an existing file if called.
    np.savetxt(f, blobs, delimiter=',', fmt='%09.5f')

    return f


def test_mean_pairwise_distance(N=1000, data_file=None):
    """
    This function computes the pairwise distances on a (small) simulated
    dataset to make sure the distributed function returns the same sum and
    mean for pairwise distances as SciPy's pdist function.

    Optional arguments:
    N           -- The number of points to generate in the simulated dataset.
                   (default: 1000)
    data_file   -- The name of the file containing sample data. (default: None)
    """

    # Get the sample data. If it doesn't exist yet, create some in a temp file.
    if data_file is None:
        data_file = get_data(N, save_data=False)

    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',')
    X, weights = np.array_split(data, indices_or_sections=[2], axis=1)
    weights = np.squeeze(weights)  # Remove singleton dimension

    ##########################################################################
    # Parallel:
    # Parallelised code partially based on:
    # https://gist.github.com/baojie/6047780
    t_start_parallel = time()
    parallel_sum, parallel_mean = mean_pairwise_distance(X, weights=weights)
    t_end_parallel = time()
    ##########################################################################

    ##########################################################################
    # Serial:
    # Comment this out if you use a high N as it will eat RAM!
    t_start_serial = time()
    new_weights = np.array([weights[i] * weights[j]
                            for i in xrange(N - 1) for j in xrange(i + 1, N)])
    serial_sum = np.dot(new_weights, pdist(X, 'euclidean'))
    N = weights.sum()
    serial_mean = serial_sum / (((N - 1)**2 + (N + 1)) / 2 + N)
    t_end_serial = time()
    ##########################################################################

    # There is minor rounding error, but check for equality:
    assert np.isclose(serial_sum, parallel_sum)
    assert np.isclose(serial_mean, parallel_mean)

    # Print out a nice summary of the performance and data measures:
    def print_time(s, t):
        return "%10s: %10.6f s" % (s, t)
    print  # Print a newline to make things nice, due to pytest
    print print_time('parallel', t_end_parallel - t_start_parallel)
    print print_time('serial', t_end_serial - t_start_serial)
    print 'sum = {}'.format(parallel_sum)
    print 'mean = {}'.format(parallel_mean)


# This file shouldn't be executed or imported on its own. This __main__ block
# is so that external tools (e.g. memory_profiler) work correctly:
if __name__ == "__main__":
    test_mean_pairwise_distance()
