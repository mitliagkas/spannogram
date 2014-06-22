#!/usr/bin/env python
"""Contains the implementation of truncated power method"""
import math

import numpy as np
from numpy import random


__author__ = 'Ioannis Mitliagkas'
__copyright__ = "Copyright 2014, Ioannis Mitliagkas"
__license__ = "MIT"
__version__ = "0.0.1"
__email__ = "ioannis@utexas.edu"
__status__ = "alpha"


def tpower(a, s, k):
    """
    Implements the truncated power method for sparse PCA.
    For the extraction of many components deflation is performed
    via zero-forcing the row and columns of previously extracted elements.

    :param a: Input matrix a
    :param s: The desired level of sparsity
    :param k: The desired number of components to extract

    The algorithm works as follows:
    1. Start from a random dense initial direction.
        2. Perform one iteration of the power method
        3. Truncate the result, keeping the s top elements in magnitude
        4. Until you reach a number of steps, go to 2
        5. Get truncated result, x
    5. Deflate matrix a using x
    6. Until you read the desired number of components, k, go to 1.
    """

    assert isinstance(a, np.ndarray)

    p = a.shape[0]
    X = np.zeros((p, k))

    for ik in xrange(k):
        x = random.randn(p, 1)
        iter=0
        while True:
            # Power step
            x = a.dot(x)

            # Truncate result
            idx = np.abs(x).argsort(axis=0)
            for l in idx[:-s]:
                x[l] = 0

            # Normalize
            x /= np.linalg.norm(x)

            # Check for termination
            # TODO: Add convergence check
            if iter>math.ceil(math.log(p)):
                break
            iter+=1

        # Store component x
        X[:, ik] = x[:, 0]

        # Deflate a
        idx = np.abs(x).argsort(axis=0)
        for i in idx[-s:]:
            a[i, :] = 0
            a[:, i] = 0

    return X




