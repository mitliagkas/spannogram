import math
import numpy as np
import scipy as sp

# Pseudo code
# [U, D] = eigs(A,d)
#
# for i = 1:O(eps^-d)
#     v = randn(d,1)
#     x_i = colinear unit norm vector sto n-dimensional vector v'*sqrt(D)*U
#     metric_i = xi'*U*D*U'*xi
# end
#
# rank_d_eps_optimal = max mextric vector
#
# gia deflation se kathe vima kanw zero forcing pou einai the cheapest choice
# diladi otan vreis to kalytero x_i apo ta parapanw, pairneis ton arxiko sou pinaka, tou skotwneis ta cols/rows indexed by x_i, kaneis svd ston neo "truncated" kai repeat the above steps


def spannogram(u, w, eps=0.1):
    """
    Runs the spannogram algorithm on a rank-d matrix.
    Uses the \epsilon-net argument

    :param u: A p x d array-like structure containing d orthonormal columns.
    :param w: A d x 1 array-like structure containing d eigenvalues.
    :param eps: Desired accuracy. Defaults to 0.1
    :rtype : object a p x 1 array-like structure, containing the optimal vector.
    """

    assert isinstance(u, np.ndarray)
    d = u.shape[1]

    assert w.shape()[0] == d

    for i in math.ceil(eps**(-d)):
        pass



