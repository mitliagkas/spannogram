import math
import numpy as np
from numpy import linalg


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

    assert w.shape[0] == d

    maximum = float("-inf")

    xprime = None

    for i in range(int(math.ceil(eps ** (-d)))):
        v = np.random.randn(d, 1)

        x = v.T.dot( np.sqrt(np.diag(w)).dot(u.T) )
        x /= np.linalg.norm(x)

        value = x.dot(u).dot(np.diag(w)).dot(u.T).dot(x.T)

        if value > maximum:
            xprime = x
            maximum = value

    return xprime.T, maximum


def spca(a, s, k, d):
    """
    Runs the spannogram-based sparse PCA algorithm.
    Uses zero-forcing 'deflation' for multiple components.

    :param a: The Hermitian matrix to be decomposed.
    :param s: An integer describing the desired sparsity.
    :param k: The number of components to be extracted.
    :param d: The number of components to use for the spectral approximation.

    Current algorithm:
        1. Approximate A using d-top eigen-vectors
        2. Run spannogram
        3. Get direction
        4. Keep k-strongest elements as component
        5. Zero-force corresponding rows/columns of A
        6. Go to 1
    """

    p = a.shape[0]
    X = np.zeros((p, k))

    for l in range(k):
        # 1
        [w, V] = linalg.eigh(a)
        idx = w.argsort()
        w = w[idx]
        V = V[:, idx]

        # 2,3
        xprime, value = spannogram(V[:, -d:], w[-d:])

        # 4
        idx = np.abs(xprime).argsort(axis=0)
        for i in idx[:-s]:
            xprime[i] = 0

        X[:, l] = xprime[:, 0]

        # 5
        for i in idx[-s:]:
            a[i, :] = 0
            a[:, i] = 0

    return X




