#!/usr/bin/env python
"""The main execution file"""

import numpy

from spannogram import *
from tpower import *


__author__ = 'Ioannis Mitliagkas'
__copyright__ = "Copyright 2014, Ioannis Mitliagkas"
__license__ = "MIT"
__version__ = "0.0.1"
__email__ = "ioannis@utexas.edu"
__status__ = "alpha"


p = 6
n = 10000
d = 2
sigma = 0.01

v = random.randn(p, 1)
v /= numpy.linalg.norm(v)

X = v.dot(random.randn(1, n)) + sigma * random.randn(p, n)

if False:
    [w, V] = linalg.eigh(X.dot(X.T))
    idx = w.argsort()
    w = w[idx]
    V = V[:, idx]

    [xprime, value] = spannogram(V[:, -d:], w[-d:], eps=0.3)

    print v.T.dot(V[:, -d:])
    print v.T.dot(xprime)
    print
    print value
else:
    k=2
    s=2
    Vo = spcaold(X.dot(X.T), s, k, 2)
    print Vo.T
    print
    V = spca(X.dot(X.T), s, k, 2)
    print V.T
    print
    Vt = tpower(X.dot(X.T), s, k)
    print Vt.T
    print



