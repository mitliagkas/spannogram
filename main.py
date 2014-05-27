from numpy import random
import numpy
from scipy import linalg

__author__ = 'migish'

print "Hello world"

p = 10
n = 10000
sigma = 1.0

v = random.randn(p, 1)
v /= numpy.linalg.norm(v)

X = v.dot(random.randn(1, n)) + sigma*random.randn(p,n)

[w, V] = linalg.eigh(X.dot(X.T))

print V[:, -1]

print v.T

print

print v.T.dot(V[:,-1])

