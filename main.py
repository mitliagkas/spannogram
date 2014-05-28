from numpy import random
import numpy
from scipy import linalg
from spannogram import *

__author__ = 'migish'

print "Hello world"

p = 10
n = 10000
d = 3
sigma = 0.01

v = random.randn(p, 1)
v /= numpy.linalg.norm(v)

X = v.dot(random.randn(1, n)) + sigma*random.randn(p,n)

#[w, V] = linalg.eigh((1.0/n)*X.dot(X.T))
[w, V] = linalg.eigh(X.dot(X.T))
idx = w.argsort()
w = w[idx]
V = V[:,idx]

print w
print w[-d:]

[xprime, value] = spannogram(V[:, -d:], w[-d:], eps=0.3)

print v.T.dot(V[:, -d:])
print v.T.dot(xprime)
print
print value



