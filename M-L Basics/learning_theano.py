import theano.tensor as T
from theano import function
import numpy as np

a = T.dscalar('a')
b = T.dscalar('b')
c = T.dscalar('c')
d = a+b-c
f = function([a,b,c],d)
print(f(20,2,3))

a = T.dmatrix('a')
b = T.dmatrix('b')
c = a*b
f = function([a,b],c)
x = np.array([[1,1],[2,2]])
y = np.array([[3,3],[4,4]])
print(f(x,y))

a = T.dmatrix('a')
activator = T.nnet.sigmoid(a)
f = function([a],activator)
print(f[1,-1,0])

import open