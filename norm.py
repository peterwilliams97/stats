from __future__ import division
import sys
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 

def norm(X, axis=-1):
   return np.apply_along_axis(np.linalg.norm, axis, X)
        
def p(X):
    print '-' * 80
    print X
    print X.shape
        
X = np.ones((4,3,2))
p(X)

X[0, 0, 0] = 1
p(X)

Y = [norm(X, i) for i in range(3)]
for i in range(3):
    p(Y[i])
    
p(norm(X))    


   