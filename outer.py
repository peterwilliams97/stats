from __future__ import division
import sys
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 


        
def p(X):
    print '-' * 80
    print X
    print X.shape
  

X = np.ones((3, 2)) * 2  
Y = np.ones((4, 2)) * 2 

p(X)
p(Y)

A = np.subtract.outer(X, Y)
p(A)

A = np.subtract.outer(X, Y, axis=0)
p(A)

    
          

   