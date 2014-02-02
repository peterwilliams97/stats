"""
http://stackoverflow.com/questions/2682144/matplotlib-analog-of-rs-pairs
"""
from __future__ import division
import numpy as np
import pylab as pl
import pandas as pd
import rpy2

pd.set_option('display.width', 180)

pd.read_csv('Credit.csv')
credit = pd.read_csv('Credit.csv')
axes = pd.tools.plotting.scatter_matrix(credit)

cr2 = credit.set_index('Unnamed: 0')
axes = pd.tools.plotting.scatter_matrix(cr2)

pl.show()