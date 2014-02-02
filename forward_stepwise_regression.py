"""
http://statsmodels.sourceforge.net/devel/examples/generated/example_ols.html
"""
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def test_linear():
    np.random.seed(111)

    n_sample = 100
    max_val = 10

    x = np.linspace(0, max_val, n_sample)
    #X = np.column_stack((x, x**2))
    X = np.c_[x, x**2]
    beta = np.array([1, 0.1, 10])
    e = np.random.normal(size=n_sample) 

    X = sm.add_constant(X, prepend=False)
    y = np.dot(X, beta) + e

    for i in xrange(5):
        print '%3d: %s %s' % (i, X[i, :], y[i])

    print
    print
    model = sm.OLS(y, X)
    results = model.fit()
    print results.summary()
    print
    print
    print results.params
    print results.rsquared
 

def test_nonlinear():
    np.random.seed(111)
    
    n_sample = 50
    max_val = 30
    sig = 0.5

    x = np.linspace(0, max_val, n_sample)
    X = np.c_[x, np.sin(x), (x - 5)**2, np.ones(n_sample)]
    beta = np.array([0.5, 0.5, -0.02, 5.0])
    e = np.random.normal(size=n_sample) 

    #X = sm.add_constant(X, prepend=False)
    y_true = np.dot(X, beta)
    y = y_true + sig * e

    for i in xrange(5):
        print '%3d: %s %s' % (i, X[i, :], y[i])

    print
    print
    model = sm.OLS(y, X)
    results = model.fit()
    print results.summary()
    print
    print
    print results.params
    print results.rsquared 
    print results.bse
    print results.predict()

    
    plt.figure()
    plt.plot(x, y, 'o', x, y_true, 'b-')
    prstd, iv_l, iv_u = wls_prediction_std(results)
    plt.plot(x, results.fittedvalues, 'r--.')
    plt.plot(x, iv_u, 'r--')
    plt.plot(x, iv_l, 'r--')
    plt.title('blue: true,   red: OLS')
    plt.show()

    
def test_dummy():
    nsample = 50
    groups = np.zeros(nsample, int)
    groups[20:40] = 1
    groups[40:] = 2
    dummy = (groups[:, None] == np.unique(groups)).astype(float)
    x = np.linspace(0, 20, nsample)
    X = np.c_[x, dummy[:, 1:], np.ones(nsample)]
    beta = [1., 3, -3, 10]
    y_true = np.dot(X, beta)
    e = np.random.normal(size=nsample)
    y = y_true + e   
 
  
#test_linear()    
test_nonlinear()
