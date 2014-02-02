import numpy as np

a = np.random.randn(2, 3)

def my_pinv(a):
    b = np.dot(a.T, a)
    print b
    c = np.linalg.inv(b)
    print c
    d = np.dot(c, a.T)
    return d
    return np.dot(np.linalg.inv(np.dot(a.T, a)), a.T)

def test(a):
    print '-' * 80
    b = np.linalg.pinv(a)
    x = np.dot(b, a)
    #c = my_pinv(a)
    print a
   # print
   # print b
    #print c
    print
    print x
    #print np.dot(c, a)
    print a.shape, np.max(np.abs(x- np.eye(x.shape[0]) ))

test(np.eye(3, dtype=float) * 2)
#test(np.ones((2, 2), dtype=float) * 2)
test(np.ones((2, 3), dtype=float) * 2)
test(np.ones((3, 2), dtype=float) * 2)
a = np.zeros((3, 3), dtype=float)
for i in xrange(a.shape[0]):
    a[i, i] = np.random.randn()
test(a)

a = np.random.randn(3, 3) * 0.5
for i in xrange(a.shape[0]):
    a[i, i] = np.random.randn()
test(a)
    
    
a = np.eye(3, dtype=float) +  np.random.randn(3, 3) * 1e-9
test(a)

print '=' * 80
for _ in xrange(5):
    a = np.eye(3, dtype=float) +  np.random.randn(3, 3) * 1e-9
    a[1, :] = a[0, :] + np.random.randn(3) * 1e-9
    a[2, :] = a[0, :] + np.random.randn(3) * 1e-9
    a[:, 0] = 1
    a[0, 1] = 0
    a[1, 1] = 0
    a[0, 2] = 0
    test(a)        

    
a = np.eye(10, dtype=float) +  np.random.randn(10, 10) * 1e-9
test(a)    