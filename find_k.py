"""
    http://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
    http://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
"""
from __future__ import division
import sys, random
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

print np.version.version 


def find_centers(X, k):
    estimator = KMeans(init='k-means++', n_clusters=k, n_init=10)
    estimator.fit(X)
    mu = estimator.cluster_centers_
    labels = estimator.labels_
    #print 'mu:', type(mu), len(mu), mu.shape, mu
    #print 'labels:', type(labels), len(labels), labels.shape, labels
    return mu, labels

    
def Wk(X, mu, labels):
    """Intra-cluster distances"""
    k = len(mu)
    clusters = [X[np.where(labels == i)] for i in range(k)]
    n = [x.shape[0] for x in clusters]
    return sum(np.linalg.norm(clusters[i] - mu[i])**2/(2*n[i]) for i in range(k))


def bounding_box(X):
    xmin, xmax = min(X, key=lambda a:a[0])[0], max(X, key=lambda a:a[0])[0]
    ymin, ymax = min(X, key=lambda a:a[1])[1], max(X, key=lambda a:a[1])[1]
    return (xmin, xmax), (ymin, ymax)


B = 10    
n_K_0 = 1
n_K_1 = 10

def gap_statistic(X):
    """Return
        ks: k values
        Wks: log(intra-cluster distance) for each k
        Wkbs: average referemce log(intra-cluster distance) for each k
        sk: Normalized std dev log(intra-cluster distance) for each k
    """
    (xmin, xmax), (ymin, ymax) = bounding_box(X)
    #print X.shape
    #print (xmin, xmax), (ymin, ymax) 
    # Dispersion for real distribution
    ks = range(n_K_0, n_K_1)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, labels = find_centers(X, k)
        #print indk, k
        #print mu.shape, mu
        #print labels.shape, labels
        Wks[indk] = np.log(Wk(X, mu, labels))

        # Create B reference datasets
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = np.vstack([np.random.uniform(xmin, xmax, X.shape[0]),
                            np.random.uniform(ymin, ymax, X.shape[0])]).T
            mu, labels = find_centers(Xb, k)
            BWkbs[i] = np.log(Wk(Xb, mu, labels))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    #assert False
    return ks, Wks, Wkbs, sk    


#mport random
 
def init_board(N):
    return np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])


def board_separation(x, args): 
    k = args
    x = x.reshape((k, 2))
    k = x.shape[0]
    dx = np.empty((k, k, 2))
    #print dx.shape
    for i in 0, 1:
        tmp = np.subtract.outer(x[:,i], x[:,i]) 
        #print tmp.shape
        dx[:,:,i] = np.subtract.outer(x[:,i], x[:,i]) 
     
       
    diffs = np.apply_along_axis(np.linalg.norm, 2, dx)
    
    for i in range(k):
        diffs[i, i] = 100.0  
    
    if False:
        print '^' * 80
        #print dx, dx.shape
        print x.reshape((k,2))
        print diffs, diffs.shape
        print -diffs.min()
    
    return -diffs.min()
    
    
def calc_centroids(k, r):
    """Return best spaced centroids in square of radius 1 around origin
    """
    from scipy import optimize   
    x0 = np.zeros((k, 2))
    
    if k == 1:
        return x0
    
    #x0[0, :] = -1.0, 1.0
    #x0[1, :] = 1.0, -1.0
    #x0[2:, 0] = np.linspace(-1.0, 1.0, k - 2) 
    #x0[2:, 1] = np.linspace(-1.0, 1.0, k - 2)   
    #print k
    #print x0 
    
    M = 1000
    x1 = np.zeros((k-1, 2))
    test_points = np.random.uniform(-1, 1, size=(10000, 2))
    
     
    
    for m in range(10):
        #print '=' * 80
        #print m
        changed = False
        #print x0, x0.shape
        for i in range(k):
            #print i,
            #print x0, x0.shape
            #print x0[:i, :], x0[:i, :].shape
            #print x0[i+1:, :], x0[i+1:, :].shape
            x1 = np.vstack((x0[:i, :], x0[i+1:, :]))
            #print x0, x0.shape
            #print x1, x1.shape    
            current_min = np.apply_along_axis(np.linalg.norm, 1, x1 - x0[i]).min()
            #print current_min, 
            
            diffs = np.empty(M)
            for j in range(M):
                diffs[j] = np.apply_along_axis(np.linalg.norm, 1, x1- test_points[j]).min()
            max_j_min = np.argmax(diffs)
            #print diffs.shape, current_min.shape
            if diffs[max_j_min] > current_min:
                #print '%s => %s' % (x0[i], test_points[max_j_min])
                x0[i] = test_points[max_j_min]
                changed = True
        #print changed
        if not changed and m > 1:
            break
            
    scale = 1.0 - min(r, 0.5) ** 2

    x0 = x0 * scale  
        
    return x0        
    
    bounds = [(-1, 1) for _ in range(k * 2)]
    res = optimize.minimize(board_separation, x0.ravel(), args=([k]), method='L-BFGS-B' ,bounds=bounds)
    centroids = res.x.reshape((k, 2))
    print centroids
    #exit()
    return centroids
    
    #exit()

def init_board_gauss(N, k, r):
    """Initialize board of N points with k clusters
    """ 
    n = N/k
    X = np.empty((N, 2))
    centroids = np.empty((k, 2))
    
    centroids = calc_centroids(k, r)

    def add_cluster(X, j0, j1, cx, cy, s):
        j = j0
        while j < j1:
            a, b = np.random.normal(cx, s), np.random.normal(cy, s)
            # Continue drawing points from the distribution in the range [-1, 1]
            if abs(a) < 1 and abs(b) < 1:
                X[j, :] = a, b
                j += 1

    def d2(x1, x2): 
        return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2) 

    if False:
        best_dist = 0
        best_clusters = None
        #corners = [(x, y) for x in (-1.0, 1.0) for y in (-1.0, 1.0)] 
        for _ in range(1000):
            labels = [(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) 
                        for _ in range(k)]
            dist = 0
            for i, x in enumerate(labels):
                assert -1 <= x[0] <= 1, x[0]
                assert -1 <= x[1] <= 1, x[1]
                dist += min(x[0] + 1.0, 1.0 - x[0])
                dist += min(x[1] + 1.0, 1.0 - x[1])

                #print dist,
                for j in range(k):
                    if j == i: continue
                    dist += d2(x, labels[j])
                #print dist    
            if dist > best_dist:
                best_dist, best_clusters = dist, labels


    for i, (cx, cy) in enumerate(centroids):
        #cx, cy = best_clusters[i] # np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        s = r # np.random.uniform(r, r)
        j0, j1 = int(round(i * n)), int(round((i + 1) * n))
        add_cluster(X, j0, j1, cx, cy, s)
        #centroids[i] = cx, cy
   
    return X, centroids    
 
 
import matplotlib.pyplot as plt 

random.seed(111) 
np.random.seed(111) 


def pbb(X, c=None):
    return
    (xmin, xmax), (ymin, ymax) = bounding_box(X)
    print '(%5.2f, %5.2f) (%5.2f, %5.2f) %s' % (xmin, xmax, ymin, ymax, X.shape),
    if c:
        cx, cy, s = c
        print '(%5.2f, %5.2f %5.2f)' % (cx, cy, s),
    print    


def closest_indexes(centroids, mu):
    """
        k: number of labels
        centroids: cluster centroids
        mu: detected cluster centers
    """
   
    k = centroids.shape[0]
         
    x = np.empty((k, k, 2))
    for i in 0, 1:
        x[:,:,i] = np.subtract.outer(mu[:,i], centroids[:,i]) 
   
    diffs = np.apply_along_axis(np.linalg.norm, 2, x)
    order = np.argsort(diffs, axis=None)
    if k == 1:
        order = order.ravel()
    #print order.shape    
 
    if False:
        print
        print diffs.shape
        print diffs
        print order.shape
        print order

        for i in range(k):
            for j in range(k):
                x = order[i, j] % k
                y = order[i, j] // k
                print diffs[y, x], mu[y,:], centroids[x,:]

    mu_done = set()
    centroids_done = set()
    mu_indexes = [-1] * k 
    for i in range(k**2):
        x = order[i] % k
        y = order[i] // k
        if y in mu_done or x in centroids_done:
            continue
        mu_indexes[x] = y
        mu_done.add(y)
        centroids_done.add(x)
        #print '%2d %2d %2d %.2f' % (i, y, x , diffs[y, x]), mu[y,:], centroids[x,:]
        if len(mu_done) >= k:
            break

    return mu_indexes    


def test(actual_k, N, r, do_graph=False):

    X, centroids = init_board_gauss(N, actual_k, r)  
    mu, labels = find_centers(X, actual_k) 

    #print 'centroids', centroids.shape # , centroids
    #print 'mu', mu.shape
    #print 'labels', labels.shape
    
    indexes = closest_indexes(centroids, mu)
    mu = mu[indexes]
    labels = labels[indexes]

    #for i in range(actual_k):
    #    print '>>>', centroids[i, :], mu[i, :]
    
    
    #print actual_k, N, r, do_graph
    #print len(centroids), len(mu)
    #for i, (c, m) in enumerate(zip(centroids, mu)):
    #    print c, m

    pbb(X)    

    if do_graph:
        k = actual_k
        color_map = ['c', 'k', 'y', 'm']
        marker_map = ['v', 'o', 's', 'o']
        fig, ax = plt.subplots()
        n0 = 0 
        for i in range(k):
            n = N * (i + 1)/k
            x = X[n0:n, :]
            n0 = n
            pbb(x, centroids[i])
            ax.scatter(x[:, 0], x[:, 1], 
                c=color_map[i % len(color_map)], 
                marker=marker_map[i % len(marker_map)])

        ax.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
        ax.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=1,
                color='b', zorder=11)

        ax.scatter(mu[:, 0], mu[:, 1],
                marker='+', s=169, linewidths=3,
                color='k', zorder=9) 
        ax.scatter(mu[:, 0], mu[:, 1],
                marker='+', s=121, linewidths=1,
                color='r', zorder=10)   

        for i in range(k): 
            ax.plot([centroids[i, 0], mu[i, 0]], [centroids[i, 1], mu[i, 1]], 'r-', lw=3, zorder=20)
            ax.plot([centroids[i, 0], mu[i, 0]], [centroids[i, 1], mu[i, 1]], 'k-', lw=1, zorder=21)


        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_title('Clusters: k=%d, N=%d, r=%.2f' % (k, N, r))
        plt.xlim((-1.0, 1.0))
        plt.ylim((-1.0, 1.0))

        ax.grid(True)
        #fig.tight_layout()    
        plt.show()


    ks, logWks, logWkbs, sk = gap_statistic(X) 
    if False:
        print 'ks', ks 
        print 'logWks', logWks 
        print 'logWkbs', logWkbs
        print 'sk',  sk    

    results = zip(ks, logWks, logWkbs, sk)
    ok = None
    predicted_k = -1

    for i, (k, wk, wkb, sk) in enumerate(results):
        gap = wkb - wk
        if i + 1 < len(results):
            _, wk1, wkb1, sk1 = results[i + 1]  
            gap1 = wkb1 - wk1    
            ok = gap > gap1 - sk1
            #ok = not ok
            if ok and predicted_k < 0:
                predicted_k = k
        #print '%2d %5.2f %5.2f %5.2f : %5.2f %s' % (k, wk, wkb, sk, gap, ok)
        
    correct = predicted_k == actual_k
    print 'k=%d,N=%3d,r=%.2f: predicted_k=%d,correct=%s' % (actual_k, N, r, predicted_k, correct)
    sys.stdout.flush()
    #exit()
    return correct

    
r = 0.1    
#test(4, 200, r, do_graph=True) 
#test(9, 200, r, do_graph=True) 
#test(4, 200, 0.3, do_graph=True) 
#test(7, 200, 0.3, do_graph=True) 
#test(2, 200, r, do_graph=True)  
 
#test(4, 200, r, do_graph=True)    
#test(5, 50, r, do_graph=True) 
#test(5, 50, r, do_graph=True) 
#test(7, 400, r, do_graph=True) 
    
M = 5    
results = []

print('M=%d' % M)

for N in (20, 50, 100, 200, 400, 1e3, 1e4)[4:]:
    for k in (1, 2, 3, 5, 7, 9)[1:]:
        for r in (0.01, 0.1, 0.3, 0.5, 0.5**0.5, 1.0):
            if 5 * (k**2) > N: continue
            if not n_K_0 <= k < n_K_1: continue
            m = sum(test(k, N, r, do_graph=False) for _ in range(M))
            results.append((k, N, r, m))
            print 'k=%d,N=%3d,r=%.2f: %d of %d = %3d%%' % (k, N, r, m, M, int(100.0 * m/ M))
            print '-' * 80
            sys.stdout.flush()

for k, N, r, m in results:
    print 'k=%d,N=%3d,r=%.2f: %d of %d = %3d%%' % (k, N, r, m, M, int(100.0 * m/ M))

