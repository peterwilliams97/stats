"""
    http://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
    http://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
"""
from __future__ import division
import sys #, random
import numpy as np
from sklearn.cluster import KMeans
#from collections import defaultdict

print('Numpy: %s' % np.version.version) 


def find_centers(X, k):
    """Find the k cluster in the points in X
        
        X: points 
        k: number of clusters to separate points into
        
        Returns: mu, labels
            mu: centers of clusters
            labels: indexes of points in X belonging to each cluster
    """
    estimator = KMeans(init='k-means++', n_clusters=k, n_init=10)
    estimator.fit(X)
    mu = estimator.cluster_centers_
    labels = estimator.labels_
    return mu, labels

    
def Wk(X, mu, labels):
    """Intra-cluster distances
    
        X: points
        mu: centers of clusters
        labels: indexes of points in X belonging to each cluster
        
        Returns: Normalized intra-cluster distance as defined in 
            http://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
    """
    k = mu.shape[0]
    clusters = [X[np.where(labels == i)] for i in range(k)]
    n = [x.shape[0] for x in clusters]
    return sum(np.linalg.norm(clusters[i] - mu[i])**2/(2*n[i]) for i in range(k))


def bounding_box(X):
    """Bounding box for X
    
        X: points
        
        Returns: (xmin, xmax), (ymin, ymax)
            xmin, xmax and min and max of x coordinates of X
            ymin, ymax and min and max of y coordinates of X
    """
    minmax = lambda x: (x.min(), x.max())    
    return minmax(X[:, 0]), minmax(X[:, 1])


B = 10    
MIN_K = 1
MAX_K = 10

def gap_statistic(X):
    """Calculate gap statistic for X
    
        X: points
        
        Returns: ks, logWks, logWkbs, sk
            ks: k values
            Wks: log(intra-cluster distance) for each k
            Wkbs: average referemce log(intra-cluster distance) for each k
            sk: Normalized std dev log(intra-cluster distance) for each k
    """
    (xmin, xmax), (ymin, ymax) = bounding_box(X)
   
    N = X.shape[0]
    ks = range(MIN_K, MAX_K + 1)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    
    def reference_results(k):
        # Create B reference data sets
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = np.vstack([np.random.uniform(xmin, xmax, N),
                            np.random.uniform(ymin, ymax, N)]).T
            mu, labels = find_centers(Xb, k)
            BWkbs[i] = np.log(Wk(Xb, mu, labels))
        Wkbs_i = sum(BWkbs)/B
        sk_i = np.sqrt(sum((BWkbs - Wkbs_i)**2)/B) * np.sqrt(1 + 1/B)
        return Wkbs_i, sk_i
   
    for indk, k in enumerate(ks):
        mu, labels = find_centers(X, k)
        Wks[indk] = np.log(Wk(X, mu, labels))
        Wkbs[indk], sk[indk] = reference_results(k)    
        
        if False:
            BWkbs = np.zeros(B)
            for i in range(B):
                Xb = np.vstack([np.random.uniform(xmin, xmax, X.shape[0]),
                                np.random.uniform(ymin, ymax, X.shape[0])]).T
                mu, labels = find_centers(Xb, k)
                BWkbs[i] = np.log(Wk(Xb, mu, labels))
            Wkbs[indk] = sum(BWkbs)/B
            sk[indk] = np.sqrt(sum((BWkbs - Wkbs[indk])**2)/B)

    #sk *= np.sqrt(1 + 1/B)

    return ks, Wks, Wkbs, sk    


#mport random
 
def init_board(N):
    return np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])

    
def calc_centroids(k, r):
    """Return best spaced centroids in square of radius 1 around origin
    """

    x0 = np.zeros((k, 2))

    if k == 1:
        return x0

    M = 1000
    x1 = np.zeros((k - 1, 2))
    test_points = np.random.uniform(-1, 1, size=(M, 2))

    for m in range(10):
        changed = False
        for i in range(k):
            x1 = np.vstack((x0[:i, :], x0[i+1:, :]))
            current_min = np.apply_along_axis(np.linalg.norm, 1, x1 - x0[i]).min()
            diffs = np.empty(M)
            for j in range(M):
                diffs[j] = np.apply_along_axis(np.linalg.norm, 1, x1- test_points[j]).min()
            max_j_min = np.argmax(diffs)

            if diffs[max_j_min] > current_min:
                x0[i] = test_points[max_j_min]
                changed = True
        if not changed and m > 1:
            break

    scale = 1.0 - min(r, 0.5) ** 2

    x0 *= scale  

    return x0        

    

def init_board_gauss(N, k, r):
    """Initialize board of N points with k clusters
    """ 

    def add_cluster(X, j0, j1, cx, cy, s):
        j = j0
        while j < j1:
            a, b = np.random.normal(cx, s), np.random.normal(cy, s)
            # Continue drawing points from the distribution in the range [-1, 1]
            if abs(a) < 1 and abs(b) < 1:
                X[j, :] = a, b
                j += 1

    n = N/k
    X = np.empty((N, 2))
    centroids = calc_centroids(k, r)

    for i, (cx, cy) in enumerate(centroids):
        s = r # np.random.uniform(r, r)
        j0, j1 = int(round(i * n)), int(round((i + 1) * n))
        add_cluster(X, j0, j1, cx, cy, s)
   
    return X, centroids    
 
 



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
    if k == 1:
        return [0]
         
    x = np.empty((k, k, 2))
    for i in 0, 1:
        x[:, :, i] = np.subtract.outer(mu[:, i], centroids[:, i]) 
   
    #print centroids
    #print mu
    diffs = np.apply_along_axis(np.linalg.norm, 2, x)
    #print diffs
    order = np.argsort(diffs, axis=None)
    #print order
    #print order.shape    
 
    mu_done = set()
    centroids_done = set()
    mu_indexes = [-1] * k 
    for i in range(k**2):
        c = order[i] % k
        m = order[i] // k
        if c in centroids_done or m in mu_done:
            continue
        #print c, m    
        mu_indexes[c] = m
        centroids_done.add(c)
        mu_done.add(m)
        
        if len(mu_done) >= k:
            break

    return mu_indexes    


color_map = ['c', 'k', 'y', 'm']
marker_map = ['v', 'o', 's']    

def graph_data(k, N, r, X, centroids, mu): 

    import matplotlib.pyplot as plt    
    
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

    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=1, color='b', zorder=11)

    ax.scatter(mu[:, 0], mu[:, 1], marker='+', s=169, linewidths=3, color='k', zorder=9) 
    ax.scatter(mu[:, 0], mu[:, 1], marker='+', s=121, linewidths=1, color='r', zorder=10)   

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


def test(actual_k, N, r, do_graph=False):

    X, centroids = init_board_gauss(N, actual_k, r)  

    if do_graph:
        mu, labels = find_centers(X, actual_k)
        indexes = closest_indexes(centroids, mu)
        graph_data(actual_k, N, r, X, centroids, mu[indexes])
        
    if False:
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
test(4, 200, r, do_graph=True) 
test(9, 200, r, do_graph=True) 
#test(4, 200, 0.3, do_graph=True) 
#test(7, 200, 0.3, do_graph=True) 
#test(2, 200, r, do_graph=True)  
 
#test(4, 200, r, do_graph=True)    
#test(5, 50, r, do_graph=True) 
#test(5, 50, r, do_graph=True) 
#test(7, 400, r, do_graph=True) 
    
M = 1    
results = []

print('M=%d' % M)

for N in (20, 50, 100, 200, 400, 1e3, 1e4)[4:]:
    for k in (1, 2, 3, 5, 7, 9)[1:]:
        for r in (0.01, 0.1, 0.3, 0.5, 0.5**0.5, 1.0):
            if 5 * (k**2) > N: continue
            if not MIN_K <= k < MAX_K: continue
            m = sum(test(k, N, r, do_graph=False) for _ in range(M))
            results.append((k, N, r, m))
            print 'k=%d,N=%3d,r=%.2f: %d of %d = %3d%%' % (k, N, r, m, M, int(100.0 * m/ M))
            print '-' * 80
            sys.stdout.flush()

for k, N, r, m in results:
    print 'k=%d,N=%3d,r=%.2f: %d of %d = %3d%%' % (k, N, r, m, M, int(100.0 * m/ M))

