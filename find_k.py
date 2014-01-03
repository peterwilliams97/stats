"""
    http://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
    http://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
"""
from __future__ import division
import sys #, random
import numpy as np
from sklearn.cluster import KMeans
#from collections import defaultdict

np.random.seed(111) 

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
            ks: k values  MIN_K <= k <= MAX_K]
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

    return ks, Wks, Wkbs, sk    


#mport random
 
def init_board(N):
    return np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])

 
GRID_NUMBER_TARGET = 1000
GRID_WIDTH = int(np.sqrt(GRID_NUMBER_TARGET))
GRID_NUMBER = GRID_WIDTH**2 
    
UNIFORM_GRID = np.random.uniform(-1, 1, size=(GRID_NUMBER, 2))
print UNIFORM_GRID.shape
xv, yv = np.meshgrid(np.linspace(-1, 1, GRID_WIDTH), np.linspace(-1, 1, GRID_WIDTH))
UNIFORM_GRID = np.empty((GRID_WIDTH * GRID_WIDTH, 2))
UNIFORM_GRID[:, 0] = xv.ravel()
UNIFORM_GRID[:, 1] = yv.ravel()
print UNIFORM_GRID.shape
 
def calc_centroids(k, r):
    """Return best spaced centroids in square of radius 1 around origin
    
     TODO: centroids = > nuclie
           cache this call. See Peter Norton code 
    """

    #scale = 1.0 - min(r, 0.5) ** 2
    scale = 1.0 - abs(min(r, 0.5))

    if k == 1:
        return np.random.uniform(-scale, scale, size=(k, 2))
    
    # Start with centroids near middle of square
    x0 = np.random.uniform(-1.0, 1.0, size=(k, 2))
    
    #print '@@1'

    # Maximize minimum distance between centroids
    x1 = np.empty((k - 1, 2))
    for m in range(10):
        changed = False
        for i in range(k):
            x1 = np.vstack((x0[:i, :], x0[i+1:, :]))
            current_min = np.apply_along_axis(np.linalg.norm, 1, x1 - x0[i]).min()
            diffs = np.empty(GRID_NUMBER)
            for j in range(GRID_NUMBER):
                diffs[j] = np.apply_along_axis(np.linalg.norm, 1, x1 - UNIFORM_GRID[j]).min()
            max_j_min = np.argmax(diffs)

            if diffs[max_j_min] > current_min:
                x0[i] = UNIFORM_GRID[max_j_min]
                changed = True
        #print '@@2', m        
        if not changed and m > 1:
            break

    x0 *= scale  
    return x0        

    

def init_board_gauss(N, k, r):
    """Initialize board of N points with k clusters
    
    TODO: Compute actual centroids
    """ 

    def add_cluster(X, j0, j1, cx, cy, s):
        j = j0
        while j < j1:
            a, b = np.random.normal(cx, s), np.random.normal(cy, s)
            # Continue drawing points from the distribution in the range [-1, 1]
            if abs(a) < 1 and abs(b) < 1:
                X[j, :] = a, b
                j += 1
        return np.mean(X[j0:j1], axis = 0)        

    n = N/k
    X = np.empty((N, 2))
    nuclei = calc_centroids(k, r)
    centroids = np.empty((k, 2))
    labels = np.empty(N)

    for i, (cx, cy) in enumerate(nuclei):
        s = r # np.random.uniform(r, r)
        j0, j1 = int(round(i * n)), int(round((i + 1) * n))
        centroids[i] = add_cluster(X, j0, j1, cx, cy, s)
        labels[j0:j1] = i
   
    return X, centroids, labels    
 
 



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
        mu_indexes[c] = m
        centroids_done.add(c)
        mu_done.add(m)
        if len(mu_done) >= k:
            break

    return mu_indexes    


color_map  = ['b', 'r', 'k', 'y', 'c', 'm']
# http://matplotlib.org/api/markers_api.html
marker_map = ['v', 'o', 's', '^', '<', '>', '8']    

def graph_data(k, N, r, X, centroids, mu, labels, predicted_labels): 
    """
        TODO: Draw circles of radius r around centroids
    """

    import matplotlib.pyplot as plt    
    
    fig, ax = plt.subplots()
    n0 = 0 
    for i in range(k):
        n = N * (i + 1)/k
        x = X[n0:n, :]
        n0 = n
        pbb(x, centroids[i])
        ax.scatter(x[:, 0], x[:, 1], 
            s=50,
            c=color_map[i % len(color_map)], 
            marker=marker_map[i % len(marker_map)])
            
    print labels.shape, predicted_labels.shape
    for i in range(N):    
        if labels[i] != predicted_labels[i]:
            #plt.Circle(X[i, :], radius=0.3, color='k')
            #plt.Circle(X[i, :], radius=0.1, color='b')
            #plt.Circle(X[i, :], radius=0.2, color='r', clip_on=False)
            ax.scatter(X[i, 0], X[i, 1], marker='x', s=100, linewidths=1, color='k', zorder=4)    

    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=169, linewidths=3, color='g', zorder=10)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=121, linewidths=1, color='b', zorder=20)

    ax.scatter(mu[:, 0], mu[:, 1], marker='+', s=169, linewidths=3, color='k', zorder=11) 
    ax.scatter(mu[:, 0], mu[:, 1], marker='+', s=121, linewidths=1, color='r', zorder=21)   

    for i in range(k): 
        x, y = centroids[i, :] 
        dx, dy = mu[i, :] - centroids[i, :] 
        if dx ** 2 + dy **2 >= 0.01:
            ax.arrow(x, y, dx, dy, lw=1, head_width=0.05, length_includes_head=True,
                zorder=9, fc='y', ec='k')
        else:
            print '>>>', dx, dy
        #ax.plot([centroids[i, 0], mu[i, 0]], [centroids[i, 1], mu[i, 1]], 'r-', lw=3, zorder=20)
        #ax.plot([centroids[i, 0], mu[i, 0]], [centroids[i, 1], mu[i, 1]], 'k-', lw=1, zorder=21)

    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_title('Clusters: k=%d, N=%d, r=%.2f' % (k, N, r))
    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))

    ax.grid(True)
    #fig.tight_layout()    
    plt.show()    

    
def find_k(X, verbose=False):

    ks, logWks, logWkbs, sks = gap_statistic(X) 
   
    statistics = zip(ks, logWks, logWkbs, sks)
    predicted_k = -1

    for i in range(len(statistics) - 1):
        k, logWk,  logWkb,  sk  = statistics[i]
        _, logWk1, logWkb1, sk1 = statistics[i + 1]
        gap = logWkb - logWk
        gap1 = logWkb1 - logWk1
        ok = gap > gap1 - sk1
        if ok and predicted_k < 0:
            predicted_k = k
            if not verbose:
                break
        if verbose:    
            print('%3d %5.2f %5.2f %5.2f : %5.2f %s' % (k, logWk, logWkb, sk, gap, ok))
        
    return predicted_k    
    

def test(k, N, r, do_graph=False, verbose=False):

    assert MIN_K <= k <= MAX_K, 'invalid k=%d' % k

    X, centroids, labels = init_board_gauss(N, k, r)  
    #print '@@@'

    if do_graph:
        mu, predicted_labels = find_centers(X, k)
        mu_indexes = closest_indexes(centroids, mu)
        
        print '**', labels.shape, predicted_labels.shape
        
        for i in range(N):
            predicted_labels[i] = mu_indexes[predicted_labels[i]]
        graph_data(k, N, r, X, centroids, mu[mu_indexes], labels, predicted_labels)

    predicted_k = find_k(X, verbose)    
    
    correct = predicted_k == k
    print('k=%d,N=%3d,r=%.2f: predicted_k=%d,correct=%s' % (k, N, r, predicted_k, correct))
    sys.stdout.flush()
    return correct
    

def test_all(verbose=False):

    np.random.seed(111) 
    if False:
        r = 0.1    
        test(10,100, 0.01, do_graph=True)
        test(10,100, 0.2, do_graph=True)
        test(2, 50, 1.0, do_graph=True)
        test(2, 100, 0.25, do_graph=True)
        
         
        test(9, 200, 0.2, do_graph=True) 
        test(4, 200, 0.3, do_graph=True) 
        test(7, 200, 0.3, do_graph=True) 
          
         
        test(4, 200, r, do_graph=True)    
        test(5, 50, r, do_graph=True) 
        test(5, 50, r, do_graph=True) 
        test(7, 400, r, do_graph=True) 
        
    M = 3    
    results = []

    
    print('M=%d' % M)

    for N in (20, 50, 100, 200, 400, 1e3, 1e4)[4:]:
        for k in (1, 2, 3, 5, 7, 9)[1:]:
            for r in (0.01, 0.1, 0.3, 0.5, 0.5**0.5, 1.0):
                if 5 * (k**2) > N: continue
                if not MIN_K <= k <= MAX_K: continue
                m = sum(test(k, N, r, do_graph=False, verbose=verbose) for _ in range(M))
                results.append((k, N, r, m))
                print 'k=%d,N=%3d,r=%.2f: %d of %d = %3d%%' % (k, N, r, m, M, int(100.0 * m/ M))
                print('-' * 80)
                sys.stdout.flush()

    for k, N, r, m in results:
        print 'k=%d,N=%3d,r=%.2f: %d of %d = %3d%%' % (k, N, r, m, M, int(100.0 * m/ M))
        
test_all()
        

