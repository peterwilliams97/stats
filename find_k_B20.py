#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Testing the Gap Statistic to find the k in k-means

    http://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
"""
from __future__ import division
import sys
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 


################################################################################ 
#                       ARRAY  MANIPULATION CODE
################################################################################

def norm(a, axis=-1):
    """NumPy 1.8 style norm(). Needed for the NumPy 1.7.1 I am using.
        
        a: A NumPy ndarray
        axis: Axis to calculate norm along. Whole ndarray is normed if axis is None
        
        Returns: norm along axis
    """
    if axis is None:
        return np.linalg.norm(a)
    return np.apply_along_axis(np.linalg.norm, axis, a)


def subtract_outer(a, b):
    """The outer difference of a and b where
        a_b = subtract_outer(a, b) => a_b[i, j, :] = a[i, :] - b[j, :] 

        a: A NumPy ndarray
        b: A NumPy ndarray 
        
        Returns: outer difference of a and b
    """
    assert a.shape[1] == b.shape[1]
    assert len(a.shape) == 2 and len(b.shape) == 2 
    n = a.shape[1]
    a_b = np.empty((a.shape[0], b.shape[0], n))
    for i in xrange(n):
        a_b[:, :, i] = np.subtract.outer(a[:, i], b[:, i])
    return a_b        


################################################################################ 
#                      CODE TO FIND K IN K-MEANS
################################################################################    
    
def find_centers(X, k):
    """Divide the points in X into k clusters
        
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
    """Compute the intra-cluster distances for the k clusters in X described by mu and labels.

        X: points
        mu: centers of clusters
        labels: indexes of points in X belonging to each cluster

        Returns: Normalized intra-cluster distance as defined in 
            http://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
    """
    k = mu.shape[0]
    clusters = [X[np.where(labels == i)] for i in xrange(k)]
    n = [x.shape[0] for x in clusters]
    return sum(norm(clusters[i] - mu[i], None)**2/(2 * n[i]) for i in xrange(k))


def bounding_box(X):
    """Compute the bounding box for the points in X. This is the highest and lowest 
            x and y coordinates of all the points.
    
        X: points
        
        Returns: (xmin, xmax), (ymin, ymax)
            xmin, xmax: min and max of x coordinates of X
            ymin, ymax: min and max of y coordinates of X
    """
    x, y = X[:, 0], X[:, 1]
    return (x.min(), x.max()), (y.min(), y.max()) 


def gap_statistic(X, min_k, max_k, b):
    """Calculate gap statistic for X for k = min_k through k = max_k
        using b reference data sets
    
        X: points
        min_k: lowest k to test
        max_k: highest k to test
        b: number of reference data sets to test against    
 
        Returns: Generator yielding k, logWk, logWkb, sk for min_k <= k <= max_k
            k: This k
            Wks: log(intra-cluster distance) for k
            Wkbs: average reference log(intra-cluster distance) for k
            sk: Normalized std dev log(intra-cluster distance) for k
    """
    N = X.shape[0]
    (xmin, xmax), (ymin, ymax) = bounding_box(X)

    def reference_results(k):
        # Create b reference data sets
        BWkbs = np.zeros(b)
        for i in xrange(b):
            Xb = np.vstack([np.random.uniform(xmin, xmax, N),
                            np.random.uniform(ymin, ymax, N)]).T
            mu, labels = find_centers(Xb, k)
            BWkbs[i] = np.log(Wk(Xb, mu, labels))
        logWkb = np.sum(BWkbs)/b
        sk = np.sqrt(np.sum((BWkbs - logWkb)**2)/b) * np.sqrt(1 + 1/b)
        return logWkb, sk

    for k in xrange(min_k, max_k + 1):
        mu, labels = find_centers(X, k)
        logWk = np.log(Wk(X, mu, labels))
        logWkb, sk = reference_results(k) 
        yield k, logWk, logWkb, sk 


# Parameters for gap statistic calculation
B = 20      # Number of reference data sets
MIN_K = 1   # Lowest k to test
MAX_K = 10  # Highest k to test        

def find_k(X, verbose=1):
    """Find the best k for k-means gap for X using the Gap Statistic
    
        X: points
        verbose: verbosity level: 0, 1 or 2    
 
        Returns: best k if found, otherwise -1
    """
    for i, (k1, logWk1, logWkb1, sk1) in enumerate(gap_statistic(X, MIN_K, MAX_K + 1, B)):
        gap1 = logWkb1 - logWk1
        if i > 0: 
            if verbose >= 2: 
                print('%5d %5.2f %5.2f %5.2f : %5.2f' % (k, logWk, logWkb, sk, gap))
            if gap > gap1 - sk1:
                return k
        k, logWk, logWkb, sk, gap = k1, logWk1, logWkb1, sk1, gap1

    return -1    
    
 
################################################################################ 
#                       TESTING     CODE
################################################################################
 
# GRID_NUMBER is a square number close to GRID_NUMBER_TARGET  
GRID_NUMBER_TARGET = 1000
GRID_WIDTH = int(np.sqrt(GRID_NUMBER_TARGET))
GRID_NUMBER = GRID_WIDTH**2 
 
# UNIFORM_GRID an array of GRID_WIDTH x GRID_WIDTH evenly spaced points on [-1, 1] x [-1, 1]
xv, yv = np.meshgrid(np.linspace(-1, 1, GRID_WIDTH), np.linspace(-1, 1, GRID_WIDTH))
UNIFORM_GRID = np.vstack([xv.ravel(), yv.ravel()]).T

 
def maximally_spaced_points(k, r):
    """Return maximally spaced points in square of radius 1 around origin
        (i.e. square containing x, y such that -1 <= x <= 1, -1 <= y <= 1)
        Try to keep points at least distance r from edges of square 

        k: number of points
        r: desired minimum distance from point to edge of square    
 
        Returns: ndarray of N 2-d points
    """

    if k == 1:
        return np.random.uniform(-min(r, 0.5), min(r, 0.5), size=(k, 2))

    scale = 1.0 - min(r, 0.5)    

    # Start by randomly distributing points over unit radius square
    x0 = np.random.uniform(-1.0, 1.0, size=(k, 2))

    # Maximize minimum distance between centroids
 
    for m in xrange(10):
        changed = False
        for i in xrange(k):
            # Test replacing ith element in x0 all with elements in UNIFORM_GRID to find 
            #  the one that maximizes the minimum distance to elements other than ith in x0 
            # If this minimum distance is greater than current_min then make it the ith element 
            #  in x0    
            x1 = np.vstack((x0[:i, :], x0[i+1:, :]))

            # minimum distance between ith element in x0 and all other elements in x0
            current_min = norm(x1 - x0[i], 1).min()

            # diffs[j] = minimum distance between jth element in UNIFORM_GRID and all elements 
            # in x0 other than ith
            diffs = norm(subtract_outer(UNIFORM_GRID, x1)).min(axis=-1)

            #  max_j_min = index of element in UNIFORM_GRID that maximizes 
            #      minimum distance between ith element in x0 and all other elements in x0
            max_j_min = np.argmax(diffs)

            if diffs[max_j_min] > current_min:
                x0[i] = UNIFORM_GRID[max_j_min]
                changed = True
      
        if not changed and m > 1:
            break

    # Shrink square to get points r-ish distant from edges of unit radius square
    return x0 * scale      


def init_board_gauss(N, k, r):
    """Initialize board of N points with k clusters

        Board is square of radius 1 around origin
        (i.e. square containing x, y such that -1 <= x <= 1, -1 <= y <= 1)
 
        Try to space cluster centers as far apart as possible while keeping them at least distance
        r from edges of unit radius square. This is done in an approximate way by generating 
        random points around maximally spaced nuclei.
  
        N: number of points
        k: number of cluster
        r: desired std dev of points in cluster from cluster center    
 
        Returns: X, centroids, labels 
            X: points
            centroids: centroids of clusters
            labels: cluster index for each point
    """

    def add_cluster(X, j0, j1, cx, cy, s):
        """Add a cluster of normally distributed points to x for indexes [j0,j1) 
            around center cx, cy and std dev s.
            
            X: points
            j0, j1: Add points with indexs j such that j0 <= j < j1
            cx, cy: Centers of normal distrubtion in x any y directions
            s: Desired std dev of normal distrubtion in x any y directions
        """
        j = j0
        while j < j1:
            a, b = np.random.normal(cx, s), np.random.normal(cy, s)
            # Continue drawing points from the distribution in the range (-1, 1)
            if abs(a) < 1 and abs(b) < 1:
                X[j, :] = a, b
                j += 1
        return np.mean(X[j0:j1], axis=0)        

    nuclei = maximally_spaced_points(k, r)

    n = N/k
    X = np.empty((N, 2))
    centroids = np.empty((k, 2))
    labels = np.empty(N, dtype=int)

    for i, (cx, cy) in enumerate(nuclei):
        j0, j1 = int(round(i * n)), int(round((i + 1) * n))
        centroids[i] = add_cluster(X, j0, j1, cx, cy, r)
        labels[j0:j1] = i

    return X, centroids, labels    


def closest_indexes(centroids, mu):
    """Find the elements in centroids that are closest to the elements of mu and 
        return arrays of indexes that 
            map a centroid element to the closest element of mu, and 
            map a mu element to the closest element of centroids

        centroids: ndarray of 2d points
        mu: ndarray of 2d points
        
        Returns: centroid_indexes, mu_indexes
            centroid_indexes[m] is the centroid index corresponding to mu index m
            mu_indexes[c] is the mu index corresponding to centroid index c
    """

    k = centroids.shape[0]
    if k == 1:
        return [0], [0]

    # separations[m, c] = distance between mu[m] and centroid[c]
    separations = norm(subtract_outer(mu, centroids), 2)

    # indexes of diffs in increasing order of distance 
    order = np.argsort(separations, axis=None)
 
    centroids_done = set()
    mu_done = set()
    centroid_indexes = [-1] * k
    mu_indexes = [-1] * k 

    # Go through the mu[m], centroid[c] pairs in order of increasing separation
    # If m and c indexes are not assigned, set centroid_indexes[m] = c and mu_indexes[c] = m
    for i in xrange(k**2):
        c = order[i] % k
        m = order[i] // k
        if c in centroids_done or m in mu_done:
            continue
        centroid_indexes[m] = c
        mu_indexes[c] = m
        centroids_done.add(c)
        mu_done.add(m)
        if len(mu_done) >= k:
            break

    return centroid_indexes, mu_indexes    


def match_clusters(centroids, mu, predicted_labels): 
    """Return versions of mu and predicted_labels that are re-indexed so that 
        mu[i] is closer to centroids[i] than any other element of centroids.
 
        centroids: ndarray of 2d points
        mu: ndarray of 2d points
        predicted_labels: ndarray of integers based on the mu indexes
        
        Returns: mu2, predicted_labels2
            mu2: mu re-indexed as described above
            predicted_labels2: predicted_labels updated for the mu => mu2 re-indexing
    """

    centroid_indexes, mu_indexes = closest_indexes(centroids, mu)

    mu2 = mu[mu_indexes]
    
    predicted_labels2 = np.empty(predicted_labels.shape, dtype=int)
    for i in xrange(predicted_labels.shape[0]):
        predicted_labels2[i] = centroid_indexes[predicted_labels[i]]

    return mu2, predicted_labels2    


def estimate_difficulty(k, X, centroids, labels): 
    """Estimate difficulty of matching
        1) find clusters for known k,
        2) match them to the test clusters and 
        3) find which points don't belong to the clusters they were created for
        This gives a crude measure of how much the test clusters overlap and of 
            how well the detected clusters match the test cluster

        k: Number of clusters in test board
        X: Points in test board 
        centroids: Centroids of clusters in test board
        labels: Centroid labels of X
  
        Returns: mu, different_labels  
            mu: Centroids of attempted clustering of test board
            different_labels: Indexes of points in X for which the attempted clustering 
                gave different labels than board was created with
    """
    mu, predicted_labels = find_centers(X, k)
    mu, predicted_labels = match_clusters(centroids, mu, predicted_labels)
    different_labels = np.nonzero(labels != predicted_labels)[0] 
    return mu, different_labels   


COLOR_MAP  = ['b', 'r', 'k', 'y', 'c', 'm']
# http://matplotlib.org/api/markers_api.html
MARKER_MAP = ['v', 'o', 's', '^', '<', '>', '8']   

def COLOR(i): return COLOR_MAP[i % len(COLOR_MAP)]
def MARKER(i): return MARKER_MAP[i % len(MARKER_MAP)]    


def graph_board(k, N, r, X, centroids, labels, mu, different_labels): 
    """Graph a test board
    
        k, N, r are the instructions for creating the test board
        X, centroids, labels describe the test board that was created
        mu, different_labels are an indication of how difficult the test board is.
            boards with mu a long way from centroids or with a high 
            proprorting of different_labels are expected to be more difficult

        k: Number of clusters in test board
        N: Number of points in test board 
        r: Radius of cluster distributions in test board
        X: Points in test board 
        centroids: Centroids of clusters in test board
        labels: Centroid labels of X
        mu: Centroids of attempted clustering of test board
        different_labels: Indexes of points in X for which the attempted clustering gave different
                labels than board was created with

    """

    fig, ax = plt.subplots()

    for i in xrange(k):
        x = X[np.where(labels == i)]
        ax.scatter(x[:, 0], x[:, 1], s=50, c=COLOR(i), marker=MARKER(i))
 
    for i in different_labels:    
        ax.scatter(X[i, 0], X[i, 1], s=100, c='k', marker='x', linewidths=1, zorder=4)    

    for i in xrange(k): 
        cx, cy = centroids[i, :] 
        mx, my = mu[i, :]
        dx, dy = mu[i, :] - centroids[i, :] 

        ax.scatter(cx, cy, marker='*', s=199, linewidths=3, c='k', zorder=10)
        ax.scatter(cx, cy, marker='*', s=181, linewidths=2, c=COLOR(i), zorder=20)
        ax.scatter(mx, my, marker='+', s=199, linewidths=4, c='k', zorder=11)
        ax.scatter(mx, my, marker='+', s=181, linewidths=3, c=COLOR(i), zorder=21)
        if dx**2 + dy**2 >= 0.001:
            ax.arrow(cx, cy, dx, dy, lw=1, head_width=0.05, length_includes_head=True,
                zorder=9, fc='y', ec='k')

    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_title('Clusters: k=%d, N=%d, r=%.2f, diff=%d (%.2f)' % (k, N, r, 
            different_labels.size, different_labels.size/N))
    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))

    ax.grid(True)
    fig.tight_layout()    
    plt.show()    


def run_test(k, N, r, do_graph=False, verbose=1):
    """Run a test to see if find_k(X) returns the correct number of clusters for 
        test board X created with parameters k, N, r

        k, N, r are the instructions for creating the test board
        
        k: Number of clusters in test board
        N: Number of points in test board 
        r: Radius of cluster distributions in test board
        do_graph: Graph the test board if True
        verbose: verbosity level: 0, 1 or 2

        Returns: correct, n_different
            correct: True if find_k() returned correct k
            n_diffrent: Number points in test board for which the attempted clustering 
                gave different labels than board was created with. This is a measure
                of difficulty
    """

    assert MIN_K <= k <= MAX_K, 'invalid k=%d' % k

    # Create a board of points to test
    X, centroids, labels = init_board_gauss(N, k, r)  
    
    # Do the test!
    predicted_k = find_k(X, verbose)    
    correct = predicted_k == k
   
    # Estimate difficulty
    mu, different_labels = estimate_difficulty(k, X, centroids, labels)
   
    if verbose >= 1:
        print('  k=%d,N=%3d,r=%.2f,diff=%.2f: predicted_k=%d,correct=%s' % (k, N, r, 
                different_labels.size/N, predicted_k, correct))

    if do_graph:
        graph_board(k, N, r, X, centroids, labels, mu, different_labels)
    
    return correct, different_labels.size

    
def test_with_graphs():
    """Run some tests and graph the results
        This lets you see what some typical test boards look like
    """  

    run_test(2,  50, 1.0, do_graph=True)
    run_test(10, 100, 0.01, do_graph=True)
    run_test(2, 100, 0.01, do_graph=True)
    run_test(10, 100, 0.2, do_graph=True)
    run_test(2, 100, 0.25, do_graph=True)
    run_test(9, 200, 0.2, do_graph=True) 
    run_test(4, 200, 0.3, do_graph=True) 
    run_test(7, 200, 0.3, do_graph=True) 
    run_test(4, 200, r, do_graph=True)    
    run_test(5, 50, r, do_graph=True) 
    run_test(5, 50, r, do_graph=True) 
    run_test(7, 400, r, do_graph=True)  


def test_range(n_repeats, verbose=1):
    """Run a range of tests with different test board parameters and printthe results to stdout.
     
        n_repeats: Number of tests to run for each k, N, r combination
        verbose: verbosity level: 0, 1 or 2

        Returns: correct, n_different
            correct: True if find_k() returned correct k
            n_diffrent: Number points in test board for which the attempted clustering 
                gave different labels than board was created with. This is a measure
                of difficulty
    """

    results = []

    print('n_repeats=%d' % n_repeats)
    print('=' * 80)

    # Run tests printing results as we go  
    for N in (20, 50, 100, 200, 400, 10**3, 10**4):
        for k in (1, 2, 3, 5, 7, 9):
            for r in (0.01, 0.1, 0.3, 0.5, 0.5**0.5, 1.0):
                if not MIN_K <= k <= MAX_K: 
                    continue
                corrects, differents = zip(*(run_test(k, N, r, do_graph=False, verbose=verbose) 
                                                  for _ in xrange(n_repeats)))
                n_correct, n_different = sum(corrects), sum(differents) 
                results.append((k, N, r, n_correct, n_different))
                print('k=%d,N=%3d,r=%.2f: %2d of %d = %3d%% (diff=%.2f)' % (k, N, r, 
                            n_correct, n_repeats, int(100.0 * n_correct/n_repeats), 
                            n_different/(n_repeats * N)))
                if verbose >= 1:
                    print('-' * 80)
                sys.stdout.flush()

    # Print summary
    print('=' * 80)
    for k, N, r, n_correct, n_different in results:
        print('k=%d,N=%3d,r=%.2f: %2d of %d = %3d%% (diff=%.2f)' % (k, N, r, 
                    n_correct, n_repeats, int(100.0 * n_correct/n_repeats), 
                    n_different/(n_repeats * N)))


def main():
   
    print(__doc__)
    print('')
    print('NumPy: %s' % np.version.version) 
    
    np.random.seed(111) 
    
    #test_with_graphs()
    n_repeats = 10
    test_range(n_repeats, verbose=1)
    print('')


main()
    
