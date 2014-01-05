#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Analyze results returns by find_k

    usage: python k_results.py <results file> [sort key(s)]

    Sort keys: k n r m d
    Lower-case sorts in increasing order
    Upper-case sorts in decreasing order

    k: number of clusters
    n: total number of points
    r: cluster radius (~ std dev)
    m: number of correct k estimates
    d: a measure of difficulty, higher is more difficult
    
    e.g. 
        python k_results.py results.txt k N
            sorts results.txt by number of clusters in increasing order then by number of points
                in decreasing order
                
        python k_results.py results.txt m r
            sorts results.txt by number of correct results in increasing order then by cluster
                radius in decreasing order        

"""
from __future__ import division
import re, sys

def is_upper(s):
    return s.upper() == s
    

DIVIDER = '=' * 80
RE_RESULT = re.compile(r'''
    \s*k\s*=\s*(\d+)\s*,
    \s*N\s*=\s*(\d+)\s*,
    \s*r\s*=\s*(\d*\.\d*)\s*:
    \s*(\d+)\s*of\s*(\d+)
    \s*=\s*(\d+)%\s* 
    \(\s*diff\s*=\s*(\d*\.\d*)\s*\)
    ''',
    re.VERBOSE)

CONVERTERS = [
    lambda s: int(s),
    lambda s: int(s),
    lambda s: float(s),
    lambda s: int(s),
    lambda s: int(s),
    lambda s: int(s)/100.0,
    lambda s: float(s),
]    

SORT_KEY_NAMES = ['k', 'n', 'r', 'm', 'd']

def make_sort_key(options):
    """Make results sort key described in module doc string
    """
    sort_order = SORT_KEY_NAMES[:]
    sort_order_lower = [s.lower() for s in sort_order]

    for s in reversed(options):
        if not s.lower() in sort_order_lower:
            continue
        i = sort_order_lower.index(s.lower())
        sort_order = [s] + sort_order[:i] + sort_order[i + 1:]
        sort_order_lower = [s.lower() for s in sort_order]    

    return [(SORT_KEY_NAMES.index(s.lower()), -1 if is_upper(s) else 1) for s in sort_order]
 

def sort_func(result):
    """Sort function for results described in module doc string
    """
    return [result[i] * sign for i, sign in sort_key] 


def read_results(path):    
    results = []
    num_dividers = 0

    with open(path, 'rt') as f:
        for i, line in enumerate(f):
            line = line.rstrip('\r\n')
            if line.startswith(DIVIDER):
                num_dividers += 1
                continue
            elif num_dividers < 2:
                continue
            m = RE_RESULT.search(line) 
            if not m:
                continue
            k, N, r, m, M, mM, diff = [CONVERTERS[i](m.group(i + 1)) for i in range(7)]
            n_correct = m
            n_repeats = M
            n_different = (int)(n_repeats * N * diff)
            results.append((k, N, r, n_correct, n_different))
            
    return results, n_repeats    

  
if len(sys.argv) < 1:
    print (__doc__)
    exit(1)

path = sys.argv[1]
options = [s[0] for s in sys.argv[2:]]  
 
results, n_repeats = read_results(path)
sort_key = make_sort_key(options)     
results.sort(key=sort_func)

for k, N, r, n_correct, n_different in results:
    print('k=%d,N=%3d,r=%.2f: %2d of %d = %3d%% (diff=%.2f)' % (k, N, r, 
            n_correct, n_repeats, int(100.0 * n_correct/n_repeats), 
            n_different/(n_repeats * N)))
            
