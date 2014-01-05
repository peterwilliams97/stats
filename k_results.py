#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Sort keys: k n r m d
    Lower-case sorts in increasing order
    Upper-case sorts in decreasing order

    k: number of clusters
    n: total number of points
    r: radius of clusters (~ std dev)
    m: number of correct k estimates
    d: a measure of difficulty, higher is more difficult


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
    
path = sys.argv[1]
options = reversed([s[0] for s in sys.argv[2:]])

SORT_KEYS = ['k', 'n', 'r', 'm', 'd']
sort_order = SORT_KEYS[:]


for s in options:
    if not s.lower() in sort_order:
        continue
    i = sort_order.index(s.lower())
    sort_order = [s] + sort_order[:i] + sort_order[i + 1:]  

sort_indexes = [(SORT_KEYS.index(s.lower()), -1 if is_upper(s) else 1) for s in sort_order]
 
if False:    
    print sort_order
   
    print sort_indexes
    exit()    

def sort_func(result):
    return [result[i] * sign for i, sign in sort_indexes] 


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
            #print 'line %d: "%s"' % (i, line)
            continue
        k, N, r, m, M, mM, diff = [CONVERTERS[i](m.group(i + 1)) for i in range(7)]
        n_correct = m
        n_repeats = M
        n_different = (int)(n_repeats * N * diff)
        results.append((k, N, r, n_correct, n_different))
print num_dividers
results.sort(key=sort_func)

for k, N, r, n_correct, n_different in results:
        print('k=%d,N=%3d,r=%.2f: %2d of %d = %3d%% (diff=%.2f)' % (k, N, r, 
                    n_correct, n_repeats, int(100.0 * n_correct/n_repeats), 
                    n_different/(n_repeats * N)))
    
        