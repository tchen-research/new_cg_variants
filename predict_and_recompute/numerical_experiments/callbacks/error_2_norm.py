#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def error_2_norm(**kwargs):
    '''
    callback function to compute 2-norm of error at each step
    
    Parameters
    ---------- 
    kwargs['k'] : inetger
                  current iteration
    kwargs['x'] : (n,) array like
                  solution at step k-1
    kwargs['x_true'] : (n,) array like
                       actual solution to system (if this is not provided, the callback function will compute it using kwargs['A'] and kwargs['b'])
    
    Modifies
    -------
    output['error_2_norm'] : (max_iter,) array like
                             list of A-norm of error at each iteration
    
    '''
    
    output = kwargs['output']
    A = kwargs['A']
    x = kwargs['x_k']
    b = kwargs['b']
    k = kwargs['k']

    # check if actual solution is known, otherwise compute it
    if 'x_true' not in kwargs['kwargs'].keys():
#        print('true solution unknown; computing...')
        solver = sp.sparse.linalg.spsolve if sp.sparse.issparse(A) else np.linalg.solve
        kwargs['kwargs']['x_true'] = solver(A.astype(np.double),b.astype(np.double))

    # initialize
    if k==0:
        max_iter = kwargs['max_iter']
        output['error_2_norm'] = np.zeros(max_iter,dtype=A.dtype)
        
    # compute A-norm of error
    error = x - kwargs['kwargs']['x_true'].astype(A.dtype)     
    output['error_2_norm'][k] = np.linalg.norm(error)
