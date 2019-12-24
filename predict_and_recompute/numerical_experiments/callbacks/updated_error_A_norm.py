#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def updated_error_A_norm(**kwargs):
    '''
    callback function to compute A-norm of error at each step
    
    Parameters
    ---------- 
    kwargs['k'] : inetger
                  current iteration
    kwargs['x'] : (n,) array like
                  solution at step k-1
    kwargs['A'] : (n,n) array like
    
    Modifies
    -------
    output['updated_error_A_norm'] : (max_iter,) array like
                             list of A-norm of error at each iteration
    
    Note
    ----
    This varies from error_A_norm in that this is not the A-norm of the actual error, but the A^{-1} norm of the updated residual which the algorithm uses at each step

    '''
    
    output = kwargs['output']
    A = kwargs['A']
    r = kwargs['r_k']
    k = kwargs['k']

    # initialize
    if k==0:
        max_iter = kwargs['max_iter']
        output['updated_error_A_norm'] = np.zeros(max_iter)
        
    # compute A-norm of error
    solver = sp.sparse.linalg.spsolve if sp.sparse.issparse(A) else np.linalg.solve
    error = solver(A.astype(np.double),r.astype(np.double))
    output['updated_error_A_norm'][k] = np.sqrt(error.T@r)
