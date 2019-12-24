#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def save_x(**kwargs):
    '''
    callback function to save the updated solution at each step
    
    Parameters
    ---------- 
    kwargs['k'] : inetger
                  current iteration
    kwargs['x'] : (n,) array like
    
    Modifies
    -------
    output['x'] : (max_iter,n) array like
                  list of iterates
    
    '''
    
    output = kwargs['output']
    x = kwargs['x_k']
    k = kwargs['k']

    # initialize
    if k==0:
        max_iter = kwargs['max_iter']
        A = kwargs['A']
        output['x'] = np.zeros((max_iter,len(x)),dtype=A.dtype)
        
    # compute A-norm of error 
    output['x'][k] = x