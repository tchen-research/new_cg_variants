#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def residual_2_norm(**kwargs):
    '''
    callback function to compute 2-norm of residual at each step
    
    Parameters
    ---------- 
    kwargs['k'] : inetger
                  current iteration
    kwargs['x'] : (n,) array like
                  solution at step k-1
    kwargs['A'] : (n,n) array like
    kwargs['b'] : (n,) array like
    
    Modifies
    -------
    output['residual_2_norm'] : (max_iter,) array like
                                list of 2-norm of residual at each iteration
    
    '''
    
    output = kwargs['output']
    A = kwargs['A']
    x = kwargs['x_k']
    b = kwargs['b']
    k = kwargs['k']

    # initialize
    if k==0:
        max_iter = kwargs['max_iter']
        output['residual_2_norm'] = np.zeros(max_iter)
        
    # compute A-norm of error 
    output['residual_2_norm'][k] = np.linalg.norm(b-A@x)