#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def updated_residual_2_norm(**kwargs):
    '''
    callback function to compute 2-norm of updated residual at each step
    
    Parameters
    ---------- 
    kwargs['k'] : inetger
                  current iteration
    kwargs['r'] : (n,) array like
    
    Modifies
    -------
    output['updated_residual_2_norm'] : (max_iter,) array like
                                        list of 2-norm of updated residual at each iteration
    
    Notes
    -----
    This varies from residual_2_norm in that this is not the 2-norm of the actual residual, but the two norm of the updated residual which the algorithm uses at each step
    '''
    
    output = kwargs['output']
    r = kwargs['r_k']
    k = kwargs['k']

    # check if output has key
    if k==0:
        max_iter = kwargs['max_iter']
        A = kwargs['A']
        output['updated_residual_2_norm'] = np.zeros(max_iter)
        
    # compute A-norm of error 
    output['updated_residual_2_norm'][k] = np.linalg.norm(r)