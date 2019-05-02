#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
import mpmath as mp

def mp_cg(A,b,x0,max_iter,callbacks=[],**kwargs):
    '''
    Hestenes and Stiefel Conjugate Gradient
    (implementation from Greenbaum, Liu, Chen 2019)

    Parameters
    ----------           
    A : (n,n) array_like
        SPD matrix from system $Ax=b$
    b : (n,) array_like
        right hand side from system $Ax=b$
    x0 : (n,) array_like
         Initial guess to solution $Ax=b$
    max_iter : integer
               number of iterations
    callbacks : list of functions
                each callback function will be called on all local variables each iteration
    
    Returns
    -------
    Bunch object with fields defined based on which callback functions were listed in callbacks
    
    Notes
    -----
    '''
    
    # get true solution
    if 'x_true' not in kwargs.keys():
        print('solving for true solution')
        x_true = mp.qr_solve(A,b)[0]
    else:
        x_true = kwargs['x_true']

    # get size of problem
    n = len(b)
    
    # initialize
    output = {}
    output['name'] = 'exact_cg'
    output['max_iter'] = max_iter
    
    # sort of round about way to get callbacks
    cb_names = [cb.__name__ for cb in callbacks]
    if 'error_A_norm' in cb_names:
        output['error_A_norm'] = np.zeros(max_iter)
    if 'residual_2_norm' in cb_names:
        output['residual_2_norm'] = np.zeros(max_iter)
    if 'updated_residual_2_norm' in cb_names:
        output['updated_residual_2_norm'] = np.zeros(max_iter)
    
    x = x0.copy()
    r = (b - A * x).copy()
    
    p = r.copy()
    nu_k = (r.T * r)[0,0]
    
    s = A * p

    a_k1 = nu_k / (p.T * s)[0,0]
    b_k = 0

    # run main optimization
    for k in range(1,max_iter):
               
        # call callback functions before x and r are updated
        if 'error_A_norm' in cb_names:
            error = x-x_true
            eAe = (error.T*(A*error))[0,0]
            output['error_A_norm'][k-1] = float(mp.sqrt(eAe)) if eAe >=0 else -1
        if 'residual_2_norm' in cb_names:
            output['residual_2_norm'][k-1] = float(mp.norm(b-A*x))
        if 'updated_residual_2_norm' in cb_names:
            output['updated_residual_2_norm'][k-1] = float(mp.norm(r))
        
        x += a_k1 * p
        r -= a_k1 * s
            
        nu_k1 = nu_k
        nu_k = (r.T * r)[0,0]
        
        b_k = nu_k/nu_k1
        
        p = r + b_k * p
        
        s = A * p
        
        a_k2 = a_k1
        a_k1 = nu_k / (p.T * s)[0,0]
    
    # call callback functions one last time
    k+=1
    if 'error_A_norm' in cb_names:
        error = x-x_true
        eAe = (error.T*(A*error))[0,0]
        output['error_A_norm'][k-1] = float(mp.sqrt(eAe)) if eAe >=0 else -1    
    if 'residual_2_norm' in cb_names:
        output['residual_2_norm'][k-1] = float(mp.norm(b-A*x))
    if 'updated_residual_2_norm' in cb_names:
        output['updated_residual_2_norm'][k-1] = float(mp.norm(r))

    return output