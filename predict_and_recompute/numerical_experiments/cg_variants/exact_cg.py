#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def exact_cg(A,b,x0,max_iter,callbacks=[],**kwargs):
    '''
    Hestenes and Stiefel Conjugate Gradient (with reorthogonalization)
    (implementation from Greenbaum, Liu, Chen 2019)
    '''
    
    # get size of problem
    n = len(b)
    
    # initialize
    output = {}
    output['name'] = 'exact_cg'
    output['max_iter'] = max_iter
    
    R = np.zeros((min(n,max_iter),n),dtype=A.dtype)

    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    R[0]   =  r_k / np.linalg.norm(r_k) 
    p_k    =  np.copy(r_k)
    nu_k   =  r_k @ r_k
    s_k    =  A @ p_k
    mu_k   =  p_k @ s_k
    a_k    =  nu_k / mu_k
    a_k1   =  0
    a_k2   =  0
    b_k    =  0
    b_k1   =  0
        
    k=0
    for callback in callbacks:
        callback(**locals())
    
    # run main optimization
    for k in range(1,max_iter):
        
        # update indexing
        a_k2   =  a_k1
        a_k1   =  a_k
        b_k1   =  b_k
        nu_k1  =  nu_k
        mu_k1  =  mu_k
        
        x_k1   =  np.copy(x_k)
        r_k1   =  np.copy(r_k)
        p_k1   =  np.copy(p_k)
        s_k1   =  np.copy(s_k)

        # main loop
        x_k    =  x_k1  + a_k1 * p_k1
        r_k    =  r_k1  - a_k1 * s_k1
        for j in range(k): # enforce orthogonality of residuals
            r_k -= (R[j]@r_k)*R[j] 
        R[k] = r_k / np.linalg.norm(r_k)
        nu_k   =  r_k @ r_k
        b_k    =  nu_k/nu_k1
        p_k    =  r_k   + b_k * p_k1
        s_k    =  A   @ p_k
        mu_k   =  p_k @ s_k
        a_k    =  nu_k / mu_k
        
        # call callback functions
        for callback in callbacks:
            callback(**locals())
        
    return output

def exact_pcg(A,b,x0,max_iter,preconditioner=lambda x:x,callbacks=[],**kwargs):
    '''
    Hestenes and Stiefel Conjugate Gradient (with reorthogonalization)
    (implementation from Greenbaum, Liu, Chen 2019)
    '''
    
    # get size of problem
    n = len(b)
    
    # initialize
    output = {}
    output['name'] = 'exact_pcg'
    output['max_iter'] = max_iter
    
    R = np.zeros((min(n,max_iter),n),dtype=A.dtype)
    Rt = np.zeros((min(n,max_iter),n),dtype=A.dtype)

    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    rt_k   =  preconditioner(r_k)
    p_k    =  np.copy(rt_k)
    nu_k   =  r_k   @ rt_k
    R[0]   =  r_k / np.sqrt(nu_k)
    Rt[0]  =  rt_k / np.sqrt(nu_k)
    s_k    =  A     @ p_k
    mu_k   =  p_k   @ s_k
    a_k    =  nu_k / mu_k
    a_k1   =  0
    a_k2   =  0
    b_k    =  0
    b_k1   =  0

    tol = np.sqrt(nu_k) * 1e-14
        
    k=0
    for callback in callbacks:
        callback(**locals())
    
    # run main optimization
    for k in range(1,max_iter):
        
        # update indexing
        a_k2   =  a_k1
        a_k1   =  a_k
        b_k1   =  b_k
        nu_k1  =  nu_k
        mu_k1  =  mu_k
        
        x_k1   =  np.copy(x_k)
        r_k1   =  np.copy(r_k)
        rt_1k  =  np.copy(rt_k)
        p_k1   =  np.copy(p_k)
        s_k1   =  np.copy(s_k)

        # main loop
        x_k    =  x_k1  + a_k1 * p_k1
        r_k    =  r_k1  - a_k1 * s_k1
        for j in range(k): # residuals are M^{-1} orthogonal
            r_k -= (Rt[j]@r_k)*R[j]
        rt_k   =  preconditioner(r_k)
        nu_k   =  r_k   @ rt_k
        R[k]   =  r_k  / np.sqrt(nu_k)
        Rt[k]  =  rt_k / np.sqrt(nu_k)
        b_k    =  nu_k / nu_k1
        p_k    =  rt_k  +   b_k * p_k1
        s_k    =  A     @ p_k
        mu_k   =  p_k   @ s_k
        a_k    =  nu_k / mu_k
        
        # call callback functions
        for callback in callbacks:
            callback(**locals())

        if np.sqrt(nu_k) < tol:
            break

        if k%10==0:
            print(np.sqrt(nu_k),tol)
        
    return output
