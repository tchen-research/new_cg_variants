#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def gv_cg(A,b,x0,max_iter,w_replace=(lambda **kwargs: False),callbacks=[],**kwargs):
    '''
    Ghysels and Vanroose (piplined) Conjugate Gradient
    (implementation from Greenbaum, Liu, Chen 2019)
    '''
    
    # get size of problem
    n = len(b)
    
    # initialize
    output = {}
    output['name'] = 'gv_cg'
    output['max_iter'] = max_iter 
    
    wk_replace = np.zeros(max_iter,dtype=bool)

    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    w_k    =  A     @ r_k
    p_k    =  np.copy(r_k)
    s_k    =  np.copy(w_k)
    u_k    =  A     @ w_k
    nu_k   =  r_k   @ r_k
    eta_k  =  w_k   @ r_k
    mu_k   =  p_k   @ s_k
    a_k    =  nu_k / mu_k
    a_k1   =  0
    a_k2   =  0
    b_k    =  0
    b_k1   =  0
    
    wk_replace_flags = {} # this can be used to store data between iterations for wk_replace functions
    
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
        eta_k1 =  eta_k
        mu_k1  =  mu_k
        
        x_k1   =  np.copy(x_k)
        r_k1   =  np.copy(r_k)
        w_k1   =  np.copy(w_k)
        p_k1   =  np.copy(p_k)
        s_k1   =  np.copy(s_k)
        u_k1   =  np.copy(u_k)
        
        # main loop
        x_k    =  x_k1  +   a_k1 * p_k1
        r_k    =  r_k1  -   a_k1 * s_k1
        w_k    =  w_k1  -   a_k1 * u_k1
        
        if w_replace(k=k,A=A,b=b,x=x_k,w=w_k,r=r_k,r_=r_k1,u=u_k,s=s_k,p=p_k,wk_replace_flags=wk_replace_flags):
            wk_replace[k] = True
            w_k = A @ r_k

        t_k    =  A     @ w_k
        nu_k   =  r_k   @ r_k
        eta_k  =  w_k   @ r_k
        b_k    =  nu_k / nu_k1
        p_k    =  r_k   +   b_k * p_k1
        s_k    =  w_k   +   b_k * s_k1
        u_k    =  t_k   +   b_k * u_k1
        mu_k   =  eta_k - (b_k / a_k1) * nu_k
        a_k    =  nu_k / mu_k 
 
        # call callback functions
        for callback in callbacks:
            callback(**locals())
            
    return output

def gv_pcg(A,b,x0,max_iter,w_replace=(lambda **kwargs: False),preconditioner=lambda x:x,callbacks=[],**kwargs):
    '''
    Ghysels and Vanroose (piplined) Preconditioned Conjugate Gradient
    (implementation from Greenbaum, Liu, Chen 2019)
    '''
    
    # get size of problem
    n = len(b)
    
    # initialize
    output = {}
    output['name'] = 'gv_pcg'
    output['max_iter'] = max_iter 
    
    wk_replace = np.zeros(max_iter,dtype=bool)

    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    rt_k   =  preconditioner(r_k)
    w_k    =  A     @ rt_k
    wt_k   =  preconditioner(w_k)
    p_k    =  np.copy(rt_k)
    s_k    =  np.copy(w_k)
    st_k   =  np.copy(wt_k)
    u_k    =  A     @ wt_k
    nu_k   =  r_k   @ rt_k
    eta_k  =  w_k   @ r_k
    mu_k   =  p_k   @ s_k
    a_k    =  nu_k / mu_k
    a_k1   =  0
    a_k2   =  0
    b_k    =  0
    b_k1   =  0
    
    wk_replace_flags = {} # this can be used to store data between iterations for wk_replace functions
    
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
        eta_k1 =  eta_k
        mu_k1  =  mu_k
        
        x_k1   =  np.copy(x_k)
        r_k1   =  np.copy(r_k)
        rt_k1  =  np.copy(rt_k)
        w_k1   =  np.copy(w_k)
        wt_k1  =  np.copy(wt_k)
        p_k1   =  np.copy(p_k)
        s_k1   =  np.copy(s_k)
        st_k1  =  np.copy(st_k)
        u_k1   =  np.copy(u_k)
        
        # main loop
        x_k    =  x_k1  +   a_k1 * p_k1
        r_k    =  r_k1  -   a_k1 * s_k1
        rt_k   =  rt_k1 -   a_k1 * st_k1
        w_k    =  w_k1  -   a_k1 * u_k1
        
        if w_replace(k=k,A=A,b=b,x=x_k,w=w_k,r=r_k,r_=r_k1,u=u_k,s=s_k,p=p_k,wk_replace_flags=wk_replace_flags):
            wk_replace[k] = True
            w_k = A @ r_k

        wt_k   =  preconditioner(w_k)
        t_k    =  A     @ wt_k
        nu_k   =  r_k   @ rt_k
        eta_k  =  w_k   @ rt_k
        b_k    =  nu_k / nu_k1
        p_k    =  rt_k  +   b_k * p_k1
        s_k    =  w_k   +   b_k * s_k1
        st_k   =  wt_k  +   b_k * st_k1
        u_k    =  t_k   +   b_k * u_k1
        mu_k   =  eta_k - (b_k / a_k1) * nu_k
        a_k    =  nu_k / mu_k 
 
        # call callback functions
        for callback in callbacks:
            callback(**locals())
            
    return output