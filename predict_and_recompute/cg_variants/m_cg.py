#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def m_cg(A,b,x0,max_iter,variant='',callbacks=[],**kwargs):
    '''
    Meurant Conjugate Gradient
    (implementation from Chen 2019)
    '''
    
    # get size of problem
    n = len(b)
    
    # initialize
    output = {}
    output['name'] = f'm_cg'
    output['max_iter'] = max_iter
    
    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    w_k    =  A     @ r_k
    p_k    =  np.copy(r_k)
    nu_k   =  r_k   @ r_k
    s_k    =  A     @ p_k
    mu_k   =  p_k   @ s_k
    a_k    =  nu_k / mu_k
    a_k1   =  0
    a_k2   =  0
    b_k    =  0
    b_k1   =  0
    gam_k  =  s_k   @ s_k
    

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
        gam_k1 =  gam_k
        
        x_k1   =  np.copy(x_k)
        r_k1   =  np.copy(r_k)
        w_k1   =  np.copy(w_k)
        p_k1   =  np.copy(p_k)
        s_k1   =  np.copy(s_k)
        
        # main loop
        x_k    =  x_k1  +   a_k1 * p_k1
        r_k    =  r_k1  -   a_k1 * s_k1
        w_k    =  A     @ r_k
        nu_k   =  - nu_k1 + a_k1**2 * gam_k1
        b_k    =  nu_k / nu_k1
        p_k    =  r_k   +    b_k * p_k1
        s_k    =  A     @ p_k
        mu_k   =  p_k   @ s_k 
        gam_k  =  s_k   @ s_k
        nu_k   =  r_k   @ r_k
        a_k    =  nu_k / mu_k
    
        # call callback functions
        for callback in callbacks:
            callback(**locals())
            
    return output


def m_pcg(A,b,x0,max_iter,preconditioner=lambda x:x,variant='',callbacks=[],**kwargs):
    '''
    Meurant Conjugate Gradient (preconditioned)
    (implementation from Chen 2019)
    '''
    
    # get size of problem
    n = len(b)
    
    # initialize
    output = {}
    output['name'] = f'm_pcg'
    output['max_iter'] = max_iter
    
    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    rt_k   =  preconditioner(r_k)
    p_k    =  np.copy(rt_k)
    nu_k   =  rt_k  @ r_k
    s_k    =  A     @ p_k
    st_k   =  preconditioner(s_k)
    mu_k   =  p_k   @ s_k
    a_k    =  nu_k / mu_k
    a_k1   =  0
    a_k2   =  0
    b_k    =  0
    b_k1   =  0
    gam_k  =  st_k  @ s_k
    

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
        gam_k1 =  gam_k
        
        x_k1   =  np.copy(x_k)
        r_k1   =  np.copy(r_k)
        rt_k1  =  np.copy(rt_k)
        p_k1   =  np.copy(p_k)
        s_k1   =  np.copy(s_k)
        st_k1  =  np.copy(st_k)
        
        # main loop
        x_k    =  x_k1  +   a_k1 * p_k1
        r_k    =  r_k1  -   a_k1 * s_k1
        rt_k   =  rt_k1 -   a_k1 * st_k1
        nu_k   =  - nu_k1 + a_k1**2 * gam_k1
        b_k    =  nu_k / nu_k1
        p_k    =  rt_k  +   b_k * p_k1
        s_k    =  A     @ p_k
        st_k   =  preconditioner(s_k)
        mu_k   =  p_k   @ s_k 
        gam_k  =  st_k  @ s_k
        nu_k   =  rt_k  @ r_k 
        a_k    =  nu_k / mu_k
    
        # call callback functions
        for callback in callbacks:
            callback(**locals())
            
    return output
