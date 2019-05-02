#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def pipe_m_cg_rr(A,b,x0,max_iter,variant='',rk_replace=lambda k:False,callbacks=[],**kwargs):
    '''
    Pipelined communication hiding conjugate gradient
    (implementation from Chen 2019)
    '''
    
    # get size of problem
    n = len(b)
    
    # initialize
    output = {}
    output['name'] = f"pipe_m_cg{'_'+variant if variant!= '' else''}_rr"
    output['max_iter'] = max_iter
    
    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    p_k    =  np.copy(r_k)
    nu_k   =  r_k   @ r_k
    s_k    =  A     @ p_k
    w_k    =  np.copy(s_k)
    u_k    =  A     @ w_k
    mu_k   =  p_k   @ s_k
    a_k    =  nu_k / mu_k
    a_k1   =  0
    a_k2   =  0
    b_k    =  0
    b_k1   =  0
    del_k  =  r_k   @ s_k
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
        del_k1 =  del_k
        gam_k1 =  gam_k
        
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
        if rk_replace(k):
            r_k = b - A@x_k
            s_k1 = A@p_k1
            w_k = A@r_k
        nu_k   =  - nu_k1 + a_k1**2 * gam_k1
        b_k    =  nu_k / nu_k1
        p_k    =  r_k   +   b_k * p_k1
        s_k    =  w_k   +   b_k * s_k1
        u_k    =  A     @ s_k
        w_k    =  A     @ r_k if variant == 'b' else w_k
        mu_k   =  p_k   @ s_k
        del_k  =  r_k   @ s_k
        gam_k  =  s_k   @ s_k
        nu_k   =  r_k   @ r_k 
        a_k    =  nu_k / mu_k
    
        # call callback functions
        for callback in callbacks:
            callback(**locals())
            
    return output

def pipe_m_cg_b_rr(*args,**kwargs):
    return pipe_m_cg_rr(*args,variant='b',**kwargs)


def pipe_m_pcg_rr(A,b,x0,max_iter,variant='',preconditioner=lambda x:x,rk_replace=lambda k:False,callbacks=[],**kwargs):
    '''
    Pipelined communication hiding conjugate gradient (preconditioned)
    (implementation from Chen 2019)
    '''
    
    # get size of problem
    n = len(b)
    
    # initialize
    output = {}
    output['name'] = f"pipe_m_pcg{'_'+variant if variant!= '' else''}_r"
    output['max_iter'] = max_iter
    
    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    rt_k   =  preconditioner(r_k)
    p_k    =  np.copy(rt_k)
    nu_k   =  rt_k   @ r_k
    s_k    =  A     @ p_k
    st_k   =  preconditioner(s_k)
    w_k    =  np.copy(s_k)
    wt_k   =  np.copy(st_k)
    u_k    =  A     @ st_k
    ut_k   =  preconditioner(u_k)
    mu_k   =  p_k   @ s_k
    a_k    =  nu_k / mu_k
    a_k1   =  0
    a_k2   =  0
    b_k    =  0
    b_k1   =  0
    del_k  =  r_k   @ st_k
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
        del_k1 =  del_k
        gam_k1 =  gam_k
        
        x_k1   =  np.copy(x_k)
        r_k1   =  np.copy(r_k)
        rt_k1  =  np.copy(rt_k)
        w_k1   =  np.copy(w_k)
        wt_k1  =  np.copy(wt_k)
        p_k1   =  np.copy(p_k)
        s_k1   =  np.copy(s_k)
        st_k1  =  np.copy(st_k)
        u_k1   =  np.copy(u_k)
        ut_k1  =  np.copy(ut_k)
        
        # main loop
        x_k    =  x_k1  +   a_k1 * p_k1
        r_k    =  r_k1  -   a_k1 * s_k1
        rt_k   =  rt_k1 -   a_k1 * st_k1       
        w_k    =  w_k1  -   a_k1 * u_k1
        wt_k   =  wt_k1 -   a_k1 * ut_k1
        if rk_replace(k):
            r_k = b - A@x_k
            rt_k = preconditioner(r_k)
#            w_k = A@r_k
#            wt_k = preconditioner(w_k)
            s_k1 = A@p_k1
            st_k1 = preconditioner(s_k1)

        nu_k   =  - nu_k1 + a_k1**2 * gam_k1
        b_k    =  nu_k / nu_k1
        p_k    =  rt_k  +   b_k * p_k1
        s_k    =  w_k   +   b_k * s_k1
        st_k   =  wt_k  +   b_k * st_k1
        u_k    =  A     @ st_k 
        ut_k   =  preconditioner(u_k)
        w_k    =  A     @ rt_k if variant == 'b' else w_k
        wt_k   =  preconditioner(w_k) if variant == 'b' else wt_k
        mu_k   =  p_k   @ s_k
        del_k  =  r_k   @ st_k # or rt_k @ s_k or p_k @ s_k
        gam_k  =  st_k  @ s_k 
        nu_k   =  rt_k  @ r_k 
        a_k    =  nu_k / mu_k
    
        
        # call callback functions
        for callback in callbacks:
            callback(**locals())
            
    return output

def pipe_m_pcg_b_rr(*args,**kwargs):
    return pipe_m_pcg_rr(*args,variant='b',**kwargs)
