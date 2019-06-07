#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def pr_master_cg(A,b,x0,max_iter,variant='',callbacks=[],**kwargs):
    '''
    master template for predict-and-recompute conjugate gradients
    (implementation from Chen 2019)
    '''
    
    # get size of problem
    n = len(b)
    
    # initialize
    output = {}
    output['name'] = f'{variant}_cg'
    output['max_iter'] = max_iter
    
    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    nu_k   =  rt_k  @ r_k
    p_k    =  np.copy(r_k)
    s_k    =  A     @ p_k
    mu_k   =  p_k   @ s_k
    a_k    =  nu_k / mu_k
    del_k  =  r_k   @ s_k
    gam_k  =  s_k  @ s_k
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
        del_k1 =  del_k
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
        nu_k   = - nu_k1 + a_k1**2 * gam_k1 if variant == 'm' else \
                 nu_k1 - 2 * a_k1 * del_k1 + a_k1**2 * gam_k1
        b_k    =  nu_k / nu_k1
        p_k    =  r_k   +    b_k * p_k1
        s_k    =  A     @ p_k
        mu_k   =  p_k   @ s_k 
        del_k  =  r_k   @ st_k
        gam_k  =  s_k   @ s_k
        nu_k   =  r_k   @ r_k
        a_k    =  nu_k / mu_k
    
        # call callback functions
        for callback in callbacks:
            callback(**locals())
            
    return output

def pr_cg(A,b,x0,max_iter,callbacks=[],**kwargs):
    '''
    predict-and-recompute conjugate gradient
    (implementation from Chen 2019)
    '''
    return pr_master_cg(A,b,x0,max_iter,callbacks=callbacks,variant='pr',**kwargs)

def m_cg(A,b,x0,max_iter,callbacks=[],**kwargs):
    '''
    predict-and-recompute Meurant conjugate gradient
    (implementation from Chen 2019)
    '''
    return pr_master_cg(A,b,x0,max_iter,callbacks=callbacks,variant='m',**kwargs)


def pr_master_pcg(A,b,x0,max_iter,preconditioner=lambda x:x,variant='',callbacks=[],**kwargs):
    '''
    master template for predict-and-recompute conjugate gradients (preconditioned)
    (implementation from Chen 2019)
    '''
    
    # get size of problem
    n = len(b)
    
    # initialize
    output = {}
    output['name'] = f"{variant}_pcg"
    output['max_iter'] = max_iter
    
    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    rt_k   =  preconditioner(r_k)
    nu_k   =  rt_k  @ r_k
    p_k    =  np.copy(rt_k)
    s_k    =  A     @ p_k
    st_k   =  preconditioner(s_k)
    mu_k   =  p_k   @ s_k
    a_k    =  nu_k / mu_k
    del_k  =  r_k   @ st_k
    gam_k  =  st_k  @ s_k
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
        del_k1 =  del_k
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
        nu_k   = - nu_k1 + a_k1**2 * gam_k1 if variant == 'm' else nu_k1 - 2 * a_k1 * del_k1 + a_k1**2 * gam_k1
        b_k    =  nu_k / nu_k1
        p_k    =  rt_k  +   b_k * p_k1
        s_k    =  A     @ p_k
        st_k   =  preconditioner(s_k)
        mu_k   =  p_k   @ s_k 
        del_k  =  r_k   @ st_k
        gam_k  =  st_k  @ s_k
        nu_k   =  rt_k  @ r_k 
        a_k    =  nu_k / mu_k
    
        # call callback functions
        for callback in callbacks:
            callback(**locals())
            
    return output

def pr_pcg(A,b,x0,max_iter,preconditioner=lambda x:x,callbacks=[],**kwargs):
    '''
    predict-and-recompute conjugate gradient (preconditioned)
    (implementation from Chen 2019)
    '''
    return pr_master_pcg(A,b,x0,max_iter,preconditioner=preconditioner,callbacks=callbacks,variant='pr',**kwargs)

def m_pcg(A,b,x0,max_iter,preconditioner=lambda x:x,callbacks=[],**kwargs):
    '''
    predict-and-recompute Meurant conjugate gradient (preconditioned)
    (implementation from Chen 2019)
    '''
    return pr_master_pcg(A,b,x0,max_iter,preconditioner=preconditioner,callbacks=callbacks,variant='m',**kwargs)
