#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def lanczos_recurrence(**kwargs):
    """
    callback to compute quantities about the Lanczos recurrence
    
    Parameters
    ---------- 
    kwargs['k'] : inetger
                  current iteration
    kwargs['r'] : (n,) array like
    kwargs['ak1'] : 
    kwargs['ak2'] :
    kwargs['bk'] :
    kwargs['bk1'] :
    
    Modifies
    -------
    output['lanczos_z'] : (max_iter,n) array like
                          list of lanczos vectors
    output['lanczos_alpha'] : (max_iter,) array like
                              list of "alpha" for Lanczos recurrence
    output['lanczos_beta'] : (max_iter-1,) array like
                             list of "beta" for Lanczos recurrence
    output['lanczos_3_term_error'] : (max_iter-1,) array like
                                     2-norm of error in 3 term Lanczos recurrence
    output['lanczos_orthogonality] : (max_iter-1,) array like
                                     orthogonality of successive Lanczos vectors
                                     
    Notes
    -----
    could rearange the logic control staements to reduce the length but I'm not sure if it would increase performance or clarity

    if only the 3 term error and orthogonality are wanted, then maybe better to not save the whole set of lanczos vectors for storage reasons
    """
    
    output = kwargs['output']
    max_iter = kwargs['max_iter']
    k = kwargs['k']
    r = kwargs['r_k']
    a_k1 = kwargs['a_k1']
    b_k1 = kwargs['b_k1']
    
        
    if k==0:
        A = kwargs['A']
        output['lanczos_alpha'] = np.zeros((max_iter),dtype=A.dtype)
        output['lanczos_beta'] = np.zeros((max_iter),dtype=A.dtype)
        output['lanczos_z'] = np.zeros((len(r),max_iter),dtype=A.dtype)
        
        output['lanczos_z'][:,k] = (-1)**(k)*r/np.linalg.norm(r)

    elif k<max_iter:
        r_k1 = kwargs['r_k1']
        a_k2 = kwargs['a_k2']
        
        output['lanczos_alpha'][k-1] = 1/a_k1 + b_k1/a_k2 if k>1 else 1/a_k1
        output['lanczos_beta'][k-1] = np.linalg.norm(r)/(a_k1 * np.linalg.norm(r_k1))
        output['lanczos_z'][:,k] = (-1)**(k)*r/np.linalg.norm(r)

    if k==max_iter-1:
        # construct tridiagonal matrix T
        T = sp.sparse.diags([output['lanczos_alpha'],output['lanczos_beta'][:max_iter-2],output['lanczos_beta'][:max_iter-1]],[0,1,-1],shape=(max_iter,max_iter-1))
            
        # compute error from exact 3-term recusion
        A = kwargs['A']
        Z = output['lanczos_z']
        E = A@Z[:,:-1]-Z@T
        
        output['lanczos_3_term_error'] = np.linalg.norm(E,axis=0)
        output['lanczos_orthogonality'] = np.abs(np.einsum('ji,ji->i',output['lanczos_beta'][:max_iter-1]*Z[:,:-1],Z[:,1:]))
