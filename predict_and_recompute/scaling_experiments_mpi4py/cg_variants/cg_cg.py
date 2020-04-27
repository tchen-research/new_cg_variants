#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np

def cg_cg(comm,A,b,max_iter):
    size = comm.Get_size()
    rank = comm.Get_rank()

    m = len(b) # = size // n

    if rank == 0:
        times = {'tot':0.,'c_ip':0.,'c_mv':0.,'w_mv':0.,'w_ip':0.,'w_vec':0.}
    else:
        times = None
 
    beta = np.zeros((1))
    alpha = np.zeros((1))
    nu_ = np.zeros((1))
    
    nu_eta = np.ones((2))
    nu_eta_part = np.ones((2))
    
    nu = np.ndarray.view(nu_eta[0:1]) 
    eta = np.ndarray.view(nu_eta[1:2])
    
    nu_part = np.ndarray.view(nu_eta_part[0:1]) 
    eta_part = np.ndarray.view(nu_eta_part[1:2])
    
    mu = np.zeros((1))

    x = np.zeros_like(b)
    r = np.copy(b)
    p = np.zeros_like(b)
    s = np.zeros_like(b)
    w_part = np.zeros((m*size))
    w_full = np.zeros((m*size))
    w = np.ndarray.view(w_full[rank*m:(rank+1)*m])
    
    # wait for all processes to reach this point, then begin timing
    comm.Barrier()
    if rank==0:
        times['tot'] -= MPI.Wtime()    

    for k in range(max_iter):
 
        w_part = A@r#np.dot(A,r,out=w_part)

        comm.Allreduce([w_part,MPI.DOUBLE],[w_full,MPI.DOUBLE],op=MPI.SUM)	

        nu_[:] = nu 
        nu_part[:] = np.dot(r,r)
        eta_part[:] = np.dot(r,w)
        
        comm.Allreduce([nu_eta_part,MPI.DOUBLE],[nu_eta,MPI.DOUBLE],op=MPI.SUM)
        
        beta = nu / nu_
        
        p *= beta
        p += r
        s *= beta
        s += w
  
        mu = eta - (beta / alpha) * nu if k>0 else eta
        alpha = nu / mu
  
        x += alpha * p
        r -= alpha * s

    comm.Barrier()
    if rank == 0:
        times['tot'] += MPI.Wtime()    

    return x,times
