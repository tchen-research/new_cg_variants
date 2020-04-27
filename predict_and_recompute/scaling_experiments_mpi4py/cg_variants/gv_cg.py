#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np

def gv_cg(comm,A,b,max_iter):
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
    mu = np.zeros((1))

    x = np.zeros_like(b)
    r = np.copy(b)
    p = np.zeros_like(b)
    s = np.zeros_like(b)
    w = np.zeros_like(b)
    u = np.zeros_like(b)
    t_nu_eta_part = np.ones((m*size+2))
    t_nu_eta = np.ones((m*size+2))
    
    t = np.ndarray.view(t_nu_eta[rank*m:(rank+1)*m])
    nu = np.ndarray.view(t_nu_eta[-2:-1]) 
    eta = np.ndarray.view(t_nu_eta[-1:]) 

    t_part = np.ndarray.view(t_nu_eta_part[:m*size])
    nu_part = np.ndarray.view(t_nu_eta_part[-2:-1]) 
    eta_part = np.ndarray.view(t_nu_eta_part[-1:]) 


    # need to set w = Ab : use t_part and t_full for convenience
    t_nu_eta_part[:m*size] = A@r
    comm.Allreduce([t_nu_eta_part,MPI.DOUBLE],[t_nu_eta,MPI.DOUBLE],op=MPI.SUM)
    w[:] = t_nu_eta[rank*m:(rank+1)*m]
    
    # wait for all processes to reach this point, then begin timing
    comm.Barrier()
    if rank==0:
        times['tot'] -= MPI.Wtime()    
    
    for k in range(max_iter):
         
        nu_[:] = nu       
        
        nu_part[:] = np.dot(r,r)
        eta_part[:] = np.dot(r,w)
        t_nu_eta_part[:m*size] = A@w #np.dot(A,w,out=t_part)#_nu_eta_part[:m*size])
        
        comm.Allreduce([t_nu_eta_part,MPI.DOUBLE],[t_nu_eta,MPI.DOUBLE],op=MPI.SUM)

        beta = nu / nu_
        
        p *= beta
        p += r
        s *= beta
        s += w
        u *= beta
        u += t        

        mu = eta - (beta / alpha) * nu if k>0 else eta
        alpha = nu / mu
  
        x += alpha * p
        r -= alpha * s
        w -= alpha * u

    comm.Barrier()
    if rank == 0:
        times['tot'] += MPI.Wtime()    

    return x,times


