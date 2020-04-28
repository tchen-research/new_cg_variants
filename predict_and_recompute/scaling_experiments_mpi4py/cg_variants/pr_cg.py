#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np

def pr_cg(comm,A,b,max_iter):
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
    nu = np.zeros((1))

    x = np.zeros_like(b)
    r = np.copy(b)
    p = np.copy(b)
    s_part = np.zeros((m*size))
    s_full = np.zeros((m*size))
    s = np.ndarray.view(s_full[rank*m:(rank+1)*m])
    

    data = np.ones((4))
    data_part = np.ones((4)) # u w mu delta gamma nu
    
    mu = np.ndarray.view(data[-4:-3]) 
    delta = np.ndarray.view(data[-3:-2]) 
    gamma = np.ndarray.view(data[-2:-1]) 
    nup = np.ndarray.view(data[-1:]) 

    mu_part = np.ndarray.view(data_part[-4:-3]) 
    delta_part = np.ndarray.view(data_part[-3:-2]) 
    gamma_part = np.ndarray.view(data_part[-2:-1]) 
    nup_part = np.ndarray.view(data_part[-1:]) 

    # wait for all processes to reach this point, then begin timing
    comm.Barrier()
    if rank==0:
        times['tot'] -= MPI.Wtime()    
    
    for k in range(max_iter):
        
        A.dot(p,out=s_part)#s_part = A@p#np.dot(A,p,out=s_part) #s
        comm.Allreduce([s_part,MPI.DOUBLE],[s_full,MPI.DOUBLE],op=MPI.SUM)
        
        mu_part[:] = np.dot(p,s)
        delta_part[:] = np.dot(r,s)
        gamma_part[:] = np.dot(s,s)
        nup_part[:] = np.dot(r,r)
        
        comm.Allreduce([data_part,MPI.DOUBLE],[data,MPI.DOUBLE],op=MPI.SUM)

        nu_[:] = nup       
        
        alpha = nu_ / mu
       
        x += alpha * p
        r -= alpha * s
       
        nu = nu_ - 2 * alpha * delta + alpha**2 * gamma
        beta = nu / nu_
        
        p *= beta
        p += r

    comm.Barrier()
    if rank == 0:
        times['tot'] += MPI.Wtime()    

    return x,times
