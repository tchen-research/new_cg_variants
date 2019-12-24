#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np

def pipe_pr_cg(comm,A,b,max_iter):
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
    rs = np.zeros((m,2))
    rs[:,0] = b
    p = np.copy(b)
    r = np.ndarray.view(rs[:,0])
    s = np.ndarray.view(rs[:,1])
    w = np.zeros_like(b)

    data = np.ones((m*size+2,2))
    data_part = np.ones((m*size+2,2)) # w' u mu delta gamma nu'
    
    wp = np.ndarray.view(data[rank*m:(rank+1)*m,0])
    u = np.ndarray.view(data[rank*m:(rank+1)*m,1])
    mu = np.ndarray.view(data[-2:-1,0]) 
    delta = np.ndarray.view(data[-2:-1,1]) 
    gamma = np.ndarray.view(data[-1:,0]) 
    nup = np.ndarray.view(data[-1:,1]) 

    u_wp_part = np.ndarray.view(data_part[:m*size])
    mu_part = np.ndarray.view(data_part[-2:-1,0]) 
    delta_part = np.ndarray.view(data_part[-2:-1,1]) 
    gamma_part = np.ndarray.view(data_part[-1:,0]) 
    nup_part = np.ndarray.view(data_part[-1:,1]) 

    # need to set w = Ab : use t_part and t_full for convenience
    data_part[:m*size,0] = np.dot(A,r)
    comm.Allreduce([data_part,MPI.DOUBLE],[data,MPI.DOUBLE],op=MPI.SUM)
    #w[:] = data[rank*m:(rank+1)*m,0]
    rs[:,1] = data[rank*m:(rank+1)*m,0]
    
    # wait for all processes to reach this point, then begin timing
    comm.Barrier()
    if rank==0:
        times['tot'] -= MPI.Wtime()    
    
    for k in range(max_iter):
        
        mu_part[:] = np.dot(p,s)
        delta_part[:] = np.dot(r,s)
        gamma_part[:] = np.dot(s,s)
        nup_part[:] = np.dot(r,r)
               
        np.dot(A,rs,out=u_wp_part)
        
        comm.Allreduce([data_part,MPI.DOUBLE],[data,MPI.DOUBLE],op=MPI.SUM)

        nu_[:] = nup       
        
        alpha = nu_ / mu
       
        x += alpha * p
        r -= alpha * s
        w = wp - alpha * u
       
        nu = nu_ - 2 * alpha * delta + alpha**2 * gamma
        beta = nu / nu_
        
        p *= beta
        p += r
        s *= beta
        s += w

    comm.Barrier()
    if rank == 0:
        times['tot'] += MPI.Wtime()    

    return x,times
