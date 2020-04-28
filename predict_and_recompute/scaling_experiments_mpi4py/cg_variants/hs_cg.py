#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np

def hs_cg(comm,A,b,max_iter):
    size = comm.Get_size()
    rank = comm.Get_rank()

    m = len(b) # = n // size 

    if rank == 0:
        times = {'tot':0.,'c_ip':0.,'c_mv':0.,'w_mv':0.,'w_ip':0.,'w_vec':0.}
    else:
        times = None
 
    beta = np.zeros((1))
    alpha = np.zeros((1))
    nu_ = np.ones((1))
    nu = np.ones((1))
    mu = np.ones((1))

    x = np.zeros_like(b)
    r = np.copy(b)
    p = np.zeros_like(b)
    s_part = np.zeros((m*size))
    s_full = np.zeros((m*size))
    s = np.ndarray.view(s_full[rank*m:(rank+1)*m])
 
    # wait for all processes to reach this point, then begin timing
    comm.Barrier()
    if rank==0:
        times['tot'] -= MPI.Wtime()    

    for k in range(max_iter):
        
        nu_[:] = nu
        
        nu_part = np.dot(r,r)
        
        comm.Allreduce([nu_part,MPI.DOUBLE],[nu,MPI.DOUBLE],op=MPI.SUM)

        beta = nu / nu_
        
        p *= beta
        p += r
        
        A.dot(p,out=s_part)#s_part = A@p

        comm.Allreduce([s_part,MPI.DOUBLE],[s_full,MPI.DOUBLE],op=MPI.SUM)

        mu_part = np.dot(p,s)

        comm.Allreduce([mu_part,MPI.DOUBLE],[mu,MPI.DOUBLE],op=MPI.SUM)

        alpha = nu / mu
 
        x += alpha * p
        r -= alpha * s 
   
    comm.Barrier()
    if rank == 0:
        times['tot'] += MPI.Wtime()    

    return x,times
    
