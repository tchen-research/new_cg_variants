#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
import scipy as sp
import sys

from cg_variants import hs_cg, cg_cg, gv_cg, pr_cg, pipe_pr_cg


"""
Run parallel variants on model problem and return timings

mpiexec -n 2 python scaling_tests.py <n> <max_iter> <trial_name>

n = integer size of model problem
max_iter = number of iterations
trial_name = identifier for save data
"""

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

trial_name = sys.argv[3]
n = int(sys.argv[1])
assert n%size == 0, "n must be a multiple of the number of processes"

if rank == 0:
    print("start setting up on {} ranks".format(size))

# solution is constant vector of unit length
kappa = 1e6
rho = 0.9

lambda1 = 1/kappa
lambdan = 1
Lambda = lambda1+(lambdan-lambda1)*np.arange(n)/(n-1)*rho**np.arange(n-1,-1,-1,dtype='float')
#Lambda = lambda1+(lambdan-lambda1)*np.arange(n)/(n-1)*rho**(n-np.arange(n)-1)
Lambda = lambda1+(lambdan-lambda1)*np.arange(rank*(n//size),(rank+1)*(n//size))/(n-1)*rho**(n-np.arange(rank*(n//size),(rank+1)*(n//size)))

b = np.empty(n//size,dtype='float')
#b = Lambda[rank*(n//size):(rank+1)*(n//size)]
b[:] = Lambda

sparsity_q = False
bandwidth = int(sys.argv[4])

if bandwidth == -1:
    # allocate A as zeros
    A = np.zeros((n,n//size),dtype='float') # maybe make very small in case zeros somehow speed things up

    # fill in diagonal blocks of A with eigenvalues of model problem
    A[rank*(n//size):(rank+1)*(n//size)] += np.diag(b)

else:
    """class block_mat:

        def __init__(self,data):
            self.data = data
      
        def dot(self,x,out=None):

            if len(x.shape) == 1:
                y = np.zeros(n)
            else:
                y = np.zeros((n,x.shape[1]))

            self.data.dot(x,out=y[rank*(n//size):(rank+1)*(n//size)])
            return y
    
    A = block_mat(np.diag(b))
    """

    # half bandwidth
    data = 1e-100*np.ones((2*bandwidth+1,n//size))
    data[0] = b

    offsets = np.hstack([[0],np.arange(1,bandwidth+1),-np.arange(1,bandwidth+1)])
    offsets -= rank*(n//size)
    A = sp.sparse.dia_matrix( (data,offsets), shape=(n,n//size)).tocsr()


# normalize b so solution is constant
b /= np.sqrt(n)

comm.Barrier()
if rank == 0:
    print("done setting up")


variants = [hs_cg,cg_cg,gv_cg,pr_cg,pipe_pr_cg]
#variants = [gv_cg,pipe_pr_cg]

max_iter = int(sys.argv[2])

for variant in variants:
    comm.Barrier()
    sol,t = variant(comm,A,b,max_iter)

    sol_raw = None
    if rank == 0:
        sol_raw = np.empty([size, n//size], dtype='float')
    comm.Gather(sol, sol_raw, root=0)

    if rank==0:

        sol_raw = np.reshape(sol_raw,(n))
        error = np.linalg.norm(np.ones(n)/np.sqrt(n)-sol_raw)

        print("{}, error:{}, time:{}".format(variant.__name__,error,t['tot']))
        
        ## now save results
        res = {"error":error,"timings":t,"sparse":bandwidth}
        np.save("./data/{}/{}_{}".format(n,variant.__name__,trial_name),res,allow_pickle=True)

