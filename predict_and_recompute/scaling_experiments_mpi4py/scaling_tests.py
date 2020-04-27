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

# solution is constant vector of unit length
if rank == 0:
    kappa = 1e6
    rho = 0.9
    
    lambda1 = 1/kappa
    lambdan = 1
    Lambda = lambda1+(lambdan-lambda1)*np.arange(n)/(n-1)*rho**np.arange(n-1,-1,-1,dtype='float')
    sendbuf = Lambda.reshape(size,-1)
else:
    Lambda = None
    sendbuf = None

comm.Barrier()
if rank == 0:
    print("trial name: {}".format(trial_name))
    print("start distributing to {} ranks".format(size))

b = np.empty(n//size,dtype='float')
comm.Scatter(sendbuf,b, root=0)

# allocate A as zeros
A = np.zeros((n,n//size),dtype='float') # maybe make very small in case zeros somehow speed things up

# fill in diagonal blocks of A with eigenvalues of model problem
A[rank*(n//size):(rank+1)*(n//size)] += np.diag(b)


small_off_diagonals = np.zeros_like(A)

if rank == 0:
    small_off_diagonals[rank*(n//size):(rank+1)*(n//size)] += np.diag(np.ones(n//size-1),1)
    small_off_diagonals[rank*(n//size)+1:(rank+1)*(n//size)+1] += np.diag(np.ones(n//size),)
elif rank == size-1:
    small_off_diagonals[rank*(n//size):(rank+1)*(n//size)] += np.diag(np.ones(n//size-1),-1)
    small_off_diagonals[rank*(n//size)-1:(rank+1)*(n//size)-1] += np.diag(np.ones(n//size),)
else:
    small_off_diagonals[rank*(n//size)+1:(rank+1)*(n//size)+1] += np.diag(np.ones(n//size),)
    small_off_diagonals[rank*(n//size)-1:(rank+1)*(n//size)-1] += np.diag(np.ones(n//size),)

# tridiagonal matrix
A_sparse = sp.sparse.csc_matrix(A + 1e-100*small_off_diagonals)

# normalize b so solution is constant
b /= np.sqrt(n)

comm.Barrier()
if rank == 0:
    print("done distributing")


variants = [hs_cg,cg_cg,gv_cg,pr_cg,pipe_pr_cg]


max_iter = int(sys.argv[2])

for variant in variants:
    comm.Barrier()
    sol,t = variant(comm,A,b,max_iter)
    sol_sparse,t_sparse = variant(comm,A_sparse,b,max_iter)

    sol_raw = None
    sol_sparse_raw = None
    if rank == 0:
        sol_raw = np.empty([size, n//size], dtype='float')
        sol_sparse_raw = np.empty([size, n//size], dtype='float')
    comm.Gather(sol, sol_raw, root=0)
    comm.Gather(sol_sparse, sol_sparse_raw, root=0)

    if rank==0:

        sol_raw = np.reshape(sol_raw,(n))
        sol_sparse_raw = np.reshape(sol_sparse_raw,(n))
        error = np.linalg.norm(np.ones(n)/np.sqrt(n)-sol_raw)
        error_sparse = np.linalg.norm(np.ones(n)/np.sqrt(n)-sol_sparse_raw)
        print("{} error: {}, {}".format(variant.__name__,error,error_sparse))

        ## now save results
#        res = {"error":error,"timings":t}
#        np.save("./data/{}/{}_{}".format(n,variant.__name__,trial_name),res,allow_pickle=True)




