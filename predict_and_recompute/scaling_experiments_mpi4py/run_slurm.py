import subprocess
import sys
import os

"""
Request nodes from cluster and run scaling_test.py for a range of number of number of processes 
"""

def get_slurm_script(nodes,cores,name,time):
    slurm_settings = \
"""#!/bin/bash
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --job-name={2}
#SBATCH --nodes={0}
#SBATCH --ntasks-per-node={1}
#SBATCH --mem=10G
#SBATCH --time={3}
##SBATCH --workdir=/gscratch/stf/chentyl/new_cg_variants/predict_and_recompute/scaling_experiments_mpi4py
#SBATCH --export=all
module load contrib/anaconda3.4.2
module load icc_18-impi_2018
export MX_RCACHE=0
export I_MPI_ASYNC_PROGRESS=1
 
echo "ENVIORNOMENT VARIBALES"
echo "================================================================================"
env
echo "================================================================================"
echo "MPI SOURCE"
which mpiexec
echo "================================================================================"
""".format(nodes,cores,name,time,1)

    return slurm_settings


n_cores = 8
node_list = [1,2,4,6,8,10,12,16,24,32]
node_list = [1,2,4,6,8,12,16,24,32,48]
node_list = [16,4,24,8,24,32,2,1]
max_trials = 8
n = 1536*8#10#7
max_iter = 1500

time = '30:00'

experiment_dir = "data/{}".format(n)
os.makedirs(experiment_dir, exist_ok=True)

for trial_number in range(max_trials):
    slurm_settings = get_slurm_script(max(node_list),n_cores,'strong_scale',time)
    slurm_script = slurm_settings

    for n_nodes in node_list:

        trial_name = "{}x{}:{}".format(n_nodes,n_cores,trial_number)

        mpi_calls = "mpiexec -n {} python -u scaling_tests.py {} {} {} \n".format(n_nodes*n_cores,n,max_iter,trial_name)

        slurm_script += mpi_calls


    #print(slurm_script)
    p = subprocess.Popen(['sbatch','--account=stf-ckpt','--partition=ckpt'], stdin=subprocess.PIPE)
    p.communicate(slurm_script.encode('utf-8'))

