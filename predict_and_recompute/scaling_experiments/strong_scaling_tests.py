import subprocess
import sys
import os


def get_slurm_script(nodes,cores,name,time):
    slurm_settings = \
"""#!/bin/bash

#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --job-name={2}
#SBATCH --nodes={0}
###SBATCH --ntasks-per-node={1}
#SBATCH --mem=10G
#SBATCH --time={3}
#SBATCH --workdir=/gscratch/stf/chentyl/cg_scaling_tests/
#SBATCH --export=all


module load contrib/anaconda4.4.0
module load contrib/gcc/6.2.0_mpich-3.2

export MX_RCACHE=0

export MPICH_ASYNC_PROGRESS=1
export MPICH_MAX_THREAD_SAFETY=multiple
 
echo "ENVIORNOMENT VARIBALES"
echo "================================================================================"
env
echo "================================================================================"
echo "MPI SOURCE"
which mpirun
echo "================================================================================"
""".format(nodes,cores,name,time,1)

    return slurm_settings


def get_mpi_call(n_nodes,n_cores,variant,max_iter,mesh_pts,trial_name):
    
    pc_type = "jacobi"
    
    opt_args = ''
    if variant == "pipeprcg_0":
        opt_args = "-recompute_q 0"
        ksp_type = "pipeprcg"
    else:
        ksp_type = variant

    problem_setup = "-m {0} -n {0} -pc_type {1} -ksp_type {2} {3} ".format(mesh_pts,pc_type,ksp_type,opt_args)
    iters = "-ksp_norm_type none -ksp_max_it {0} ".format(max_iter)
    log_view = "-ksp_view -log_view :./logs/{0}/{1}/{2:02d}/{3}.xml:ascii_xml".format(variant,mesh_pts,n_nodes,trial_name)

    return "\nmpirun -genvall -binding socket -map-by hwthread -n {0} -ppn {1} ./ex2 ".format(n_nodes*n_cores,n_cores)+problem_setup+iters+log_view


##
## STRONG SCALING EXPERIMENT
##
def strong_scale(node_list,n_cores,variants,max_iter,mesh_pts,time):
    
    n_nodes = max(node_list)
 
    slurm_settings = get_slurm_script(n_nodes,n_cores,'strong_scale',time)
    mpi_calls = ''
 
    for n_nodes in node_list:
       
        # run all variants on same set of nodes
        for variant in variants:
            
            experiment_dir = "logs/{0}/{1}/{2:02d}/".format(variant,mesh_pts,n_nodes)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # check if file with log call exists, if not, create a new one    
            for i in range(0,99):
                if not os.path.isfile(experiment_dir+"{0:02d}.call".format(i)):
                    with open(experiment_dir+"{0:02d}.call".format(i), "w") as text_file:
                        mpi_call = get_mpi_call(n_nodes,n_cores,variant,max_iter,mesh_pts,str(i).zfill(2))
                        text_file.write(mpi_call)
                    break

            mpi_calls += mpi_call + '\n echo "'+'='*80+'" \n'

    # generate slurm call
    slurm_script = slurm_settings + mpi_calls
            
    # run slurm call
    p = subprocess.Popen(['sbatch','--account=stf','--partition=stf'], stdin=subprocess.PIPE)
    p.communicate(slurm_script.encode('utf-8'))


##
## NOW RUN
##

time = '2:00:00'
n_cores = 28
max_iter = 4000
mesh_pts = 2000

variants = ['cg','pipecg','pipeprcg','pipeprcg_0']
node_list = list(range(1,16))

for k in range(15):
    strong_scale(node_list,n_cores,variants,max_iter,mesh_pts,time)



