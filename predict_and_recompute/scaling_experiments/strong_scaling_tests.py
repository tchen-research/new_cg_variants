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

#mpirun -genvall -n 28 -ppn 28 ./ex2a -n 500000 -rho 0.99 -kappa 1e6 -k 500 -off_value 1e-4 -mat_type mpiaij -pc_type none -ksp_type cg -ksp_norm_type natural -ksp_max_it 400 -ksp_cfgonverged_reason -ksp_monitor -ksp_view -log_view :::ascii_xml 
#-n 100000 -rho 0.99 -kappa 1e6 -k 500 -off_value 1e-4 -mat_type mpiaij 

def get_mpi_call(n_nodes,n_cores,variant,max_iter,mesh_pts,trial_name):
    
    pc_type = "none"
    num_repeat = 1

    ## problem parameters
    k = 32

    # off value must be less than 1 / [nnz per row] to ensure matrix is positive definite
    off_value = 1e-4
    
    # pick rho near 1 and kappa larg enough to get slow convergence in finite precision
    rho = .95
    kappa = 1e6   

    opt_args = ''
    if variant == "pipeprcg_0":
        opt_args = "-recompute_q 0"
        ksp_type = "pipeprcg"
    else:
        ksp_type = variant

    
    problem_setup = "-n {0} -rho {3} -kappa {4} -k {1} -off_value {2} ".format(mesh_pts,k,off_value,rho,kappa) 
    
    solver_setup = "-pc_type {0} -ksp_type {1} {2} ".format(pc_type,ksp_type,opt_args)
    
    iters = "-num_repeat {0} -ksp_norm_type none -ksp_max_it {1} ".format(num_repeat,max_iter)
    
    log_view = "-ksp_converged_reason -ksp_view -log_view :./logs/{0}/{1}/{2:02d}/{3}.xml:ascii_xml".format(variant,mesh_pts,n_nodes,trial_name)

    return "\nmpirun -genvall -n {0} -ppn {1} ./ex2b ".format(n_nodes*n_cores,n_cores)+problem_setup+solver_setup+iters+log_view


##
## STRONG SCALING EXPERIMENT
##
def strong_scale(node_list,n_cores,variants,max_iter,mesh_pts,time,num_experiment_repeats):
    
    n_nodes = max(node_list)

    slurm_settings = get_slurm_script(n_nodes,n_cores,'strong_scale',time)
    mpi_calls = ''


    for _ in range(num_experiment_repeats):
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

                mpi_calls += '\necho '+mpi_call[1:]+'\n\n'+mpi_call+'\n echo "'+'='*80+'" \n'

    # generate slurm call
    slurm_script = slurm_settings + mpi_calls
            
    # run slurm call
    p = subprocess.Popen(['sbatch','--account=stf-ckpt','--partition=ckpt'], stdin=subprocess.PIPE)
    p.communicate(slurm_script.encode('utf-8'))


##
## NOW RUN
##

time = '8:00:00'
n_cores = 28
max_iter = 4000
mesh_pts = 2048

n_cores = 14
max_iter = 4000
mesh_pts = 650000

num_experiment_repeats = 10

variants = ['cg','chcg','pipecg','pipeprcg','pipeprcg_0']
node_list = [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32][::-1]

#node_list = [1,32,16,8,4]

strong_scale(node_list,n_cores,variants,max_iter,mesh_pts,time,num_experiment_repeats)



