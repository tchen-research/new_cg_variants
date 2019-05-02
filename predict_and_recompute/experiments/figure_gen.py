import numpy as np
import scipy as sp
from scipy import sparse
from scipy.io import loadmat
import matplotlib.pyplot as plt
  
import sys
sys.path.append('../')

import os

from cg_variants import *
from callbacks import error_A_norm, residual_2_norm

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

EPSILON_MACHINE = 1e-16

#%%
def test_matrix(A,max_iter,title,preconditioner=None):
    """
    generate data for convergence of variants on A
    
    Note
    ----
    should write it to adaptively add methods if they have already been complete
    """
    
    # set up problem
    N = A.get_shape()[0]
    x_true = np.ones(N) / np.sqrt(N)
    b = A@x_true
    x0 = np.zeros(N)
#    x_true = sp.sparse.linalg.spsolve(A,b)

    # define methods to use
    methods = [hs_pcg,cg_pcg,m_pcg,gv_pcg,ch_pcg,pipe_m_pcg,pipe_m_pcg_b,pipe_ch_pcg,pipe_ch_pcg_b]
    
    # define callbacks to use
    callbacks = [error_A_norm,residual_2_norm]
    
    prec = lambda x:x
    if preconditioner=='jacobi':
        prec = lambda x: (1/A.diagonal())*x
#        x_true = np.sqrt(A.diagonal())*x_true
#        D = sp.sparse.spdiags([1/np.sqrt(A.diagonal())],0,N,N)
#        A = D@A@D
#        b = D@b
        
    # run methods
    trials={}
    for method in methods:
        trials[method.__name__] = method(A,b,x0,max_iter,callbacks=callbacks,x_true=x_true,preconditioner=prec)

    np.save(f'./data/{title}_{preconditioner}',trials,allow_pickle=True)
    
#%%
def gen_convergence_data(matrix_name,preconditioner=None):
    """
    gather convergence data from trials and save as a row of a latex table
    """
    
    # get matrix information
    A = sp.sparse.csr_matrix(sp.io.mmread(f"../matrices/{matrix_name}.mtx"))

    n,_ = A.shape
    nnz = A.nnz    
        
    # get convergence information
    trials = np.load(f'./data/{matrix_name}_{preconditioner}.npy',allow_pickle=True).item()
    
    methods = ['hs_pcg','cg_pcg','m_pcg','ch_pcg','gv_pcg','pipe_m_pcg','pipe_ch_pcg','pipe_m_pcg_b','pipe_ch_pcg_b']
    
    min_iters = []
    min_errors = []
    
    error_tol = 1e-5
    
    for method in methods:
        
        trial = trials[method]
        
        rel_error_A_norm = trial['error_A_norm']/trial['error_A_norm'][0]

        min_iters.append(np.argmin(rel_error_A_norm>error_tol))
        min_errors.append(np.log10(np.nanmin(rel_error_A_norm)))
        
    
    # generate data string for tex table
    formatted_matrix_name = r'{\tt '+matrix_name.replace('_','\_')+r' }'
    
    formatted_preconitioner = '-'
    if preconditioner == 'jacobi':
        formatted_preconitioner = 'Jac.'
    
    data = f"{formatted_matrix_name} & {formatted_preconitioner} & {n} & {nnz}"
    
    data_iter = ''
    data_err = ''
    
    for k in range(len(min_errors)):
        formatted_min_iter = min_iters[k] if min_iters[k] != 0 else '-'
        
        mi_bold_start = '\\tableemph' if ((min_iters[k] > 1.1*min_iters[0]) or (min_iters[k]==0)) else ''
        me_bold_start = '\\tableemph' if (min_errors[k] > .9*min_errors[0]) else ''
                
        data_iter += f'& {mi_bold_start}{{{formatted_min_iter}}}'
        data_err += f'&{me_bold_start}{{{min_errors[k]:1.2f}}}'
    
    data += data_iter + data_err + '\\\\ \n'
    with open(f'./data/{matrix_name}_{preconditioner}_convergence.txt', 'w') as text_file:
        text_file.write(data)
        
#%%
def gen_convergence_table():
    """
    merge all data from gen_convergence_data into single latex table
    """
    
    os.chdir('./data/')
    print(os.getcwd() + "\n")
    os.system('cat *None_convergence.txt > ../figures/convergence_table_data.tex')
    os.system('cat *jacobi_convergence.txt >> ../figures/convergence_table_data.tex')
    os.chdir('..')

#%%
def variant_name(name):
    
    formatted_name = name.upper().replace('PCG','CG').replace('_','-')

    variant = ''
    if formatted_name.find('-B') != -1:
        formatted_name = formatted_name.replace('-B','')
        variant = '(b)'
    
    name = formatted_name.replace('PIPE','pipe')
    
    return f"\\textsc{{{formatted_name}}}{variant}"

def variant_marker(name):
    
    if name[:3] == 'hs_':
        return 'o'
    elif name[:3] == 'cg_':
        return '^'
    elif name[:2] == 'm_':
        return 'v'
    elif name[:3] == 'gv_':
        return 's'
    
    return None

def variant_offset(name,spacing):
    
    if name[:3] == 'hs_':
        return 0
    elif name[:3] == 'cg_':
        return spacing/3
    elif name[:2] == 'm_':
        return 2*spacing/3
    elif name[:3] == 'gv_':
        return spacing/2
    
    return 0

def variant_line_style(name):
    
    if name.find('_b') != -1:
        return '--'
    
    return '-'


def variant_color(name):
    
    # if one of default variants
    for string in ['hs_','cg_','m_c','m_p','gv_']:
        if name[:3] == string:
            return '#93a1a1'
    
    return '#073662'

#%%

def plot_matrix_test(title,preconditioner=None,quantity='error_A_norm'):
    """
    plot convergence data
    """
    
    num_markers = 5
    
    def add_plot(trial,ax):
        lbl = variant_name(trial['name'])
        ms = variant_marker(trial['name'])
        vo = variant_offset(trial['name'],1/num_markers)
        ls = variant_line_style(trial['name'])
        cl = variant_color(trial['name'])

        # subsample for plots
        skip = max(1,trial['max_iter'] // 1000) # downsample if there are a lot of iterations
        num_pts = len(np.arange(trial['max_iter'])[::skip])
        
        ax.plot(np.arange(trial['max_iter'])[::skip],trial[quantity][::skip]/trial[quantity][0],label=lbl,linestyle=ls,color=cl,marker=ms,markevery=(int(vo*num_pts),num_pts//num_markers))

    # load data
    trials = np.load(f'./data/{matrix_name}_{preconditioner}.npy',allow_pickle=True).item()
    
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(14,4))
    
    for method in ['hs_pcg', 'cg_pcg', 'm_pcg','ch_pcg']:
        add_plot(trials[method],ax1)

    for method in ['hs_pcg', 'gv_pcg', 'pipe_m_pcg', 'pipe_m_pcg_b']:
        add_plot(trials[method],ax2)
        
    for method in ['hs_pcg', 'gv_pcg', 'pipe_ch_pcg', 'pipe_ch_pcg_b']:
        add_plot(trials[method],ax3)
        
    ax1.set_yscale('log')
    ax1.set_ylim(1e-16,5)
    ax1.set_ylabel('$A$-norm of error: $\| x-x_k \|_A$')

    for ax in [ax1,ax2, ax3]:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='lower left')
        
        ax.set_xlabel('iteration $k$')
        ax.grid(True,linestyle=':')

#    plt.suptitle(f"{title}, $A$-norm of error: $\| x-x_k \|_A$")
#    ax1.set_title('new variants')
#    ax2.set_title('new pipelined variants')
    
    plt.subplots_adjust(wspace=.05, hspace=0)    
    plt.savefig(f'figures/{title}_{preconditioner}_{quantity}.pdf',bbox_inches='tight')
#    plt.show()
    
#%%
# PICK MATRICES TO TEST

matrices = []

matrices += [
    ['model_48_8_3',110,None],
    ['model_48_8_3',200,'jacobi'], 
]

matrices += [
    ['bcsstk03',250,'jacobi'],
    ['bcsstk14',800,'jacobi'],
    ['bcsstk15',830,'jacobi'],
    ['bcsstk16',320,'jacobi'],
    ['bcsstk17',3800,'jacobi'],
    ['bcsstk18',2700,'jacobi'],
    ['bcsstk27',380,'jacobi'],
]

matrices += [
    ['bcsstk03',1250,None], #
    ['bcsstk14',25000,None],
    ['bcsstk15',35000,None],
    ['bcsstk16',900,None], #
    ['bcsstk17',45000,None],
    ['bcsstk18',1500000,None], # tbd
    ['bcsstk27',2300,None], #
]

matrices += [
    ['nos1',900,'jacobi'],
    ['nos2',11000,'jacobi'],
    ['nos3',350,'jacobi'],
    ['nos4',120,'jacobi'],
    ['nos5',350,'jacobi'],
    ['nos6',130,'jacobi'],
    ['nos7',200,'jacobi'],
]

matrices += [
    ['nos1',4500,None],
    ['nos2',45000,None],
    ['nos3',400,None],
    ['nos4',150,None],
    ['nos5',600,None],
    ['nos6',2400,None],
    ['nos7',7000,None], # GVCG doesn't even work.. 
]

# these are weird..
matrices += [
    ['bcsstm19',1100,None],
    ['bcsstm20',700,None],
    ['bcsstm21',10,None],
    ['bcsstm22',85,None],
    ['bcsstm23',10000,None],
    ['bcsstm24',45000,None],
    ['bcsstm25',130000,None],
]

matrices += [
    ['494_bus',2500,None],
    ['662_bus',1200,None],
    ['685_bus',950,None],
    ['1138_bus',5000,None],
]

matrices += [
    ['494_bus',500,'jacobi'],
    ['662_bus',350,'jacobi'],
    ['685_bus',350,'jacobi'],
    ['1138_bus',1300,'jacobi'],
]

matrices += [
    ['s1rmq4m1',1000,'jacobi'],
    ['s1rmt3m1',1200,'jacobi'],
    ['s2rmq4m1',2100,'jacobi'],
    ['s2rmt3m1',3000,'jacobi'],
#    ['s3dkq4m2',75000,'jacobi'], # tbd n=90449
#    ['s3dkt3m2',75000,'jacobi'], # tbd n=90449
    ['s3rmq4m1',12000,'jacobi'],
    ['s3rmt3m1',17000,'jacobi'],
    ['s3rmt3m3',40000,'jacobi'],
]

matrices += [
    ['s1rmq4m1',12000,None],
    ['s1rmt3m1',12000,None],
    ['s2rmq4m1',35000,None],
    ['s2rmt3m1',48000,None],
#    ['s3dkq4m2',25000,None], #tbd
#    ['s3dkt3m2',25000,None], #tbd
    ['s3rmq4m1',100000,None],
    ['s3rmt3m1',150000,None],
    ['s3rmt3m3',250000,None],
]

matrices = [
    ['bcsstk18',1500000,None], # tbd
]

#%%
# NOW RUN TESTS

# add folders if they don't exist
os.system('mkdir -p ./data/')
os.system('mkdir -p ./figures/')

for matrix_name,max_iter,preconditioner in matrices:
    print(f'matrix: {matrix_name}')
    A = sp.sparse.csr_matrix(sp.io.mmread(f"../matrices/{matrix_name}.mtx"))
    test_matrix(A,max_iter,matrix_name,preconditioner)
    
for matrix_name,max_iter,preconditioner in matrices:
    gen_convergence_data(matrix_name,preconditioner)
    
gen_convergence_table()

for matrix_name,max_iter,preconditioner in matrices:
    print(f'matrix: {matrix_name}')
    plot_matrix_test(matrix_name,preconditioner,'error_A_norm')
    plot_matrix_test(matrix_name,preconditioner,'residual_2_norm')
