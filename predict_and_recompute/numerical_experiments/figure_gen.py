import numpy as np
import scipy as sp
from scipy import sparse
from scipy.io import loadmat
import matplotlib.pyplot as plt
  
import sys
#sys.path.append('../')

import os

from cg_variants import *
from callbacks import error_A_norm, residual_2_norm, error_2_norm, updated_residual_2_norm, print_k

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

EPSILON_MACHINE = 1e-16

#%%
def test_matrix(A,max_iter,title,preconditioner=None,variants=[]):
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
   
    # define callbacks to use
    callbacks = [error_A_norm,residual_2_norm,error_2_norm,updated_residual_2_norm,print_k(10)]
    
    # define preconditioners
    prec = lambda x:x
    prec_long = lambda x:x
    if preconditioner=='jacobi':
        prec = lambda x: (1/A.diagonal())*x
        prec_long = lambda x: (1/A.diagonal().astype(np.longdouble))*x
    
    # set up directory
    os.system(f'mkdir -p ./data/{title}_{preconditioner}')
    
    # run rest of methods
    for method in variants:

        # check if exact cg to use higher precision
        if method == exact_pcg:
            trial = exact_pcg(A.astype(np.longdouble),b.astype(np.longdouble),x0.astype(np.longdouble),min(max_iter,N),callbacks=callbacks,x_true=x_true.astype(np.longdouble),preconditioner=prec_long)
            np.save(f'./data/{title}_{preconditioner}/exact_pcg',trial,allow_pickle=True)
            continue
   
        # otherwise use normal precision
        trial = method(A,b,x0,max_iter,callbacks=callbacks,x_true=x_true,preconditioner=prec)
        np.save(f'./data/{title}_{preconditioner}/{method.__name__}',trial,allow_pickle=True)
    
#%%
def parse_convergence_data(matrix_name,preconditioner=None,variants=[]):
    """
    gather convergence data from trials and save as a row of a latex table
    """
    
    # get matrix information
    # really this should probably be saved as a different file so we don't have to load the whole matrix

    A = sp.sparse.csr_matrix(sp.io.mmread(f"../matrices/{matrix_name}.mtx"))

    n,_ = A.shape
    nnz = A.nnz    
        
    # get convergence information
    min_iters = []
    min_errors = []
    
    error_tol = 1e-5
    
    for method in variants:
        
        trial = np.load(f'./data/{matrix_name}_{preconditioner}/{method}.npy',allow_pickle=True).item()
        
        rel_error_A_norm = trial['error_A_norm']/trial['error_A_norm'][0]
        
        min_iters.append(np.argmin(rel_error_A_norm>error_tol))
        min_errors.append(np.log10(np.nanmin(rel_error_A_norm)))
        
    
    # generate data string for tex table
    formatted_matrix_name = r'\texttt{'+matrix_name.replace('_','\_')+r'}'
    
    formatted_preconditioner = '-'
    if preconditioner == 'jacobi':
        formatted_preconditioner = 'Jac.'
    
    data = f"{formatted_matrix_name} & {formatted_preconditioner} & {n} & {nnz}"
    
    data_iter = ''
    data_err = ''
    
    for k in range(len(min_errors)):
        formatted_min_iter = min_iters[k] if min_iters[k] != 0 else '-'
        
        mi_bold_start = '\\tableemph' if ((min_iters[k] > 1.1*min_iters[0]) or (min_iters[k]==0)) else ''
        me_bold_start = '\\tableemph' if (min_errors[k] > .9*min_errors[0]) else ''
                
        data_iter += f'& {mi_bold_start}{{{formatted_min_iter}}}'
        data_err += f'&{me_bold_start}{{{min_errors[k]:1.2f}}}'
    
    data += data_iter + data_err + '\\\\ \n'
    with open(f'./data/{matrix_name}_{preconditioner}/convergence.txt', 'w') as text_file:
        text_file.write(data)
        
#%%
def gen_convergence_table():
    """
    merge all data from gen_convergence_data into single latex table
    """
    
    os.system('cat ./data/*None/*convergence.txt > ./figures/convergence_table_data.tex')
    os.system('cat ./data/*jacobi/*convergence.txt >> ./figures/convergence_table_data.tex')

#%%
varaint_styles = {
    'exact_pcg': {'label':'exact','linestyle':':','marker':None,'color':'#93a1a1','offset':0},
    'hs_pcg': {'label':'HS-CG','linestyle':'-','marker':'o','color':'#93a1a1','offset':0},
    'cg_pcg': {'label':'CG-CG','linestyle':'-','marker':'^','color':'#93a1a1','offset':1/4},
    'm_pcg': {'label':'M-CG','linestyle':'-','marker':'v','color':'#93a1a1','offset':2/4},
    'gv_pcg': {'label':'GV-CG','linestyle':'-','marker':'s','color':'#93a1a1','offset':3/4},
 
    'pipe_p_m_pcg': {'label':'\\textsc{pipe-P-M-CG}','linestyle':'-','marker':None,'color':'#6c71c4','offset':0},
    'pipe_pr_m_pcg': {'label':'\\textsc{pipe-PR-M-CG}','linestyle':':','marker':None,'color':'#859900','offset':0},
    
    'pr_pcg': {'label':'PR-CG','linestyle':':','marker':None,'color':'#073642','offset':0},
    'pipe_p_pcg': {'label':'\\textsc{pipe-P-CG}','linestyle':'-','marker':None,'color':'#2aa198','offset':0},
    'pipe_pr_pcg': {'label':'\\textsc{pipe-PR-CG}','linestyle':'-','marker':None,'color':'#073642','offset':0},
   
}

def add_plot(trial,quantity,ax,num_markers,title='',pc=''):
    """
    helper function to add plots
    """
    styles = varaint_styles[trial['name']]

    lbl = styles['label']
    ms = styles['marker']
    cl = styles['color']
    ls = styles['linestyle']
    vo = styles['offset']/num_markers

    # subsample for plots
    skip = max(1,trial['max_iter'] // 1000) # downsample if there are a lot of iterations
    num_pts = len(np.arange(trial['max_iter'])[::skip])
        
    ax.plot(np.arange(trial['max_iter'])[::skip],trial[quantity][::skip]/trial[quantity][0],label=lbl,linestyle=ls,color=cl,marker=ms,markevery=(int(vo*num_pts),num_pts//num_markers))
    if title != '':
        ax.set_title(f"\\texttt{{{title}}}{', prec.='+pc.capitalize() if pc else ''}")


def plot_matrix_test(title,preconditioner=None,quantity='error_A_norm',variants=[],ylabel=True):
    """
    plot convergence data on single plot
    """
    
    num_markers = 5
    
    f, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6,4))
    
    # add plot for each method
    for method in variants: 
        trial = np.load(f'./data/{title}_{preconditioner}/{method}.npy',allow_pickle=True).item()
        add_plot(trial,quantity,ax,num_markers)
       
    # adjust axes and labels
    ax.set_yscale('log')
    ax.set_ylim(1e-16,5)
    
    if ylabel:
        ax.set_ylabel('$\mathbf{A}$-norm of error: $\| \mathbf{x}-\mathbf{x}_k \|_\mathbf{A}$')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1,1))
    else:
        ax.yaxis.set_ticklabels([])    
    ax.set_xlabel('iteration $k$')
    ax.grid(True,linestyle=':')

    os.system(f'mkdir -p ./figures')
    plt.savefig(f'figures/{title}_{preconditioner}_{quantity}{"" if ylabel else "_nolbl"}.pdf',bbox_inches='tight')
    plt.savefig(f'figures/{title}_{preconditioner}_{quantity}{"" if ylabel else "_nolbl"}.svg',bbox_inches='tight')
    plt.close()


def plot_matrices_test(titles,preconditioners,quantity='error_A_norm',variants=[]):
    """
    plot convergence data
    """
    
    num_markers = 5
    
    # load data
    f, axs = plt.subplots(2, 2, sharex=False, sharey=True, figsize=(11,7.5))
    print(axs.flatten())
    for k,ax in enumerate(axs.flatten()):
        if k >= len(titles):
            continue 

        for method in variants:
            trial = np.load(f'./data/{titles[k]}_{preconditioners[k]}/{method}.npy',allow_pickle=True).item()
            add_plot(trial,quantity,ax,num_markers,titles[k].replace('_','\_'),preconditioners[k])

    axs[0,0].set_yscale('log')
    axs[0,0].set_ylim(1e-16,5)
    axs[0,0].set_ylabel('$\mathbf{A}$-norm of error: $\| \mathbf{x}-\mathbf{x}_k \|_\mathbf{A}$')
    axs[1,0].set_ylabel('$\mathbf{A}$-norm of error: $\| \mathbf{x}-\mathbf{x}_k \|_\mathbf{A}$')
    

    handles, labels = axs[0,0].get_legend_handles_labels()
    axs[0,0].legend(handles[::-1], labels[::-1], loc='lower left')

    for k,ax in enumerate(axs.flatten()):
        
        ax.set_xlabel('iteration $k$')
        ax.grid(True,linestyle=':')

#    plt.suptitle(f"{title}, $A$-norm of error: $\| x-x_k \|_A$")
#    ax1.set_title('new variants')
#    ax2.set_title('new pipelined variants')
    
    os.system(f'mkdir -p ./figures')
    plt.subplots_adjust(wspace=.05, hspace=.35)
    plt.savefig(f"figures/{'-'.join(titles)}_{'-'.join([str(pc) for pc in preconditioners])}_{quantity}.pdf",bbox_inches='tight')
    plt.savefig(f"figures/{'-'.join(titles)}_{'-'.join([str(pc) for pc in preconditioners])}_{quantity}.svg",bbox_inches='tight')
    plt.close()



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
    ['bcsstk03',1250,None],
    ['bcsstk14',25000,None],
    ['bcsstk15',35000,None],
    ['bcsstk16',900,None],
    ['bcsstk17',45000,None],
    ['bcsstk18',1750000,None],
    ['bcsstk27',2300,None],
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
    ['nos7',7000,None],
]

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
    ['s3dkq4m2',60000,'jacobi'], # too big for github, download from matrix market
    ['s3dkt3m2',75000,'jacobi'], 
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

#%%
# NOW RUN TESTS AND GENERATE FIGURES
for matrix_name,max_iter,preconditioner in matrices:
    print(f'matrix: {matrix_name}, preconditioner: {preconditioner}')
    
    methods = [hs_pcg,cg_pcg,m_pcg,gv_pcg]
    methods += [pipe_p_m_pcg,pipe_pr_m_pcg]
    methods += [pr_pcg, pipe_p_pcg, pipe_pr_pcg]

    A = sp.sparse.csr_matrix(sp.io.mmread(f"../matrices/{matrix_name}.mtx"))
    test_matrix(A,max_iter,matrix_name,preconditioner,variants=methods)

    # make plots
    methods_str = [method.__name__ for method in methods]
    plot_matrix_test(matrix_name,preconditioner,'error_A_norm',variants=methods_str)
    plot_matrix_test(matrix_name,preconditioner,'error_2_norm',variants=methods_str)
    plot_matrix_test(matrix_name,preconditioner,'residual_2_norm',variants=methods_str)

    # generate convergence table with selected variants
    paper_methods = ['hs_pcg','cg_pcg','m_pcg','pr_pcg','gv_pcg','pipe_pr_m_pcg','pipe_pr_pcg']
    parse_convergence_data(matrix_name,preconditioner,variants=paper_methods)
    
gen_convergence_table()

#%%
# GENERATE GROUPED PLOTS
variants = ['hs_pcg', 'cg_pcg', 'm_pcg', 'gv_pcg','pr_pcg','pipe_pr_pcg']
for quantity in ['error_A_norm','error_2_norm','residual_2_norm']:
    plot_matrices_test(['bcsstk15','s3rmq4m1','bcsstk03','model_48_8_3'],['jacobi','jacobi',None,None],quantity=quantity,variants=variants)
