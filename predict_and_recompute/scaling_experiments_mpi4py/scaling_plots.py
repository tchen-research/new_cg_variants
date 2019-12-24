#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

varaint_styles = {
    'hs_cg': {'label':'HS-CG','linestyle':'-','marker':'o','color':'#93a1a1','offset':0},
    'cg_cg': {'label':'CG-CG','linestyle':'-','marker':'^','color':'#93a1a1','offset':3/4},
    'gv_cg': {'label':'GV-CG','linestyle':'-','marker':'s','color':'#93a1a1','offset':3/4},
    'pr_cg': {'label':'PR-CG','linestyle':':','marker':'.','color':'#073642','offset':0},
    'pipe_pr_cg': {'label':'\\textsc{pipe-PR-CG}','linestyle':'-','marker':'.','color':'#073642','offset':0},
    }


# generate solutions

n_cores = 16
node_list = [1,2,4,8,16,24,32,48]
max_trials = 8
n = 1536*8

timing_keys = ['tot','c_mv','w_mv','c_ip','w_ip','w_vec']

variant_names = ['hs_cg','cg_cg','gv_cg','pr_cg','pipe_pr_cg']

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,3))

for variant_name in variant_names:
    times = {}
    for key in timing_keys:
        times = np.empty((len(node_list),max_trials))
        times[:] = np.nan
        errors = np.empty((len(node_list),max_trials))
        errors[:] = np.nan

    for k,n_nodes in enumerate(node_list):
        for trial_number in range(max_trials):
            try:
                trial_name = "{}x{}:{}".format(n_nodes,n_cores,trial_number)
#                print("./data/{}/{}_{}.npy".format(n,variant_name,trial_name))
                data = np.load("./data/{}/{}_{}.npy".format(n,variant_name,trial_name),allow_pickle=True).item()
            
                times[k,trial_number] = data['timings']['tot']
                errors[k,trial_number] = data['error']

            except:
                pass
    
    min_times = np.nanmin(times,axis=1)
    ave_errors = np.nanmean(errors,axis=1)
    
    print("{}: {}".format(variant_name,min_times))
 
    styles = varaint_styles[variant_name]

    lbl = styles['label']
    ms = styles['marker']
    cl = styles['color']
    ls = styles['linestyle']

    ax1.plot(node_list,min_times,linestyle=ls,marker=ms,label=lbl,color=cl,markevery=1)
    ax2.plot(node_list,ave_errors,linestyle=ls,marker=ms,label=lbl,color=cl,markevery=1)
    #ax2.plot(node_list,min_times[0]/min_times,linestyle=ls,marker=ms,label=lbl,color=cl,markevery=1)

# add legend
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1])

# logs scale axes and grids on
ax1.set_yscale('log',basey=2)
ax2.set_yscale('log',basey=10)
for ax in [ax1,ax2]:
    ax.set_xscale('log',basex=2)
    ax.grid(True,linestyle=':')
    ax.set_xlabel(f'number of nodes ($\\times {n_cores}$ MPI processes/node)')


# set y labels
ax1.set_ylabel('runtime (seconds)')

#ax2.set_ylabel('$\\times$ speedup over single node')
ax2.set_ylabel('$\mathbf{A}$-norm of error: $\| \mathbf{x}-\mathbf{x}_k \|_\mathbf{A}$')
ax2.set_ylim(bottom=10**(-7.5),top=10**(-3.5))

plt.subplots_adjust(wspace=.25, hspace=0)

plt.savefig('figures/strong_scale.pdf',bbox_inches='tight')
plt.savefig('figures/strong_scale.svg',bbox_inches='tight')
