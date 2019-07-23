import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# set scaling experiment parameters
n_cores = 14

#mesh_pts = 2**20
mesh_pts = 650000

variants = ['cg','prcg','pipecg','pipeprcg_0','pipeprcg']
node_list = list(range(1,16))
node_list = [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32]

#node_list = [1,2,4,8,16,32]

node_list.sort()
print(node_list)

os.system(f'mkdir -p ./data')
os.system(f'mkdir -p ./figures')

# maximum number of trials
max_trials = 9

# parse scaling test data
for variant in variants:
    data = {}
    times = np.empty((len(node_list),max_trials))
    times[:] = np.nan

    # loop over number of nodes
    for k,n in enumerate(node_list):
        # now loop over trails
        for trial_number in range(max_trials):
            # if the trial exists, parse the runtime (note there is a slight overhead for setting up the KSP, but we run enough iterations that this is relatively small)
            try:
                tree = ET.parse(f'./logs/{variant}/{mesh_pts}/{n:02d}/{trial_number:02d}.xml')
                root = tree.getroot()
                globalperformance = root[0].find('globalperformance')        
                timertree = root[0].find('timertree')
                
                total_time = float(timertree.find('totaltime').text)

                # look for KSPSolve event, which is the solve stage, to figure out percent of time spent on solve.
                for child in timertree:
                    try:
                        if child.find('name').text == 'KSPSolve':
                            percent = float(child.find('time').find('value').text)/100
                            break
                    except:
                        pass
                
                times[k,trial_number] = total_time #* percent
  
            except:
                pass
    
    data['times'] = times
    np.save(f'data/{variant}_strong_scale',data,allow_pickle=True)

# now load data
data = {}
for variant in variants:
    data[variant] = np.load(f'data/{variant}_strong_scale.npy',allow_pickle=True).item()

varaint_styles = {
    'exact_pcg': {'label':'exact','linestyle':':','marker':None,'color':'#93a1a1','offset':0},
    'cg': {'label':'HS-CG','linestyle':'-','marker':'o','color':'#93a1a1','offset':0},
    'pipecg': {'label':'GV-CG','linestyle':'-','marker':'s','color':'#93a1a1','offset':3/4},
    'prcg': {'label':'PR-CG','linestyle':':','marker':'.','color':'#073642','offset':0}, # note the name difference
    'pipeprcg_0': {'label':'\\textsc{pipe-P-CG}','linestyle':'--','marker':'.','color':'#073642','offset':0},
    'pipeprcg': {'label':'\\textsc{pipe-PR-CG}','linestyle':'-','marker':'.','color':'#073642','offset':0},
}

# plot scaling times
t0 = np.nanmin(data[variants[0]]['times'][0])
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,3))

for variant in variants:
    
    # get minimum runtime for each variant and node
    times = np.nanmin(data[variant]['times'],axis=1)
#    times = data[variant]['times']

    # get style parameters
    styles = varaint_styles[variant]

    lbl = styles['label']
    ms = styles['marker']
    cl = styles['color']
    ls = styles['linestyle']
   
    ax1.plot(node_list,times,linestyle=ls,marker=ms,label=lbl,color=cl,markevery=1)
    ax2.plot(node_list,times[0]/times,linestyle=ls,marker=ms,color=cl,markevery=1)

for variant in variants:
    print(f"{variant}: {np.nanmin(data[variants[0]]['times'])/np.nanmin(data[variant]['times'])}")

# add legend
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1])

# logs scale axes and grids on
for ax in [ax1,ax2]:
    ax.set_xscale('log',basex=2)
    ax.set_yscale('log',basey=2)
    ax.grid(True,linestyle=':')
    ax.set_xlabel(f'number of nodes ($\\times {n_cores}$ MPI processes/node)')


# set y labels
ax1.set_ylabel('runtime (seconds)')

ax2.set_ylabel('$\\times$ speedup over single node')
ax2.set_ylim(bottom=.75,top=22.0)

plt.subplots_adjust(wspace=.25, hspace=0)

# save
plt.savefig('figures/strong_scale.pdf',bbox_inches='tight')
plt.savefig('figures/strong_scale.svg',bbox_inches='tight')
