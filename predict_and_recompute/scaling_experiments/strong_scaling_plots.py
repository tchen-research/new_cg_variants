import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# set scaling experiment parameters
n_cores = 14

#mesh_pts = 2**20
mesh_pts = 650000

variants = ['cg','chcg','pipecg','pipeprcg_0','pipeprcg']
node_list = list(range(1,16))
node_list = [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32]

#node_list = [1,2,4,8,16,32]

node_list.sort()
print(node_list)


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

# define some helper functions to get formatting information for plotting
def variant_name(name):
    
    formatted_name = name.upper() 
    if name == 'hscg':
        formatted_name = 'HS-CG'
    elif name == 'chcg':
        formatted_name = 'PR-CG'
    elif name == 'pipecg':
        formatted_name = 'GV-CG'
    elif name == 'pipeprcg':
        formatted_name = 'PPR-CG$^*$'
    elif name == 'pipeprcg_0':
        formatted_name = 'PP-CG'

    return f"\\textsc{{{formatted_name}}}"

def variant_marker(name):
    
    if name == 'cg':
        return 'o'
    elif name == 'pipecg':
        return 's'
    
    return '.'

def variant_line_style(name):
    
    if name == 'chcg':
        return ':'
    elif name == 'pipeprcg_0':
        return '--'
    elif name == 'pipeprcg':
        return '-'

    return '-'

def variant_color(name):
    
    # if one of default variants
    for string in ['cg','hscg','pipecg']:
        if name == string:
            return '#93a1a1'
    return '#073642'

def grey_out(color):
    new_color = '#'
    
    r,g,b = (int(color[i:i+2],16) for i in [1,3,5])
    
    f = 0.5
    L = 0.3* + 0.6*g + 0.1*b

    for c in (r,g,b):
        new_color += hex(min(int( (c + f * (L - c)) ),255))[2:].zfill(2)

    return new_color

def lighten(color):
    new_color = '#'
    
    r,g,b = (int(color[i:i+2],16) for i in [1,3,5])
 
    scale = .6

    for c in (r,g,b):
        new_color += hex(min(int( c + (255-c)*scale ),255))[2:].zfill(2)

    return new_color

   


# plot scaling times
t0 = np.nanmin(data[variants[0]]['times'][0])
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,4))

for variant in variants:
    
    # get minimum runtime for each variant and node
    times = np.nanmin(data[variant]['times'],axis=1)
#    times = data[variant]['times']

    # get style parameters
    lbl = variant_name(variant)
    ms = variant_marker(variant)
    ls = variant_line_style(variant)
    cl = variant_color(variant)

   
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
