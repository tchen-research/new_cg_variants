import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


n_nodes = 40
n_cores = 16
mesh_pts = 1000

num_repeats = 5
skip = n_nodes // 10
core_list = [0]+list(range(skip-1,n_nodes,skip))

time_components = ['MatMult','PCApply','VecAXPY','VecAYPX','VecTDot','VecNorm','VecReduceArith','VecReduceComm','VecReduceEnd','VecReduceBegin','self']
## what is self???

variants = ['hscg','pipecg','pipeprcg','pipeprcg_0']


# parse scaling test data
for variant in variants:
    data = {}
    for component in time_components:
        data[component] = np.zeros(len(core_list))
    times = np.inf*np.ones(len(core_list))
    mflops = np.inf*np.ones(len(core_list))

    for k,n in enumerate(core_list):
        # instead should find trial with min time, and then generate percents from that..
        for repeat in range(num_repeats):
            try:
                tree = ET.parse(f'./logs/{variant}/{mesh_pts}_{(n+1):02d}_trial{repeat:02d}.xml')
                root = tree.getroot()
                globalperformance = root[0].find('globalperformance')        
                
                repeat_time = float(globalperformance.find('time').find('max').text)

                # if this trial has a better time
                if repeat_time < times[k] :
                    times[k] = repeat_time
    
                    # get runtimes times
                    mflops[k] = float(globalperformance.find('mflops').find('average').text)

                    # get event breakdowns
                    events = root[0].find('timertree').find('event').find('events')
                
                    for event in events:
                        event_name = event[0].text
                        if event_name in time_components:
                            data[event_name][k] = float(event[1][0].text)
 
            except:
                pass
        
    print(times)
    data['times'] = times
    data['mflops'] = mflops

    np.save(f'data/{variant}_strong_scale',data,allow_pickle=True)

data = {}
for variant in variants:
    data[variant] = np.load(f'data/{variant}_strong_scale.npy',allow_pickle=True).item()

# scaling times
t0 = data[variants[0]]['times'][0]
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,3))
for variant in variants:
    times = data[variant]['times']
    ax1.plot(core_list,times,linestyle=':',marker='o',label=variant)
    ax2.plot(core_list,t0/times,linestyle=':',marker='o',label=variant)

fig.suptitle(f'strong scaling experiment n={mesh_pts}')
ax1.set_yscale('log')
ax1.legend()
ax2.legend()
plt.savefig('figures/strong_scale.pdf')

# mflops scaling
plt.figure()
for variant in variants:
    mflops = data[variant]['mflops']
    plt.plot(core_list,mflops,linestyle=':',marker='o',label=variant)

plt.legend()
plt.title(f'strong scaling experiment n={mesh_pts}')
plt.savefig('figures/mflops_strong_scale.pdf')


# scaling percents

fig, axs = plt.subplots(1,len(variants),figsize=(8,3))
for k,variant in enumerate(variants):
    bottom_sum = np.zeros(len(core_list))
    for component in time_components:
        axs[k].bar(np.arange(len(core_list)),data[variant][component],bottom = bottom_sum)
        bottom_sum += data[variant][component]
    axs[k].set_xticks(np.arange(len(core_list)),core_list)
    axs[k].set_title(variant)

axs[-1].legend(time_components,loc='upper left',bbox_to_anchor=(1,1))
fig.suptitle(f'strong scaling experiment n={mesh_pts}')
plt.savefig('figures/percents_strong_scale.pdf',bbox_inches='tight')

