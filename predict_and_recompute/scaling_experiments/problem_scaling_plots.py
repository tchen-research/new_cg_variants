import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


n_nodes = 30
n_cores = 16
max_iter = 500

num_repeats = 5
skip = n_nodes // 10
size_list = list(range(100,1500,100))

time_components = ['MatMult','PCApply','VecAXPY','VecAYPX','VecTDot','VecNorm','VecReduceArith','VecReduceComm','VecReduceEnd','VecReduceBegin','self']
## what is self???

variants = ['hscg','pipecg','pipeprcg','pipeprcg_0']


# parse scaling test data
for variant in variants:
    data = {}
    for component in time_components:
        data[component] = np.zeros(len(size_list))
    times = np.inf*np.ones(len(size_list))
    mflops = np.inf*np.ones(len(size_list))

    for k,mesh_pts in enumerate(size_list):
        # instead should find trial with min time, and then generate percents from that..
        for repeat in range(num_repeats):
            try:
                tree = ET.parse(f'./logs/{variant}/{mesh_pts}_{n_nodes:02d}_trial{repeat:02d}.xml')
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

    np.save(f'data/{variant}_problem_scale',data,allow_pickle=True)

data = {}
for variant in variants:
    data[variant] = np.load(f'data/{variant}_problem_scale.npy',allow_pickle=True).item()

# scaling times
t0 = data[variants[0]]['times'][0]
fig, ax1 = plt.subplots(1,1,figsize=(5,3))
for variant in variants:
    times = data[variant]['times']
    ax1.plot(size_list,times,linestyle=':',marker='o',label=variant)

#ax1.set_yscale('log')
ax1.set_ylim(bottom=0)
ax1.legend()
plt.savefig('figures/scale_problem_scale.pdf')

# mflops scaling
plt.figure()
for variant in variants:
    mflops = data[variant]['mflops']
    plt.plot(size_list,mflops,linestyle=':',marker='o',label=variant)

plt.legend()
plt.savefig('figures/mflops_problem_scale.pdf')


# scaling percents

fig, axs = plt.subplots(1,len(variants),figsize=(8,3))
for k,variant in enumerate(variants):
    bottom_sum = np.zeros(len(size_list))
    for component in time_components:
        axs[k].bar(np.arange(len(size_list)),data[variant][component],bottom = bottom_sum)
        bottom_sum += data[variant][component]
    axs[k].set_xticks(np.arange(len(size_list)),size_list)
    axs[k].set_title(variant)

axs[-1].legend(time_components,loc='upper left',bbox_to_anchor=(1,1))
plt.savefig('figures/percents_problem_scale.pdf',bbox_inches='tight')

