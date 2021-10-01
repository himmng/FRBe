import numpy as np
from chainconsumer import ChainConsumer
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py as hp

#### plotting rc params ######
mpl.rcParams['text.usetex'] = True
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.3
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.3
mpl.rcParams['patch.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams[
    'text.latex.preamble'] = r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'

## all model cases ##
mo = ['No-Sc', 'Sc-I', 'Sc-II']
runs = ['chain', 'walk']
mm = ['n', 'o', 't']
rmodel = ['sfr', 'cer']
dmmodel = ['dm50', 'dmrand']
#######################################

burin_in = 1000 # initial burn-in steps

nwalkers = 120 # number of random walkers

### Loading mcmc file ##
file1 = 'your .h5 filename' #change the filename for you

data1 = hp.File(file1, 'r')
d1 = data1.get('mcmc')
chain1 = np.array(d1.get('chain'))[burin_in:, :]
log1 = np.array(d1.get('log_prob'))[burin_in:, :]

#### rashaping all walkers into one chain
log1 = log1.reshape(len(log1) * nwalkers, )
chain1 = chain1.reshape(len(chain1) * nwalkers, 3)

print(np.min(log1), np.max(log1))

xx = np.where(log1 > (np.max(log1)-10)) #rectifying the highly likelihood region
log1 = log1[xx]
chain1 = chain1[xx]

### plotting the results
c = ChainConsumer()
c.add_chain(chain1, parameters=[r'$\alpha$', r'$\overline{E}_{33}$', r'$\gamma$'], name=r'$\rm DM_{50}$', color='blue')

c.configure(label_font_size=27, linestyles=['--', '-', '-.', '--'], sigma2d=False, bar_shade=True,
            linewidths=[3, 3], tick_font_size=23, shade=[True, True, True, True], sigmas=[1, 2, ],
            shade_alpha=[0.8, 0.2],
            legend_kwargs={'fontsize': 27, 'frameon': True, 'fancybox': False, 'framealpha': 1, 'edgecolor': 'black', })

fig = c.plotter.plot(figsize=1.6)
# fig.text(x=0.63,y=0.905,s=r'\rm %s'%mo[i], size=30,) #bbox=dict(boxstyle="square",facecolor='none', edgecolor='k'))
# fig.text(x=0.63,y=0.855,s=r'$\rm \uppercase{%s}$'%rmodel[j], size=30,)
fig.set_size_inches(2 + fig.get_size_inches())
c.plotter.plot_walks(convolve=100)
plt.savefig('walk.png', bbox_inches='tight')
# table = c.analysis.get_latex_table()
# print(table)