"""
Plot horizontally averaged profiles from snapshots.

Usage:
    plot_averages.py <file> [options]

Options:
    --output=<output>    Output directory; if blank a guess based on likely case name will be made
"""
import logging
logger = logging.getLogger(__name__.split('.')[-1])

for system in ['matplotlib', 'h5py']:
    dlog = logging.getLogger(system)
    dlog.setLevel(logging.WARNING)

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py
import scipy.integrate as sci

from docopt import docopt
args = docopt(__doc__)

import dedalus.public as de
from dedalus.tools import post
from dedalus.tools.general import natural_sort
file = args['<file>']
case = args['<file>'].split('averages')[0]
logger.debug("opening file: {}".format(file))

if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = case +'/'
    output_path = pathlib.Path(data_dir).absolute()

data = {}
z = None
with h5py.File(file, 'r') as f:
    data_slices = (slice(None), 0, 0, slice(None))
    for task in f['tasks']:
        logger.info("task: {}".format(task))
        data[task] = np.array(f['tasks'][task][data_slices])
        if z is None:
            z = f['tasks'][task].dims[3][0][:]
    times = f['scales/sim_time'][:]
    f.close()

def time_avg(f, axis=0):
    n_avg = f.shape[axis]
    return np.squeeze(np.sum(f, axis=axis))/n_avg

t_max = np.max(times)
t_min = 0.75*t_max
i_avg = np.argmin(np.abs(times-t_min))

q_avg = time_avg(data['b(z)'][i_avg:,...])
fig, ax = plt.subplots(figsize=(6,6/1.5))
fig.subplots_adjust(top=0.9, right=0.95, bottom=0.2, left=0.15)
# for qi in data[task]:
#     ax.plot(z, qi, alpha=0.3)
ax.plot(z, 1-z, color='xkcd:dark grey', alpha=0.3, linestyle='dashed', label=r'$b(t=0)$')
ax.plot(z, q_avg, linewidth=2, color='black', label=r'$\langle b(t) \rangle$')
ax.legend()
ax.set_ylabel('b')
ax.set_xlabel('z')
fig.savefig(f'{str(output_path):s}/b_profile.png', dpi=300)


F_h = time_avg(data['F_h(z)'][i_avg:,...])
F_κ = time_avg(data['F_κ(z)'][i_avg:,...])
fig, ax = plt.subplots(figsize=(6,6/1.5))
fig.subplots_adjust(top=0.9, right=0.8, bottom=0.2, left=0.2)
ax.plot(z, F_h, label='$F_h$')
ax.plot(z, F_κ, label='$F_\kappa$')
ax.plot(z, F_h+F_κ, color='xkcd:dark grey', alpha=0.5, label='$F_\mathrm{tot}$')
ax.legend()
ax.set_ylabel('fluxes')
ax.set_xlabel('z')
fig.savefig(f'{str(output_path):s}/flux_profile.png', dpi=300)
