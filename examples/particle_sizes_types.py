"""Example of routing particles with different kinds (types) and sizes.

Seeds three batches of particles (sand, mud, and plankton kinds) into the
same steady DeltaRCM flow field, routes them together, and shows the
per-particle kind presets playing out: different travel-time multipliers,
different diffusion, and deposition (sand and mud can settle permanently,
plankton never does). Produces a final-state plot colored per kind and a
short console summary of per-kind mean travel time.
"""

import numpy as np
import matplotlib.pyplot as plt

import dorado.particle_track as pt

# load some variables from a deltarcm output so stage is varied
import os.path
f_path = os.path.abspath(os.path.dirname(__file__))
data = np.load(os.path.join(f_path, 'ex_deltarcm_data.npz'))

# create params and then assign the parameters
params = pt.modelParams()

# pull depth and stage from that data
params.stage = data['stage']
params.depth = data['depth']
params.dx = 50.

# don't have any discharge data so we use a small uniform +y discharge in
# place of it: this keeps travel times physical (with zero discharge the
# velocities floor near zero and travel times explode), while the depth
# and stage from the deltarcm output still define the wet domain
params.qx = np.zeros(np.shape(params.depth))
params.qy = 0.5*np.ones(np.shape(params.depth))
params.theta = 1.0
params.model = 'DeltaRCM'  # say that our inputs are from DeltaRCM

# define initial particle locations and per-batch particle counts
seed_xloc = list(range(15, 17))
seed_yloc = list(range(137, 140))
Np_per_kind = 20

# fix the random seed for the example
np.random.seed(0)

# initialize the particles object and generate one batch per kind
particles = pt.Particles(params)
for kind in ['sand', 'mud', 'plankton']:
    particles.generate_particles(Np_per_kind, seed_xloc, seed_yloc,
                                 particle_type=kind)

# route all particles together for a modest number of iterations
num_iter = 30
walk_data = None
for _ in range(num_iter):
    walk_data = particles.run_iteration()

# a particle's kind is its constant per-particle attribute; the final
# positions and travel times come from the last entries of the histories
kinds = [walk_data['particle_type'][i][-1]
         for i in range(particles.Np_tracer)]

# final-state plot colored per kind, over the water depths
kind_colors = {'sand': 'peru', 'mud': 'dimgray', 'plankton': 'royalblue'}
plt.figure(figsize=(6, 8))
plt.imshow(params.depth, origin='lower', cmap='Blues', vmin=0, vmax=1)
for kind in kind_colors:
    kind_x = [walk_data['xinds'][i][-1]
              for i in range(particles.Np_tracer) if kinds[i] == kind]
    kind_y = [walk_data['yinds'][i][-1]
              for i in range(particles.Np_tracer) if kinds[i] == kind]
    plt.scatter(kind_x, kind_y, c=kind_colors[kind], label=kind, s=25)
plt.legend(title='particle kind')
plt.xlabel('x cell')
plt.ylabel('y cell')
plt.title('Final positions after %i iterations' % num_iter)
plt.savefig('particle_sizes_types_example.png', dpi=150,
            bbox_inches='tight')
print('Saved final-state plot to particle_sizes_types_example.png')

# short console summary of per-kind mean travel time and deposition
print('\nPer-kind summary after %i iterations:' % num_iter)
for kind in ['sand', 'mud', 'plankton']:
    idx = [i for i in range(particles.Np_tracer) if kinds[i] == kind]
    mean_time = np.mean([walk_data['travel_times'][i][-1] for i in idx])
    n_deposited = sum(walk_data['deposited'][i][-1] == 1 for i in idx)
    print('%s: mean travel time = %8.1f s, deposited %i/%i particles'
          % (kind, mean_time, n_deposited, len(idx)))
