"""Compare particle travel times for three seeding strategies.

The same pyDeltaRCM-derived flow field is routed three times with 1000
particles each, differing only in how the particles are seeded:

1. ``method='exact'`` -- all particles clustered at the inlet cells,
2. ``method='random'`` -- particles scattered uniformly over the domain,
3. ``flux_proportional_seeding`` -- particles released in batches across
   the routing window, with batch sizes proportional to discharge.

Travel-time statistics for the three strategies are printed as a table
and drawn as a histogram. Determinism is guaranteed by fixing the global
numpy random seed at the top of the script (dorado does not expose
per-object seeding); a digest of the statistic arrays is printed so that
identical runs can be verified against each other.

This example uses serial routing only, via a bare ``run_iteration`` loop.
"""
import hashlib
import os.path

import matplotlib.pyplot as plt
import numpy as np

import dorado.particle_track as pt

# fix the random seed so that runs of this script are reproducible
np.random.seed(0)

# load the pyDeltaRCM-derived flow field (see the docs page for the
# provenance of this data and the NetCDF -> npz conversion snippet)
f_path = os.path.abspath(os.path.dirname(__file__))
data = np.load(os.path.join(f_path, 'ex_seeding_strategies_data.npz'))
depth = data['depth']
stage = data['stage']
qx = data['qx']
qy = data['qy']

# create params and then assign the parameters
params = pt.modelParams()
params.depth = depth
params.stage = stage
params.qx = qx
params.qy = qy
params.dx = 50.  # m, from the pyDeltaRCM run
params.theta = 1.0
params.model = 'DeltaRCM'  # say that our inputs are from DeltaRCM

Np_tracer = 1000
num_iter = 50

# the pyDeltaRCM run used N0_meters=250 (5 cells at dx=50), which gives
# the inlet at columns 98-102 of row 0; seed just downstream, in row 2
inlet_xloc = 2
inlet_yloc = [98, 99, 100, 101, 102]
seed_xloc = [inlet_xloc] * len(inlet_yloc)
seed_yloc = inlet_yloc


def route_n_steps(particles, n):
    """Route the particles of a Particles object for n iterations."""
    walk_data = None
    for _ in range(n):
        walk_data = particles.run_iteration()
    return walk_data


# --- strategy 1: exact seeding, clustered at the inlet cells ---
inlet_particles = pt.Particles(params)
inlet_particles.generate_particles(Np_tracer, seed_xloc, seed_yloc,
                                   method='exact')
walk_inlet = route_n_steps(inlet_particles, num_iter)

# --- strategy 2: random seeding over the whole domain ---
# note: a few percent of the cells in this dataset are dry (no water);
# particles seeded on those cells stay in place for the whole window
L, W = np.shape(depth)
random_particles = pt.Particles(params)
random_particles.generate_particles(Np_tracer, list(range(L)),
                                    list(range(W)), method='random')
walk_random = route_n_steps(random_particles, num_iter)

# --- strategy 3: flux-proportional seeding across the routing window ---
# batch sizes proportional to the discharge magnitude, one batch released
# per routing iteration. The discharge of this steady field is constant,
# so this reduces to a uniform release schedule -- the point is that
# late releases have had less time to travel when the window closes.
speed = np.sqrt(qx ** 2 + qy ** 2)
Q = np.array([np.mean(speed[depth > 0])] * num_iter)
counts = pt.flux_proportional_seeding(Q, Np_tracer, num_iter)
flux_particles = pt.Particles(params)
walk_flux = None
for step in range(num_iter):
    if counts[step] > 0:
        flux_particles.generate_particles(int(counts[step]), seed_xloc,
                                          seed_yloc, method='exact')
    walk_flux = flux_particles.run_iteration()

# --- travel-time statistics ---
# accumulated travel time after the routing window; walk_data
# ['travel_times'] is ordered [particle][iteration], so the last entry
# per particle is its total time in motion
final_times = {
    'exact at inlet': np.array([t[-1] for t in
                                walk_inlet['travel_times']]),
    'random in domain': np.array([t[-1] for t in
                                  walk_random['travel_times']]),
    'flux-proportional': np.array([t[-1] for t in
                                   walk_flux['travel_times']]),
}

# time each particle spent in the ocean basin (the lower half of the
# domain), measured with the exposure_time function
basin_roi = np.zeros_like(depth, dtype='int')
basin_roi[60:, :] = 1
basin_times = {
    'exact at inlet': np.array(pt.exposure_time(walk_inlet, basin_roi)),
    'random in domain': np.array(pt.exposure_time(walk_random, basin_roi)),
    'flux-proportional': np.array(pt.exposure_time(walk_flux, basin_roi)),
}

# sanity checks: every particle is present in every statistic
for name in final_times:
    assert len(final_times[name]) == Np_tracer, name
    assert len(basin_times[name]) == Np_tracer, name
    assert np.all(np.isfinite(final_times[name])), name
    assert np.all(final_times[name] >= 0), name

# --- report ---
print('\nTravel-time statistics by seeding strategy '
      '(N = %d particles, %d iterations)' % (Np_tracer, num_iter))
print('%-18s | %10s %10s %10s | %14s' %
      ('strategy', 'p10 [s]', 'p50 [s]', 'p90 [s]', 'p50 basin [s]'))
print('-' * 72)
for name in final_times:
    p10, p50, p90 = np.percentile(final_times[name], [10, 50, 90])
    b50 = np.percentile(basin_times[name], 50)
    print('%-18s | %10.1f %10.1f %10.1f | %14.1f' %
          (name, p10, p50, p90, b50))

# digest of the statistic arrays, so that identical runs of this script
# can be verified against each other
digest = hashlib.sha256()
for name in final_times:
    digest.update(final_times[name].tobytes())
    digest.update(basin_times[name].tobytes())
print('stats digest (sha256):', digest.hexdigest()[:16])

# --- histogram of the three travel-time distributions ---
# the distributions span several orders of magnitude, so the bins are
# log-spaced and the x-axis is logarithmic
bins = np.logspace(2, 12, 61)
plt.figure()
for name, color in zip(final_times, ['tab:blue', 'tab:orange',
                                     'tab:green']):
    plt.hist(final_times[name], bins=bins, density=True, alpha=0.5,
             label=name, color=color)
plt.legend()
plt.xscale('log')
plt.xlabel('Travel time after %d iterations [s]' % num_iter)
plt.ylabel('Probability density')
plt.title('Seeding strategy comparison')
plt.tight_layout()
plt.savefig(os.path.join(f_path, 'seeding_strategies_travel_times.png'))
plt.show()
