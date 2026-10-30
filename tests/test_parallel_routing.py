import numpy as np

from dorado import particle_track, parallel_routing


class Params:
    pass


def _steady_params(n=10):
    params = Params()
    params.dx = 1
    params.depth = np.ones((n, n))
    params.stage = np.ones((n, n))
    params.qx = np.zeros((n, n))
    params.qy = np.ones((n, n))
    params.theta = 1
    params.model = 'DeltaRCM'
    return params


def test_parallel_routing_workers_generate_distinct_walks():
    """BUG-1: every pool worker used to inherit the parent's RNG state, so
    all cores routed identical walks and the combined result silently
    double-counted trajectories. Distinct per-worker seeds must yield more
    distinct walks than any single core could produce."""
    particles = particle_track.Particles(_steady_params())
    result = parallel_routing.parallel_routing(
        particles, num_iter=3, Np_tracer=2,
        seed_xloc=[3, 4], seed_yloc=[3, 4], num_cores=2)

    # 2 workers x 2 particles each = 4 walks; a single shared stream used
    # to produce exactly Np_tracer distinct walks among them
    assert len(result['xinds']) == 4
    distinct = {tuple(x) for x in result['xinds']}
    assert len(distinct) > 2
