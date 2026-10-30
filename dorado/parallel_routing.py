"""Parallel functionality using the native Python multiprocessing library.

Parallel functionality using multiprocessing (included with Python).
For local parallelization using CPU cores.

Project Homepage: https://github.com/passaH2O/dorado
"""
from __future__ import division, print_function, absolute_import
from builtins import range, map
import numpy as np
import sys
import os
import re
import string
import logging
from multiprocessing import Pool

import dorado.lagrangian_walker as lw

logger = logging.getLogger(__name__)


# convert run script into a function that returns a single dictionary 'data'
def run_iter(pobj):
    """Wrapper run script for the particle iterations.

    This is an internal function that should not be called directly. Rather it
    is wrapped by the `parallel_routing` function and mapped to the CPU cores
    to allow for particle routing in parallel. Uses a small `parallel_obj`
    class to hold a bunch of attributes and do the particle generation and
    routing.

    **Inputs** :

        pobj : :obj:`parallel_obj`
            Special parallel object.

    **Outputs** :

        all_walk_data : `list`
            Nested list of all x and y locations, travel times, and 
            any optional variables, with details same as input previous_walk_data

    """
    pobj.particles.generate_particles(pobj.Np_tracer, pobj.seed_xloc,
                                      pobj.seed_yloc,
                                      particle_size=pobj.particle_size,
                                      particle_type=pobj.particle_type)
    # do iterations
    all_walk = lw.run_walk(pobj.particles, pobj.num_iter, pobj.Np_tracer)

    return all_walk


def combine_result(par_result):
    """Combine results from each core.

    Take the parallel resulting list and combine the entries so that a single
    dictionary with the beginning/end indices and travel times for all
    particles is returned as opposed to a list with one dictionary per
    parallel process

    **Inputs** :

        par_result : `list`
            List of length(num_cores) with a dictionary of the beg/end indices
            and travel times for each particle computed by that process/core
    
    **Outputs** :
        single_result : `dict`
            Combined results dictionary containing all tracked variables.
    """
    # initiate final results dictionary
    single_result = {var: [] for var in par_result[0].keys()}

    # populate the dictionary
    for result in par_result:
        for var in result.keys():
            single_result[var].extend(result[var])

    return single_result


class parallel_obj:
    """
    Empty class to hold parallel parameters.

    This is an internal class used by `parallel_routing` that never needs to be
    defined outside of that function.
    """

    def __init__(self):
        """Initialize attributes."""
        parallel_obj.particles = None
        parallel_obj.num_iter = None
        parallel_obj.Np_tracer = None
        parallel_obj.seed_xloc = None
        parallel_obj.seed_yloc = None
        parallel_obj.particle_size = None
        parallel_obj.particle_type = None


def parallel_routing(particles, num_iter, Np_tracer, seed_xloc, seed_yloc,
                     num_cores, particle_size=None, particle_type=None):
    """Do the parallel routing of particles.

    Function to do parallel routing of particles. Does this by duplicating the
    call to *Particle.run_iteration()* across different processes.

    **Inputs** :

        particle : :obj:`dorado.particle_track.Particles`
            An initialized :obj:`particle_track.Particles` object with some
            generated particles.

        num_iter : `int`
            Number of iterations to route particles for

        Np_tracer : `int`
            Number of particles to generate.

        seed_xloc : `list`
            List of x-coordinates over which to initially distribute the
            particles.

        seed_yloc : `list`
            List of y-coordinates over which to initially distribute the
            particles.

        num_cores : `int`
            Number of processors/cores to use for parallel part

        particle_size : `float`, `int`, `list`, `numpy.ndarray`, optional
            Size attribute to assign to the newly generated particles in
            each parallel process, with the same semantics as the
            particle_size argument of
            :obj:`dorado.particle_track.generate_particles` (a single value
            applied to all Np_tracer particles, or a list/array of length
            Np_tracer). The default value is None (no sizes).

        particle_type : `str`, `list`, optional
            Name of the particle kind to assign to the newly generated
            particles in each parallel process, with the same semantics as
            the particle_type argument of
            :obj:`dorado.particle_track.generate_particles` (a single name
            applied to all Np_tracer particles, or a list of names of
            length Np_tracer). The default value is None (no types).

    **Outputs** :

        par_result : `list`
            List of length(num_cores) with a dictionary of the beg/end indices
            and travel times for each particle computed by that process/core

    """
    # make parallel object to assign to function
    pobj = parallel_obj()
    pobj.particles = particles
    pobj.num_iter = num_iter
    pobj.Np_tracer = Np_tracer
    pobj.seed_xloc = seed_xloc
    pobj.seed_yloc = seed_yloc
    pobj.particle_size = particle_size
    pobj.particle_type = particle_type

    # define list to pass to actual function that will be mapped to in parallel
    p_list = [pobj] * num_cores

    # create the parallel pool and run the process
    p = Pool(processes=num_cores)
    par_result = p.map(run_iter, p_list)
    p.terminate()

    # combine results before returning to user
    combined = combine_result(par_result)

    logger.info('Parallel routing complete: %d workers, %d particles '
                'per worker, %d iterations each', num_cores, Np_tracer,
                num_iter)

    return combined
