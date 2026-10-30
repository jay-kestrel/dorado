# -*- coding: utf-8 -*-
"""
Core functions to handle the Lagrangian random walk movement of the particles.

Project Homepage: https://github.com/passaH2O/dorado
"""
from __future__ import division, print_function, absolute_import
from builtins import range, map
from math import cos
import logging
import time
import numpy as np
from numpy.random import random
from numpy import maximum, nansum

logger = logging.getLogger(__name__)


def random_pick_seed(choices, probs=None):
    """Randomly pick a number from array of choices.

    **Inputs** :

        choices : `ndarray`
            Array of possible values to draw from

        probs : `ndarray`
            *Optional*, can add weighted probabilities to draw

    **Outputs** :

        choices[idx] : `int`
            The randomly chosen value

    """
    # randomly pick tracer drop cell to use given a list of potential spots
    # (guard on None -- `not probs` is ambiguous for ndarray probs)
    if probs is None:
        probs = np.array([1 for i in list(range(len(choices)))])
    # find the corresp. index value from input 'choices' list of indices
    cutoffs = np.cumsum(probs)
    idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))

    return choices[idx]


def big_sliding_window(raster):
    """Creates 3D array organizing local neighbors at every index

    Returns a raster of shape (L,W,9) which organizes (along the third
    dimension) all of the neighbors in the original raster at a given
    index, in the order [NW, N, NE, W, 0, E, SW, S, SE]. Outputs are
    ordered to match np.ravel(), so it functions similarly to a loop
    applying ravel to the elements around each index.
    For example, the neighboring values in raster indexed at (i,j) are
    raster(i-1:i+2, j-1:j+2).ravel(). These 9 values have been mapped to
    big_ravel(i,j,:) for ease of computations. Helper function for make_weight.

    **Inputs** :

        raster : `ndarray`
            2D array of values (e.g. stage, qx)

    **Outputs** :

        big_ravel : `ndarray`
            3D array which sorts the D8 neighbors at index (i,j) in
            raster into the 3rd dimension at (i,j,:)

    """
    big_ravel = np.zeros((raster.shape[0],raster.shape[1],9))
    big_ravel[1:-1,1:-1,0] = raster[0:-2,0:-2]
    big_ravel[1:-1,1:-1,1] = raster[0:-2,1:-1]
    big_ravel[1:-1,1:-1,2] = raster[0:-2,2:]
    big_ravel[1:-1,1:-1,3] = raster[1:-1,0:-2]
    big_ravel[1:-1,1:-1,4] = raster[1:-1,1:-1]
    big_ravel[1:-1,1:-1,5] = raster[1:-1,2:]
    big_ravel[1:-1,1:-1,6] = raster[2:,0:-2]
    big_ravel[1:-1,1:-1,7] = raster[2:,1:-1]
    big_ravel[1:-1,1:-1,8] = raster[2:,2:]

    return big_ravel


def tile_local_array(local_array, L, W):
    """Take a local array [[NW, N, NE], [W, 0, E], [SW, S, SE]]
    and repeat it into an array of shape (L,W,9), where L, W are
    the shape of the domain, and the original elements are ordered
    along the third axis. Helper function for make_weight.

    **Inputs** :

        local_array : `ndarray`
            2D array of represnting the D8 neighbors around
            some index (e.g. ivec, jvec)

        L : `int`
            Length of the domain

        W : `int`
            Width of the domain

    **Outputs** :

        tiled_array : `ndarray`
            3D array repeating local_array.ravel() LxW times

    """
    return np.tile(local_array.ravel(), (L, W, 1))


def tile_domain_array(raster):
    """Repeat a large 2D array 9 times along the third axis.
    Helper function for make_weight.

    **Inputs** :

        raster : `ndarray`
            2D array of values (e.g. stage, qx)

    **Outputs** :

        tiled_array : `ndarray`
            3D array repeating raster 9 times

    """
    return np.repeat(raster[:, :, np.newaxis], 9, axis=2)


def clear_borders(tiled_array):
    """Set to zero all the edge elements of a vertical stack
    of 2D arrays. Helper function for make_weight.

    **Inputs** :

        tiled_array : `ndarray`
            3D array repeating raster (e.g. stage, qx) 9 times
            along the third axis

    **Outputs** :

        None :
            The input `tiled_array` is modified in place, with the
            borders in the first and second dimension set to 0.

    """
    tiled_array[0,:,:] = 0.
    tiled_array[:,0,:] = 0.
    tiled_array[-1,:,:] = 0.
    tiled_array[:,-1,:] = 0.
    return


def make_weight(Particles):
    """Create the weighting array for particle routing

    Function to compute the routing weights at each index, which gets
    stored inside the :obj:`dorado.particle_track.Particles` object
    for use when routing. Called when the object gets instantiated.

    **Inputs** :

        Particles : :obj:`dorado.particle_track.Particles`
            A :obj:`dorado.particle_track.Particles` object

    **Outputs** :

        Updates the weights array inside the
        :obj:`dorado.particle_track.Particles` object

    """
    L, W = Particles.stage.shape

    # calculate surface slope weights
    weight_sfc = (tile_domain_array(Particles.stage) \
                  - big_sliding_window(Particles.stage))
    weight_sfc /= tile_local_array(Particles.distances, L, W)
    weight_sfc[weight_sfc <= 0] = 0
    clear_borders(weight_sfc)

    # calculate inertial component weights
    weight_int = (tile_domain_array(Particles.qx)*tile_local_array(Particles.jvec, L, W)) \
                 + (tile_domain_array(Particles.qy)*tile_local_array(Particles.ivec, L, W))
    weight_int /= tile_local_array(Particles.distances, L, W)
    weight_int[weight_int <= 0] = 0
    clear_borders(weight_int)

    # get depth and cell types for neighboring cells
    depth_ind = big_sliding_window(Particles.depth)
    ct_ind = big_sliding_window(Particles.cell_type)

    # set weights for cells that are too shallow, or invalid 0
    weight_sfc[(depth_ind <= Particles.dry_depth) | (ct_ind == 2)] = 0
    weight_int[(depth_ind <= Particles.dry_depth) | (ct_ind == 2)] = 0

    # if sum of weights is above 0 normalize by sum of weights
    norm_sfc = np.nansum(weight_sfc, axis=2)
    idx_sfc = tile_domain_array((norm_sfc > 0))
    weight_sfc[idx_sfc] /= tile_domain_array(norm_sfc)[idx_sfc]

    norm_int = np.nansum(weight_int, axis=2)
    idx_int = tile_domain_array((norm_int > 0))
    weight_int[idx_int] /= tile_domain_array(norm_int)[idx_int]

    # define actual weight by using gamma, and weight components
    weight = Particles.gamma * weight_sfc + \
             (1 - Particles.gamma) * weight_int

    # modify the weight by the depth and theta weighting parameter
    weight = depth_ind ** Particles.theta * weight

    # if the depth is below the minimum depth then set weight to 0
    weight[(depth_ind <= Particles.dry_depth) | (ct_ind == 2)] = 0

    # if it's a dead end with only nans and 0's, choose deepest cell
    zero_weights = tile_domain_array((np.nansum(weight, axis=2) <= 0))
    deepest_cells = (depth_ind == tile_domain_array(np.max(depth_ind,axis=2)))
    # under stranding_handling = 'park', permanent land is not a valid
    # fallback destination while any non-land cell exists in the window:
    # pick the deepest non-land cell instead, and only fall back to the
    # legacy choice when the whole window is land
    if Particles.stranding_handling == 'park':
        not_land = (ct_ind != 2)
        any_not_land = np.any(not_land, axis=2)
        deepest_dry_aware = \
            (depth_ind == tile_domain_array(np.max(np.where(not_land,
                                                            depth_ind,
                                                            -np.inf),
                                                   axis=2)))
        deepest_cells = ((deepest_dry_aware & not_land) |
                         (deepest_cells &
                          tile_domain_array(~any_not_land)))
    choose_deep_cells = (zero_weights & deepest_cells)
    weight[choose_deep_cells] = 1.0

    # Final checks, eliminate invalid choices
    clear_borders(weight)

    # set weight in the true weight array
    Particles.weight = weight


def get_weight(Particles, ind):
    """Choose new cell location given an initial location.

    Function to randomly choose 1 of the surrounding 8 cells around the
    current index using the routing weights from make_weight.

    **Inputs** :

        Particles : :obj:`dorado.particle_track.Particles`
            A :obj:`dorado.particle_track.Particles` object

        ind : `tuple`
            Tuple (x,y) with the current location indices

    **Outputs** :

        new_cell : `int`
            New location given as a value between 1 and 8 (inclusive)

    """
    # randomly pick the new cell for the particle to move to using the
    # random_pick function and the set of weights
    if Particles.steepest_descent is not True:
        new_cell = random_pick(Particles.weight[ind[0], ind[1], :])
    elif Particles.steepest_descent is True:
        new_cell = steep_descent(Particles.weight[ind[0], ind[1], :])

    return new_cell


def calculate_new_ind(ind, new_cell, iwalk, jwalk):
    """Add new cell location (1-8 value) to the previous index.

    **Inputs** :

        ind : `tuple`
            Tuple (x,y) of old particle location

        new_cell : `int`
            Integer 1-8 indicating new cell location relative to the old
            one in a D-8 sense

        iwalk : `ndarray`
            A D8 array with the positive and negative x directions

        jwalk : `ndarray`
            A D8 array with the positive and negative y directions

    **Outputs** :

        new_ind : `tuple`
            tuple (x,y) of the new particle location

    """
    # add the index and the flattened x and y walk component
    # x,y walk component is related to the next cell chosen as a
    # 1-8 location
    new_ind = (ind[0] + jwalk.flat[new_cell], ind[1] + iwalk.flat[new_cell])

    return new_ind


def step_update(new_cell, distances, dx):
    """Get distance to new particle location.

    Function to check new location is some distance away from old one,
    also provides way to track the travel distance of the particles

    **Inputs** :

        new_cell : `int`
            Integer 1-8 indicating new location in D-8 way

        distances : `ndarray`
            D8 distances between cells

        dx : `float`
            Length along one square cell face

    **Outputs** :

        dist : `float`
            Distance between current (old) and new particle location

    """
    # compute the step distance to be taken
    dist = distances.flat[new_cell]*float(dx)

    return dist


def calc_travel_times(Particles, new_cell, ind, new_ind, dist):
    """Calculate travel time for particle to get to the new location.

    Function to calculate the travel time for the particle to get from the
    current location to the new location. Calculated by taking the inverse
    of the velocity at the old and new locations.

    **Inputs** :

        Particles : :obj:`dorado.particle_track.Particles`
            A :obj:`dorado.particle_track.Particles` object

        new_cell : `int`
            Integer 1-8 indicating new location in D-8 way

        ind : `tuple`
            Tuple (x,y) of the current location

        new_ind : `tuple`
            Tuple (x,y) of the new location

        dist : `float`
            Distance between current (old) and new particle location

    **Outputs** :

        trav_time : `float`
            Travel time it takes the particle to get from the current
            location to the new proposed location using the inverse of the
            average velocity

    """
    # make sure the new location is different from the current one
    if ind != new_ind:
        # get old position velocity value
        old_vel = Particles.velocity[ind[0], ind[1]]
        # new position velocity value
        new_vel = Particles.velocity[new_ind[0], new_ind[1]]
        # Compute diffusion term
        # (sample uniform distribution centered at 0 and diff_coeff wide)
        if Particles.diff_coeff > 0:
            diff = (0.5 - random())*Particles.diff_coeff
        else:
            diff = 0.0
        # Compute distance traveled in the orientation of mean flow path
        # If we moved backwards/orthogonal, step was instantaneous
        projected_dist = dist*max(0, cos(Particles.angles.flat[new_cell] -
                                  Particles.velocity_angle[ind[0], ind[1]]))
        # Compute average velocity over step
        trav_time = 0.5*projected_dist*(1/old_vel + 1/new_vel)*(1+diff)
    else:
        trav_time = 0  # particle did not move
    return trav_time


def check_for_boundary(new_inds, current_inds, cell_type):
    """Keep particles bordering a boundary cell from moving.

    Function to check the neighborhood of each particle's current
    location for boundary (edge) cells, and to cancel the proposed
    movement of any particle whose current location borders one, so
    that particles at the edge of the domain remain where they are
    for this iteration.

    **Inputs** :

        new_inds : `list`
            List [] of tuples (x,y) of new indices

        current_inds : `list`
            List [] of tuples (x,y) of old indices

        cell_type : `numpy.ndarray`
            Array of the different types of cells in the domain where 2 = land,
            1 = channel, 0 = ocean, and -1 = edge. If not explicitly defined
            then the values are estimated based on the depth array and the
            defined dry_depth

    **Outputs** :

        new_inds : `list`
            list [] of tuples (x,y) of new indices where any proposed
            indices outside of the domain have been replaced by the old
            indices so those particles will not travel this iteration

    """
    # Check if the new indices are on an edge (type==-1)
    # If so, then stop moving particle
    for i in range(0, len(new_inds)):
        # If particle borders an edge, cancel out any additional steps
        # (lower slice bounds are clamped at 0 so that particles on the
        # first row/column check a real neighborhood instead of silently
        # wrapping around the domain via a negative slice)
        if -1 in cell_type[max(0, current_inds[i][0]-1):current_inds[i][0]+2,
                           max(0, current_inds[i][1]-1):current_inds[i][1]+2]:
            new_inds[i][0] = current_inds[i][0]
            new_inds[i][1] = current_inds[i][1]

    return new_inds


def random_pick(probs):
    """Pick value from weighted probability array.

    Randomly pick a number weighted by array probs (len 8)
    Return the index of the selected weight in array probs

    **Inputs** :

        probs : `list`
            8 values indicating the probability (weight) associated with
            the surrounding cells for the random walk

    **Outputs** :

        idx : `int`
            1-8 value chosen randomly based on the weighted probabilities

    """
    # work on a copy so the shared weight array is not modified in place
    probs = probs.copy()
    probs[np.isnan(probs)] = 0  # any nans are assigned as 0
    cutoffs = np.cumsum(probs)  # cumulative sum of all probabilities
    # randomly pick indices from cutoffs based on uniform distribution
    idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))

    return idx


def steep_descent(probs):
    """Choose value with greatest probability, no longer random.

    Pick the array value with the greatest probability, no longer a
    stochastic process, instead just choosing the steepest descent.
    The current location (index 4) is excluded from consideration, and
    the maximum is computed after that exclusion; if no neighboring
    cell carries any weight (a dead end), the current location
    (index 4) is returned. The input array is copied and not modified.

    **Inputs** :

        probs : `float`
            8 values indicating probability (weight) associated with the
            surrounding cells

    **Outputs** :

        idx : `int`
            1-8 value chosen by greatest probs

    """
    # work on a copy so the shared weight array is not modified in place
    probs = probs.copy()
    # any nans become ignored too
    probs[np.isnan(probs)] = 0
    # remove initial location (index 4) from consideration
    probs[4] = 0
    # dead end: no neighbor carries any weight, so stay at the current
    # cell rather than drifting to the first (NW) slot
    if not np.any(probs > 0):
        return 4
    # remove any locations from consideration beneath max value
    max_val = np.nanmax(probs)
    probs[probs < max_val] = 0

    # will pick either the index corresponding to the max value if there
    # is just 1, or it will randomly choose between values in the event
    # of a tie
    cutoffs = np.cumsum(probs)  # cumulative sum of all probabilities
    idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))

    return idx


def stranding_park_decisions(Particles, inds_tuple, start_optional):
    """Evaluate the park-while-dry stranding state machine per particle.

    Under stranding_handling = 'park', a particle parks (does not move)
    whenever its current cell is dry or is permanent land (cell_type ==
    2). A stranded particle whose cell is wet again is released by rule:
    'immediate' releases it on the first step with a wet cell, 'rest'
    releases it once the accumulated stranded time reaches
    Particles.rest_time, and 'probability' releases it with per-step
    probability Particles.entrain_prob. Release is only ever evaluated
    from a wet cell, so re-wetting is a necessary condition for release
    under every rule.

    **Inputs** :

        Particles : :obj:`dorado.particle_track.Particles`
            A :obj:`dorado.particle_track.Particles` object with
            stranding_handling = 'park'.

        inds_tuple : `list`
            List of tuples (x,y) of the current particle locations.

        start_optional : `dict`
            Dictionary of lists of optional variable values, one per
            particle, carrying the previous step's 'stranded' flag and
            'stranded_time' total. Missing or None values are treated
            as never stranded.

    **Outputs** :

        park_mask : `list`
            List of bool, True wherever the particle parks this step.

        stranded_times : `list`
            List of float with the cumulative stranded time after this
            step (each parked step books Particles.stranding_dt).

    """
    prev_stranded = start_optional.get('stranded',
                                       [None]*len(inds_tuple))
    prev_times = start_optional.get('stranded_time',
                                    [None]*len(inds_tuple))
    park_mask = []
    stranded_times = []
    for i, (x, y) in enumerate(inds_tuple):
        stranded_time = prev_times[i] if prev_times[i] is not None else 0.0
        stranded = prev_stranded[i] if prev_stranded[i] is not None else 0
        wet = (Particles.depth[x, y] > Particles.dry_depth
               and Particles.cell_type[x, y] != 2)
        park = True
        if wet:
            if not stranded:
                # wet and not stranded: free to move
                park = False
            elif Particles.stranding_release == 'immediate':
                park = False
            elif Particles.stranding_release == 'rest':
                # held while wet until the stranded clock reaches the
                # rest time
                park = stranded_time < Particles.rest_time
            else:  # 'probability'
                # per-step Bernoulli re-entrainment trial
                park = not (random() < Particles.entrain_prob)
        park_mask.append(park)
        if park:
            stranded_times.append(stranded_time + Particles.stranding_dt)
        else:
            stranded_times.append(stranded_time)
    return park_mask, stranded_times


def particle_stepper(Particles, current_inds, travel_times, start_optional):
    """Step particles a single iteration.

    **Inputs** :

        Particles: :obj:`dorado.particle_track.Particles`
            A :obj:`dorado.particle_track.Particles` object.

        current_inds : `list`
            List of tuples of the current particle (x,y) locations in space

        travel_times : `list`
            List of initial travel times for the particles

        start_optional: 'dict of lists'
            A dictionary of lists of optional user inputs for the particle stepper. Options are shown in the particles object documentation.
        

    **Outputs** :

        new_inds : `list`
            List of the new particle locations after the single iteration

        travel_times : `list`
            List of the travel times associated with the particle movements
            
        updated_optional : `dict of lists`
            A dictionary of lists of optional outputs for the particle stepper. Options are shown in the particles object documentation.

    """
    # Use roi_grid from the particle object if it exists, otherwise use None
    ROI = getattr(Particles, 'roi_grid', None)
        
    inds = current_inds  # get indices as coordinates in the domain
    # split the indices into tuples
    inds_tuple = [(inds[i][0], inds[i][1]) for i in range(len(inds))]

    # park-while-dry stranding state machine: decide per particle
    # whether it parks this step; parked particles stay in place without
    # consulting the weight window, and book stranding_dt of travel time
    if Particles.stranding_handling == 'park':
        park_mask, stranded_times = \
            stranding_park_decisions(Particles, inds_tuple, start_optional)
    else:
        park_mask = [False]*len(inds_tuple)
        stranded_times = None

    # for each particle index get the weights
    new_cells = [get_weight(Particles, x)
                 if (x != (0, 0) and not park_mask[i]) else 4
                 for i, x in enumerate(inds_tuple)]
    # for each particle get the new index
    new_inds = list(map(lambda x, y:
                        calculate_new_ind(x, y, Particles.iwalk,
                                          Particles.jwalk)
                    if y != 4 else x, inds_tuple, new_cells))
    # move each particle to the new index
    dist = list(map(lambda x: step_update(x, Particles.distances,
                                          Particles.dx),
                    new_cells))
    # put new indices into array
    new_inds = np.array(new_inds, dtype=np.int32)
    # particles whose picked step distance is 0 did not move: keep them at
    # their current indices instead of overwriting them with the (0, 0)
    # sentinel, which collides with a real cell at the top-left corner
    # when a user-supplied distances array contains a 0
    zero_dist = np.array(dist) == 0
    new_inds[zero_dist] = np.array(inds, dtype=np.int32)[zero_dist]
    # see if the indices are at boundaries
    new_inds = check_for_boundary(new_inds, inds, Particles.cell_type)
    # transform from np array to list
    new_inds = new_inds.tolist()

    # add the travel times
    temp_travel = list(map(lambda w, x, y, z:
                           calc_travel_times(Particles, w, x, y, z),
                           new_cells, current_inds, new_inds, dist))
    # parked steps book the stranding clock instead of a travel time
    for i, parked in enumerate(park_mask):
        if parked:
            temp_travel[i] = Particles.stranding_dt
    # add to existing times
    travel_times = [travel_times[i] + temp_travel[i]
                    for i in range(0, len(travel_times))]
    travel_times = list(travel_times)

    # Optional variables
    updated_optional = {}
    
    if ROI is not None:
        updated_optional['roi_flag'] = [1 if ROI[x, y] == 1 else 0 for x, y in new_inds]

    for var, _ in start_optional.items():
        if var not in updated_optional:
            if var in Particles.available_particle_vars:
                grid = Particles.available_particle_vars[var]
                updated_optional[var] = [grid[x, y] for x, y in new_inds]

    # under 'park' the stranding state rides the same optional-variable
    # channel: the stranded flag of this step and the cumulative
    # stranded time
    if Particles.stranding_handling == 'park':
        updated_optional['stranded'] = [1 if parked else 0
                                        for parked in park_mask]
        updated_optional['stranded_time'] = stranded_times

    return new_inds, travel_times, updated_optional


def run_walk(Particles, num_iter, Np_tracer):
    """Run a complete particle walk of num_iter iterations.

    Run-level wrapper around
    :obj:`dorado.particle_track.Particles.run_iteration`
    that emits a single INFO record at run start (particle count and
    steepest-descent vs weighted random walk mode) and a single INFO record
    at run completion (iterations and wall time). Emits nothing inside the
    per-step loops, where tqdm remains the step-level progress surface.

    **Inputs** :

        Particles : :obj:`dorado.particle_track.Particles`
            A :obj:`dorado.particle_track.Particles` object with particles
            already generated.

        num_iter : `int`
            Number of iterations to route particles for

        Np_tracer : `int`
            Number of particles in the walk.

    **Outputs** :

        all_walk_data : `dict`
            Dictionary of all x and y locations and travel times, in the
            same format as the output of *run_iteration*.

    """
    mode = ('steepest descent' if Particles.steepest_descent is True
            else 'weighted random walk')
    logger.info('Walk run starting: %d particles, %s mode, '
                '%d iterations', Np_tracer, mode, num_iter)

    start_time = time.perf_counter()
    for i in list(range(0, num_iter)):
        all_walk_data = Particles.run_iteration()

    logger.info('Walk run complete: %d iterations in %.3f seconds',
                num_iter, time.perf_counter() - start_time)

    return all_walk_data
