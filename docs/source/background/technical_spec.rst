.. _technical_spec:

=======================
Technical Specification
=======================

This page pins the behavioral contract of the routing core as built on ``master`` (version 2.7.1). It is sourced from the code in ``dorado/particle_track.py`` and ``dorado/lagrangian_walker.py``; the mathematical background for the weighting and travel-time formulas is on the :ref:`background` page. Where this page and a docstring disagree, the code wins — please open an issue.

Grid and coordinate model
-------------------------

- The domain is a rectangular grid of square cells with side length ``dx``. Particles are tracked by grid indices; ``coord2ind`` and ``ind2coord`` (in :obj:`dorado.particle_track`) convert to and from real coordinates.
- Movement is a D-8 random walk: the potential next location of a particle is one of the 8 cells surrounding its current cell. Internally each cell's neighborhood is organized as a 9-slot vector ordered ``[NW, N, NE, W, center, E, SW, S, SE]`` (``big_sliding_window``), with the center slot (index 4) never selected as a move target.
- Domain edges are cells with ``cell_type == -1``. Particles are seeded and routed on the interior.

``modelParams`` fields and validation
-------------------------------------

Validation happens once, when a :obj:`dorado.particle_track.Particles` object is constructed from a :obj:`dorado.particle_track.modelParams` instance. Inputs are copied (the caller's arrays are never modified); required-but-missing inputs raise ``ValueError``.

Required fields:

- ``dx`` — length along one square cell face. Undefined ⇒ ``ValueError``.
- Water depth — provide ``depth`` directly, or ``stage`` **and** ``topography`` (``depth = stage - topography``, negative values clamped to 0). Otherwise ``ValueError``.
- Water stage — provide ``stage`` directly, or ``topography`` **and** ``depth`` (``stage = topography + depth``). Otherwise ``ValueError``.
- Flow field — provide discharge (``qx``/``qy``) or velocity (``u``/``v``); the missing pair is derived (for ``model == 'DeltaRCM'`` inputs, ``u = qx*depth/(depth² + 1e-8)`` and ``qx = u*depth``). Otherwise ``ValueError``.

Axis convention for flow inputs: for ``model == 'DeltaRCM'``, discharge and velocity components are taken as given. For any other value of ``model`` (including none), the components are rotated on assignment — internally ``qx = -qy_input``, ``qy = qx_input`` (and identically for velocities). Be aware of this when preparing non-DeltaRCM inputs; the bundled ANUGA example data is shaped to this convention.

Derived at construction:

- Velocity magnitude, floored at ``1e-8`` to avoid infinite travel times; tiny ``u``/``v`` components are floored in a sign-preserving way so velocity angles keep pointing along the flow direction.
- ``velocity_angle = arctan2(-u, v)`` per cell, used for the flow-projection in travel-time calculation.
- Routing weights for the whole domain (``make_weight``), built once at construction.

Optional fields and their defaults (each defaulting decision is logged at INFO):

- ``theta = 1.0`` — depth weighting exponent for the random walk.
- ``gamma = 0.05`` — blend between surface-gradient and inertial routing; values outside ``[0, 1]`` raise ``ValueError`` (out-of-range values would produce negative weights).
- ``diff_coeff = 0.2`` — travel-time diffusion coefficient; ``0.0`` disables diffusion (also forced when ``steepest_descent`` is on), negative values are raised to 0 with a warning, and values ≥ 2 warn that diffusion behaves non-physically.
- ``dry_depth = 0.1`` — minimum depth (m) for a cell to count as wet.
- ``cell_type`` — array where 2 = land, 1 = channel, 0 = ocean, -1 = edge. If absent, estimated from the depth field: dry cells (``depth < dry_depth``) become land, the domain border becomes edge.
- ``steepest_descent = False`` — routing-mode toggle; ties between equally weighted neighbors are broken randomly.
- ``stranding_handling = 'park'`` — dry-cell behavior. ``'park'`` stops a particle on a dry or land cell and releases it by rule once its cell is wet again (see *Stepping semantics*); ``'none'`` restores the exact legacy behavior, with no stranded variables tracked.
- ``stranding_release = 'immediate'`` — release rule for a re-wetted stranded particle: ``'immediate'`` releases on the first wet step, ``'rest'`` holds it until ``rest_time`` seconds of model time have accumulated while stranded, ``'probability'`` releases it per step with probability ``entrain_prob``.
- ``rest_time = 0.0`` — seconds of model time a particle must stay stranded before release under ``stranding_release = 'rest'``; must be ≥ 0.
- ``entrain_prob = 1.0`` — per-step re-entrainment probability under ``stranding_release = 'probability'``; must be in ``(0, 1]``.
- ``stranding_dt = 1.0`` — model-time seconds booked onto a parked particle's stranded clock for each step it stays parked; must be > 0.
- ``roi_grid`` — optional binary region-of-interest grid, shape-validated against the depth array; registers an ``roi_flag`` tracked variable.
- ``particle_variables`` — optional list naming additional fields to sample into ``walk_data`` at every step (from ``depth``, ``stage``, ``qx``, ``qy``, ``u``, ``v``, plus ``roi_flag`` when an ROI grid is given).
- ``verbose`` — **deprecated**, see the logging section of the :ref:`userguide`.

Seeding particles
-----------------

``Particles.generate_particles(Np_tracer, seed_xloc, seed_yloc, seed_time=0, method='random', previous_walk_data=None, particle_size=None, particle_type=None)``:

- ``method='random'`` distributes the requested number of particles across the seed lists by weighted random draw (``random_pick_seed``); ``method='exact'`` places particles at the defined locations.
- Inputs are validated by ``gen_input_check``; seed locations are validated against the domain bounds and out-of-bounds seeds raise ``ValueError`` rather than silently wrapping around the domain edges via negative indexing.
- ``particle_size`` and ``particle_type`` attach per-particle behavior attributes (see *Per-particle kinds and sizes* below); each accepts a single value applied to all newly generated particles or a list/array with one entry per particle, validated all-or-nothing per call — mixing a scalar with a list of the wrong length raises ``ValueError`` stating the expected and received lengths, and unknown kind names raise ``ValueError`` listing the known kinds.
- Successive calls append to the same ``walk_data``, so particles can be seeded in several groupings before routing; ``previous_walk_data`` folds in independent walk data (e.g. loaded from a ``.json`` file), indexed ``[field][particle][iteration]``.

Stepping semantics
------------------

``Particles.run_iteration(target_time=None, max_iter=10000)`` advances every particle:

- With no ``target_time``, one iteration is run. With a ``target_time``, each particle keeps stepping (in per-particle sub-iterations) until its accumulated travel time reaches the target or ``max_iter`` steps are exhausted.
- Calling it before any particles exist raises ``ValueError`` (``walk_data is None``).
- Per step: the neighbor weight vector is read from the prebuilt weight field (``get_weight``), the next cell is drawn by ``random_pick`` (weighted random) or ``steep_descent`` (max-weight neighbor, excluding the current cell, ties broken randomly, staying put at a dead end), and the move is mapped to indices and distances.
- Boundary handling: a particle whose *current* location borders an edge cell (``cell_type == -1``) is frozen for the iteration — its proposed move is cancelled and nothing is appended to its ``walk_data`` history. Independently, a particle whose proposed step would leave the grid entirely is frozen in place: when the border cells are water there is no ``-1`` edge for the boundary check to react to, and an unchecked step would walk off-grid and silently wrap through downstream index lookups.
- Stranding (``stranding_handling = 'park'``): a particle whose current cell is dry (``depth ≤ dry_depth``) or permanent land (``cell_type == 2``) parks — it does not consult the routing weights, stays in place, and appends a record with an unchanged position, a travel time advanced by ``stranding_dt``, and its ``'stranded'`` flag set to 1 (its ``'stranded_time'`` total accumulates the parked steps). A parked particle whose cell is wet again is released by ``stranding_release`` — re-wetting is a necessary condition for every release rule. Under ``'none'`` none of this runs and no stranded variables are tracked; walk data carried over from a ``'none'`` walk is zero-padded when the walk is continued.
- Weight fallback: if every neighbor of a cell is masked (dry, land, or border), the deepest surrounding cells are given weight 1 so the walk can continue (dead ends route to the deepest neighbor rather than stalling). Under ``stranding_handling = 'park'`` the fallback is dry-aware: permanent land is not a fallback destination while any non-land cell exists in the neighborhood.

Travel times
------------

For a move of Euclidean distance ``dist`` from cell ``o`` to cell ``i``:

.. math::

    T_t = 0.5 \times d_{eq} \times \left(\frac{1}{v_o} + \frac{1}{v_i}\right) \times \left(1 + \left[dc \times U(-0.5, 0.5)\right]\right)

- ``d_eq = dist × max(0, cos(φ_step − φ_flow))`` — the step distance projected onto the local mean-flow orientation; backward or orthogonal steps (non-positive projection) are instantaneous.
- The diffusion term ``diff = (0.5 − U(0,1)) × diff_coeff`` is a uniform draw in ``[−dc/2, +dc/2]``, re-introducing diffusivity lost to the gridded scheme.
- A particle that does not move accrues no travel time — except a parked particle under ``stranding_handling = 'park'``, which books ``stranding_dt`` of model time per parked step (see *Stepping semantics*).

Per-particle kinds and sizes
----------------------------

By default every routed particle is an identical passive tracer. Optionally, ``generate_particles`` accepts a numeric ``particle_size`` (> 0) and/or a named ``particle_type`` (a *kind*, a key of ``KIND_PRESETS`` in ``lagrangian_walker.py``; shipped presets: ``sand``, ``mud``, ``plankton`` with travel-time, diffusion, and deposition multipliers):

- Each step, the kind's ``travel_time_multiplier`` scales the computed travel time, its ``diffusion_multiplier`` scales the diffusion term, and its ``deposition_probability`` is the per-step chance the particle settles permanently.
- Deposition is permanent: a deposited particle stops moving for the rest of the run and its ``'deposited'`` walk_data flag stays 1.
- A size without a kind scales travel time by the size (bigger = slower) and the diffusion term by ``1/size`` (smaller = more diffusion). When both a size and a kind are given, the size and preset multipliers multiply and the deposition probability comes from the preset.
- Unlike the optional variables re-sampled from the grid at every step, these attributes are *persistent*: the same per-particle value is appended to ``walk_data`` at every step, under the keys ``particle_size``, ``particle_type``, and the derived ``deposited`` flag. Batches generated without an attribute carry ``None`` entries so histories stay aligned when batches are merged.
- If no ``particle_size`` or ``particle_type`` is provided in any ``generate_particles`` call, none of the new walk_data keys are created and behavior is exactly the passive-tracer random walk.
- All particles in one ``Particles`` object share the routing weight field; heterogeneity comes from these per-particle behavior modifiers, not from separate per-kind weight fields. For user-facing usage, see :ref:`particle_sizes_types`.

The ``walk_data`` schema
------------------------

``Particles.walk_data`` is a dictionary; ``run_iteration`` returns the same dictionary it stores.

- Base keys: ``'xinds'``, ``'yinds'``, ``'travel_times'``.
- Each value is a nested list indexed ``[particle][iteration]``, e.g. ``walk_data['travel_times'][5][10]`` is the travel time of the 5th particle at the 10th iteration.
- Positions are stored as grid indices; convert with ``coord2ind``/``ind2coord``.
- Tracked optional variables (from ``modelParams.particle_variables``) appear as additional keys with the same shape; a variable with no recorded value yet is ``[[None]]`` per particle.
- When particles were generated with a size and/or kind, the persistent attribute keys ``particle_size``, ``particle_type``, and ``deposited`` appear with the same shape (see *Per-particle kinds and sizes* above).
- When ``stranding_handling = 'park'``, the keys ``'stranded'`` (0/1 per step) and ``'stranded_time'`` (cumulative model-time seconds stranded) appear with the same shape, making the time each particle spends stranded measurable from the walk data (see *Stepping semantics*).
- ``clear_walk_data()`` resets the attribute to ``None``.

Randomness and reproducibility
------------------------------

All stochasticity — random seed placement, neighbor selection, and the travel-time diffusion draw — uses NumPy's global random state. ``dorado`` never seeds it; callers who need reproducible runs should seed the global RNG (e.g. ``np.random.seed(...)``) before generating and routing particles, as the test suite does. ``parallel_routing`` draws one seed per worker from the parent's global RNG and reseeds each worker before it walks, so its workers produce independent — not identical — random walks, and a run stays reproducible when the global RNG is seeded before the call.

Assumptions and limitations
---------------------------

- Particles are point quantities located at cell centers; the gridded placement loses travel-time accuracy, which is partly compensated by the ``diff_coeff`` term.
- Without per-particle attributes every particle is an identical passive tracer; kinds and sizes modify per-particle transport behavior but do not change the shared weight field or the D-8 move distribution itself.
- The grid is rectangular with square cells; unstructured model output must first be interpolated onto the grid (``unstruct2grid`` helps).
- Only depth, stage, discharge, and velocity fields are consumed; the package holds no state between runs beyond ``walk_data``.
- Velocities are floored at ``1e-8``: cells with (near-)zero velocity have very large, but finite, travel times rather than infinite ones.
