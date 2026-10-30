.. _particle_sizes_types:

Particle Kinds and Sizes
========================

By default every particle routed by a :obj:`dorado.particle_track.Particles`
object is an identical passive tracer. Particles can optionally be given
per-particle attributes with behavior: a numeric ``particle_size`` and/or a
named ``particle_type`` (a *kind*), passed to
:obj:`dorado.particle_track.Particles.generate_particles`.

Kinds
-----

A kind is a named type with preset behavior multipliers, defined in the
``KIND_PRESETS`` dictionary in :download:`lagrangian_walker.py
<../../../dorado/lagrangian_walker.py>`. The shipped presets are illustrative
defaults that users may inspect and extend (adding a key to the dictionary
registers a new kind):

.. list-table::
   :header-rows: 1

   * - kind
     - travel_time_multiplier
     - diffusion_multiplier
     - deposition_probability
   * - ``sand``
     - 1.2
     - 0.5
     - 0.02
   * - ``mud``
     - 1.0
     - 1.0
     - 0.005
   * - ``plankton``
     - 1.0
     - 1.2
     - 0.0

The multipliers scale the particle's travel time and diffusion term each
step, and the deposition probability is the chance per step that a moving
particle settles. Deposition is permanent: a deposited particle stops moving
for the rest of the run (its ``deposited`` flag flips to 1).

Sizes
-----

A size is a numeric attribute (must be > 0) that scales transport behavior
when no kind overrides it: travel time is multiplied by the size (bigger =
slower) and the diffusion term by 1/size (smaller = more diffusion). When
both a size and a kind are given for a particle, the size and preset
multipliers multiply and the deposition probability comes from the preset.

Each argument accepts either a single value applied to all newly generated
particles, or a list/array with one entry per particle::

   particles.generate_particles(50, seed_xloc, seed_yloc)
   particles.generate_particles(20, seed_xloc, seed_yloc, particle_size=2.0)
   particles.generate_particles(20, seed_xloc, seed_yloc, particle_type='sand')
   particles.generate_particles(3, seed_xloc, seed_yloc,
                                particle_size=[0.5, 1.0, 2.0],
                                particle_type=['sand', 'mud', 'plankton'])

Inputs are validated all-or-nothing per call: mixing a scalar with a list of
the wrong length raises a ``ValueError`` stating the expected and received
lengths, and unknown kind names raise a ``ValueError`` listing the known
kinds.

Persistence
-----------

Unlike the optional variables that are re-sampled from the grid at every
step, these attributes are *persistent*: the same per-particle value is
appended to the walk data at every step (under the walk_data keys
``particle_size``, ``particle_type``, and the derived ``deposited`` flag,
using the usual ``[particle][iter]`` shape), and ``deposited`` flips to 1
permanently once a particle settles. Batches generated without an attribute
carry ``None`` entries for it, so histories stay aligned when batches are
merged.

If no ``particle_size`` or ``particle_type`` is provided in any
``generate_particles`` call, none of the new walk data keys are created and
particle behavior is exactly that of the passive-tracer random walk.

All particles in one :obj:`dorado.particle_track.Particles` object share the
routing weight field: heterogeneity comes from these per-particle behavior
modifiers, not from separate per-kind weight fields. See
``examples/particle_sizes_types.py`` for a runnable example.
