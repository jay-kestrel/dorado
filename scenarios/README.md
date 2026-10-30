# Dorado demo scenarios

Nine runnable, repeatable scenarios that put numbers on a dorado routing
run: how fast it went, what it cost, and exactly what it processed. Every
scenario executes the full pipeline (field load → seeding → routing →
analysis) and emits **one metrics record** with the same schema, so runs
are comparable across scenarios, sizes, and revisions.

The suite is a pure consumer of dorado's public API — it changes no core
code, no packaging, and adds no dependencies (numpy plus the standard
library). It runs from a source checkout by path:

```bash
python scenarios/runner.py --list          # all nine, with descriptions
python scenarios/runner.py --all --quick   # CI-sized: the whole suite in ~45 s
python scenarios/runner.py --scenario multi-inlet --seed 7
python scenarios/runner.py --all           # full demo sizes
```

Each invocation writes `scenarios/results/<timestamp>/` — one
`metrics.json` per scenario plus a `summary.md` comparison table. That
directory is gitignored: the suite records evidence, it does not commit it.

## Scenarios

| # | Name | Workload | Full · quick | Headline metrics |
|---|------|----------|--------------|------------------|
| S1 | `steady-rcm-baseline` | DeltaRCM 120×240, dx 50, zero discharge; inlet rows 15–16, cols 137–139 | 5,000 parcels · 100 iters · 50 · 20 | wall, parcels/s, exits per boundary cell, travel-time percentiles |
| S2 | `steady-anuga-baseline` | ANUGA 100×100, dx 10, real qx/qy; rows 20–29, cols 48–52 | 5,000 · 50 · 50 · 10 | the S1 contract on a second field family |
| S3 | `particle-scaling` | Np ∈ {500, 2k, 8k, 32k}, 20 iters | {10, 40, 160, 640}, median of 5 | fitted log-log exponent + R² of routing time vs Np, tracemalloc peak |
| S4 | `domain-scaling` | right-anchored crops {120, 180, 240} cols, 2,000 parcels · 300 iters | {60, 120} cols · 200 parcels | routing cost vs domain area, peak RSS |
| S5 | `unsteady-flow` | bundled `examples/unsteady_data` CSVs, one reload + seed + iterate per timestep | all timesteps · 4 | grids loaded, wall, per-step work |
| S6 | `multi-inlet` | two seed patches via two `generate_particles` calls | 2 × 2,500 · 2 × 20 | per-inlet seeded/routed/exits/travel times |
| S7 | `parallel-speedup` | `parallel_routing`, cores ∈ {1, 2, 4, 8}, constant total parcels | {1, 2} · 800 | speedup, parallel efficiency, merged return counts |
| S8 | `determinism-repeat` | identical seeded workload, R repeats | R = 3 · R = 2 | per-repeat walk-data SHA-256, timing CV |
| S9 | `exposure-fractions` | S1-style run + exposure analysis layer | 500 · 50 | exit fractions per boundary cell, exposure-time percentiles |

Every case carries its own built-in assertion, so "demo" and "test" are
the same run at different sizes: S1/S2 require `seeded == routed + active`;
S3 requires monotone median routing time; S4 gates against material
routing-time inversions (0.9× per wider crop in full mode; a jitter-safe
0.7× in quick, where the two-second points sit inside the measured
0.85–1.13 noise band); S5 requires `grids_loaded == timesteps` and a walk
continuation that neither drops nor duplicates particles; S6 requires
per-inlet counts to sum to Np; S7 requires every core to return exactly
its share; S8 requires byte-identical walk-data hashes across repeats —
the tripwire that fires first if routing behavior ever drifts.

## Metric taxonomy

One record schema for all nine cases (`scenarios/core.RECORD_SCHEMA`);
fields a scenario does not measure are `null`, never absent.

- **`speed`** — `wall_seconds`; `phases_seconds` (per-phase: `load_field`,
  `seeding`, `routing`, `analysis`, `export`); `particles_per_second` and
  `steps_per_second` over the routing phase.
- **`efficiency`** — `peak_rss_bytes` (`ru_maxrss`; `null` on Windows, which
  has no `resource` module) and `peak_tracemalloc_bytes` (sampled across the
  allocation-heavy seeding/routing phases); `scaling_exponent`/`scaling_r2`
  (log-log fits in S3/S4); `parallel_speedup`/`parallel_efficiency` (S7,
  at the largest core count).
- **`processed`** — everything the run touched: `field_shape`/`field_dtype`/
  `field_bytes`; `iterations`; `particles_seeded`;
  `particles_routed_to_boundary` (frozen in the edge-adjacent ring) and
  `particles_active_at_end` (still in the interior, including parcels parked
  on dry cells); `steps_appended`; `exits_per_boundary_cell` (counts, keyed
  `"x,y"` — fractions of seeded parcels in S9); `unique_cells_visited`;
  `travel_time_seconds` (min/median/p90/max of each parcel's final travel
  time, in dorado's model seconds — not wall time); `per_inlet_seeded` (S6,
  full per-inlet stats block); `grids_loaded` (S5); `walk_data_sha256`
  (dtype-normalized digest of the walk data — the determinism fingerprint);
  `timing_cv` (S8, population std/mean of per-repeat routing times).

All processed-work counters derive from one analyzer
(`scenarios.core.analyze_walk`) over `walk_data`, so counts stay consistent
across scenarios.

## Full-run evidence

Real records from this repository, full mode, seed 0, captured on the
development sandbox (Linux-x86_64, CPython 3.13.14, numpy 2.3.5, rev
`21d007d`, clean tree). Timings come from a shared, throttled machine —
absolute values are indicative; the shapes and ratios are the story.

**S1 `steady-rcm-baseline`** — 5,000 parcels routed for up to 100
iterations in 44.83 s wall (43.96 s of it routing):

```json
{
  "meta": {"scenario": "steady-rcm-baseline", "quick": false, "seed": 0,
           "git_rev": "21d007d", "git_dirty": false,
           "platform": "Linux-x86_64"},
  "speed": {"wall_seconds": 44.83,
            "phases_seconds": {"load_field": 0.001, "seeding": 0.384,
                               "routing": 43.963, "analysis": 0.473},
            "particles_per_second": 113.73, "steps_per_second": 11247.44},
  "efficiency": {"peak_rss_bytes": 260550656,
                 "peak_tracemalloc_bytes": 40940824},
  "processed": {"field_shape": [120, 240], "field_dtype": "float32",
                "field_bytes": 115200, "iterations": 100,
                "particles_seeded": 5000,
                "particles_routed_to_boundary": 0,
                "particles_active_at_end": 5000,
                "steps_appended": 494475,
                "exits_per_boundary_cell": {},
                "unique_cells_visited": 1990,
                "travel_time_seconds": {"min": 1.63e+11, "median": 3.68e+11,
                                        "p90": 4.07e+11, "max": 4.75e+11},
                "walk_data_sha256": "af9a28e1..."}
}
```

`routed_to_boundary` is genuinely 0 on this workload: with purely diffusive
transport (zero discharge) and a 100-iteration budget, parcels seeded near
column 138 cannot cross the ~100 columns to the edge-adjacent freeze ring,
so they end the run active in the interior. Exit counting engages on
workloads whose transits reach the edge ring — e.g. `domain-scaling` seeds
interior wet cells and freezes a real share at its edge. The counting
invariant `seeded == routed + active` is what carries the guarantee.

**S3 `particle-scaling`** — routing cost is almost perfectly linear in the
parcel count: the fitted log-log exponent is **1.018 at R² 0.9994**
(route medians 0.879 s at Np=500, 3.287 s at 2k, 14.828 s at 8k, 58.64 s
at 32k):

```json
{
  "meta": {"scenario": "particle-scaling", "quick": false, "seed": 0,
           "git_rev": "21d007d", "git_dirty": false},
  "speed": {"wall_seconds": 81.96,
            "phases_seconds": {"route_np500_r0": 0.879,
                               "route_np2000_r0": 3.287,
                               "route_np8000_r0": 14.828,
                               "route_np32000_r0": 58.64},
            "particles_per_second": 545.7},
  "efficiency": {"peak_rss_bytes": 367288320,
                 "peak_tracemalloc_bytes": 74920422,
                 "scaling_exponent": 1.0177,
                 "scaling_r2": 0.9994},
  "processed": {"particles_seeded": 32000, "steps_appended": 672000}
}
```

## Parallel routing note

Current master draws **one independent seed per worker** from the parent
RNG (`np.random.randint(0, 2**31 - 1, size=num_cores, dtype=np.int64)` in
`dorado/parallel_routing.py`) and reseeds each worker's global RNG before
it seeds its own parcels. Verified on this repository: distinct worker
seeds produce distinct worker trajectories, and two `parallel_routing`
calls under the same parent seed reproduce hash-identical combined walk
data. The draw is one explicit int64 per worker at any core count, so the
historical Windows int32 seed-range concern does not apply to this code.

The `parallel-speedup` scenario accordingly claims **speedup only** — it
never asserts cross-worker trajectory distinctness, because workers reseed
their own parcels and `parallel_routing` merges worker results before
return; per-worker walks are observable only through the combined count.

## Testing

`pytest tests/test_scenarios.py` runs the suite in quick mode,
parametrized over the registry: schema conformance, sparse-field
nullness, counting invariants, and per-scenario criteria (S3 scaling fit,
S4 area fit, S7 parallel shares, S8 cross-run hash identity, S5 grid
accounting, S6 inlet sums, S9 fraction bounds, and the named
missing-data error path). Following the `benchmarks/` rule, **timings are
recorded, never gated** — no test asserts an absolute wall-time
threshold. The quick suite completes in ~43 s on a throttled runner,
inside the ~2-minute CI budget.
