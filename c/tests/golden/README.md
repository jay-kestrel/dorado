# Golden walk-data

`walk_data_rcm_s200_i5.json` — the Python reference's output for the
canonical end-to-end check: the `bench/fixture_rcm` fixture (120x240,
float64-normalized), 200 particles, 5 iterations, seed 0.

Produced by `benchmarks/run_reference.py` on 2026-10-30 in the repo
sandbox (dorado master @ df863fc, numpy-legacy MT19937 stream). The file
carries all five walk-data lists: `xinds`, `yinds`, `travel_times`,
`stranded`, `stranded_time`.

Regenerate with:

```sh
python benchmarks/run_reference.py --input bench/fixture_rcm \
    --particles 200 --iterations 5 --seed 0 --out /tmp/golden_ref
```

The C driver must reproduce this file **byte-for-byte** — the
`e2e_compare` ctest diffs the driver's `walk_data.json` against it. The
comparison is strict because the port replicates the reference's RNG
stream, draw order, and operation order; a mismatch is a port bug and is
never resolved by loosening the comparison (spec: "Golden outputs are the
merge criteria").
