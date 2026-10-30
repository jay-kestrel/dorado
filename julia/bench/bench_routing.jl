# Wall-clock routing benchmark for the Julia port; run from any directory
# with --output PATH. Mirrors benchmarks/bench_routing.py: same workloads,
# same timed region (run_iteration only — weight construction and seeding are
# excluded), same result JSON schema (schema_version 1).
#
# Staging reads the committed golden-fixture arrays (the same fields the
# Python harness builds from dorado/example_data/*.npz), so the Julia bench
# needs no npz reader and stays dependency-light.

using Dorado
using Dates
using JSON
using SHA
using Statistics
using Printf

const ROOT = dirname(@__DIR__)  # julia/
const REPO = dirname(ROOT)      # repo root
const GOLD = joinpath(ROOT, "goldens", "goldens", "init_parity")

struct BenchArgs
    output::String
    quick::Bool
    steps::Union{Nothing,Int}
    repeats::Union{Nothing,Int}
    seed::Int
end

function parse_args()
    output = nothing
    quick = false
    steps = nothing
    repeats = nothing
    seed = 0
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--output"
            output = ARGS[i + 1]; i += 2
        elseif arg == "--quick"
            quick = true; i += 1
        elseif arg == "--steps"
            steps = parse(Int, ARGS[i + 1]); i += 2
        elseif arg == "--repeats"
            repeats = parse(Int, ARGS[i + 1]); i += 2
        elseif arg == "--seed"
            seed = parse(Int, ARGS[i + 1]); i += 2
        else
            error("unknown argument: $arg")
        end
    end
    output === nothing && error("--output PATH is required")
    return BenchArgs(output, quick, steps, repeats, seed)
end

# Seed boxes and iteration counts mirror bench_routing.load_workload.
workload_config(name) =
    name == "rcm" ?
    (rows=15:16, cols=137:139, steps=20, dx=50.0) :
    (rows=20:29, cols=48:52, steps=50, dx=10.0)

function stage_particles(name, count, seed)
    p = Particles(joinpath(GOLD, "fixture_$name"))
    cfg = workload_config(name)
    generate_particles(p, count, collect(cfg.rows), collect(cfg.cols);
                       method="random")
    return p
end

function appended_steps(p)
    return sum(length(path) - 1 for path in p.walk_data["xinds"])
end

function measure(name, count, steps, repeats, seed)
    staged = stage_particles(name, count, seed)
    cfg = workload_config(name)
    # Warmup: one full untimed route on a throwaway copy. This is the Julia
    # analogue of the Python harness's single warmup step — it compiles every
    # specialization the timed region will touch, so timing measures steady
    # state rather than first-run compilation.
    warm = deepcopy(staged)
    for _ in 1:steps
        run_iteration(warm)
    end
    warm = nothing
    elapsed = Float64[]
    moved = Int[]
    for _ in 1:repeats
        particles = deepcopy(staged)
        t0 = time_ns()
        for _ in 1:steps
            run_iteration(particles)
        end
        t1 = time_ns()
        push!(elapsed, (t1 - t0) / 1e9)
        push!(moved, appended_steps(particles))
    end
    median_seconds = median(elapsed)
    return Dict(
        "workload" => name, "shape" => collect(size(staged.depth)),
        "dx" => cfg.dx, "backend" => "julia",
        "particles" => count, "steps" => steps, "repeats" => repeats,
        "seed" => seed, "n_threads" => Threads.nthreads(),
        "seconds" => elapsed, "median_seconds" => median_seconds,
        "microseconds_per_particle_step" => median_seconds * 1e6 / (count * steps),
        "appended_steps" => moved,
    )
end

function metadata()
    revision = strip(String(read(`git -C $REPO rev-parse HEAD`)))
    dirty = !isempty(strip(String(read(`git -C $REPO status --porcelain`))))
    sources = ("state.jl", "weights.jl", "walker.jl", "particles.jl", "npy.jl")
    return Dict(
        "timestamp_utc" => string(Dates.now(Dates.UTC)),
        "revision" => revision,
        "working_tree_dirty" => dirty,
        "routing_source_sha256" => Dict(
            f => bytes2hex(sha256(read(joinpath(ROOT, "src", f))))
            for f in sources),
        "julia" => string(VERSION),
        "platform" => string(Sys.MACHINE),
        "machine" => string(Sys.ARCH),
        "logical_cpus" => Sys.CPU_THREADS,
        "timer" => "time_ns",
        "timed_region" => "run_iteration only",
        "warmup_iterations" => "full route (one untimed pass over the workload's step count)",
        "warmup_note" => "Julia compiles per-specialization; the warmup route covers every specialization the timed region touches",
        "state_reset" => "deepcopy staged particles outside timing",
        "normalization" => "attempted particle-steps, including frozen particles",
        "thread_environment" => Dict(
            "JULIA_NUM_THREADS" => get(ENV, "JULIA_NUM_THREADS", "1")),
    )
end

function table(results)
    lines = ["| Field | Backend | Particles | Steps | Median (s) | us/particle-step |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in results
        push!(lines, @sprintf("| %s | %s | %d | %d | %.6f | %.3f |",
            row["workload"], row["backend"], row["particles"], row["steps"],
            row["median_seconds"], row["microseconds_per_particle_step"]))
    end
    return join(lines, "\n")
end

function main()
    args = parse_args()
    names = args.quick ? ["anuga"] : ["rcm", "anuga"]
    counts = args.quick ? [500] : [500, 2500, 5000]
    repeats = args.repeats !== nothing ? args.repeats : (args.quick ? 1 : 3)
    results = []
    for name in names
        cfg = workload_config(name)
        steps = args.steps !== nothing ? args.steps :
                (args.quick ? 3 : cfg.steps)
        for count in counts
            push!(results, measure(name, count, steps, repeats, args.seed))
        end
    end
    report = Dict(
        "schema_version" => 1,
        "metadata" => metadata(),
        "skipped" => [],
        "results" => results,
    )
    open(args.output, "w") do f
        JSON.print(f, report, 2)
    end
    println(table(results))
    println("wrote ", args.output)
    return 0
end

exit(main())
