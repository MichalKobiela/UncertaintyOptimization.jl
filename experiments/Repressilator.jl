"""
Reproduce the Repressilator inference/design pipeline with
UncertaintyOptimization.jl's ODE workflow.

The reference implementation uses intrinsic SDE noise to break the symmetry of
the three identical genes. The package currently solves ODEs, so this version
uses a small asymmetric initial condition and adds measurement noise only when
constructing the synthetic inference data. It otherwise preserves the reference
parameterization, 1000-point post-transient trajectory, Fourier-domain target,
prior bounds, and two-dimensional design bounds.

The full script is intentionally expensive, matching the scale of the reference
workflow.
"""

using UncertaintyOptimization
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEq
using Turing
using Distributions
using FFTW
using CSV
using DataFrames
using Random
using Serialization
using Statistics

const MODEL_PATH = joinpath(@__DIR__, "..", "test", "test-data", "Repressilator.yaml")
const OUTPUT_DIR = joinpath(@__DIR__, "output", "repressilator")

const OBSERVATION_TIMES = collect(100.0:0.1:199.9)
const INITIAL_CONDITIONS = (0.0, 0.0, 1.0e-3)
const GROUND_TRUTH = Dict(:k_degradation => 0.5, :n => 2.0)

const TARGET_LENGTH = 1000
const TARGET_PERIOD = 250
const TARGET_HIGH_LENGTH = TARGET_PERIOD ÷ 3
const TARGET_AMPLITUDE = 500.0

function rectangular_wave(length::Int, period::Int, high_length::Int)
    low_length = period - high_length
    one_period = vcat(zeros(low_length), ones(high_length))
    repeats = cld(length, period)
    return repeat(one_period, repeats)[1:length]
end

function reference_spectrum(signal)
    transform = fft(signal)
    # The original objective uses abs.(fft(signal))[2:length(signal) ÷ 2].
    return abs.(transform[2:(length(signal) ÷ 2)])
end

function spectral_objective(signal, target_spectrum)
    signal_spectrum = reference_spectrum(signal)
    return sqrt(mean((signal_spectrum .- target_spectrum) .^ 2)) / length(signal)
end

# Load and compile the YAML model.
Repressilator_model = load_model(MODEL_PATH)
@mtkcompile sys = System(Repressilator_model.equations, t)
model = Model(Repressilator_model, sys)

# The reference observes 1000 samples of state B after a 100-time-unit
# transient. A small asymmetric initial condition replaces the symmetry
# breaking supplied by intrinsic noise in the original SDE.
sim_spec = SimulationSpec(
    t_obs = OBSERVATION_TIMES,
    obs_state = :B,
    initial_conditions = INITIAL_CONDITIONS,
    tspan = (0.0, 200.0),
    solver = Tsit5(),
    solver_opts = (abstol = 1.0e-8, reltol = 1.0e-6),
)

# Generate the synthetic inference data at the reference ground truth.
truth_sol = only(simulate!(model, sim_spec; parameters=GROUND_TRUTH))
rng = MersenneTwister(3)
observations = vec(Array(truth_sol)) .+ 2.0 .* randn(rng, length(OBSERVATION_TIMES))

# Infer the shared degradation rate and Hill coefficient.
turing_spec = TuringSpec(
    simulation = sim_spec,
    data = observations,
    noise_prior = InverseGamma(2, 3),
    noise_initial = 2.0,
    sampler = NUTS(0.65, init_ϵ = 0.005, max_depth = 8),
    n_samples = 300,
    n_chains = 1,
)

Random.seed!(0)
chain = run_inference(model, turing_spec)

# The original design stage uses 100 posterior draws.
posterior = select(DataFrame(chain), collect(model.tunable_symbols))
posterior = first(posterior, min(100, nrow(posterior)))

# Construct the phase-independent Fourier-domain design target.
const TARGET = TARGET_AMPLITUDE .* rectangular_wave(
    TARGET_LENGTH,
    TARGET_PERIOD,
    TARGET_HIGH_LENGTH,
)
const TARGET_SPECTRUM = reference_spectrum(TARGET)

function loss(_warmup_sol, predicted_sol; sys=nothing)
    signal = vec(Array(predicted_sol))
    spectral_objective(signal, TARGET_SPECTRUM)
end

# Thompson sampling over the two reference design bounds.
thompson_scan = CartesianScanner(
    simulation = sim_spec,
    scan = [
        (symbol = :k_transcription, values = LinRange(100.0, 1000.0, 100), kind = :value),
        (symbol = :K, values = LinRange(0.01, 10.0, 100), kind = :value),
    ],
    loss = loss,
)

thompson_results = run_scan(posterior, thompson_scan, model)
thompson_best = thompson_results[thompson_results.is_best, :]

# Count how often each parameter pair is selected.
thompson_counts = combine(
    groupby(thompson_best, [:k_transcription_value, :K_value]),
    nrow => :thompson_count,
)

# Evaluate the selected Thompson designs under every posterior draw.
evaluation_scan = CartesianScanner(
    simulation = sim_spec,
    scan = [
        (symbol = :k_transcription, values = unique(thompson_counts.k_transcription_value), kind = :value),
        (symbol = :K, values = unique(thompson_counts.K_value), kind = :value),
    ],
    loss = loss,
)

evaluation_results = run_scan(posterior, evaluation_scan, model)
evaluation_results = innerjoin(
    evaluation_results,
    thompson_counts;
    on = [:k_transcription_value, :K_value],
)

risk_summary = combine(
    groupby(evaluation_results, [:k_transcription_value, :K_value, :thompson_count]),
    :loss => median => :median_loss,
    :loss => (values -> quantile(values, 0.75)) => :q75_loss,
    :loss => std => :std_loss,
)
sort!(risk_summary, :q75_loss)

# Match the reference's low-median/low-upper-quartile cluster and return its
# centroid. Fall back to the lowest upper-quartile loss if the grid contains no
# design below both reference thresholds.
eligible = risk_summary[
    (risk_summary.median_loss .< 5.8) .& (risk_summary.q75_loss .< 6.5),
    :,
]

recovered_k_degradation = median(posterior.k_degradation)
recovered_n = median(posterior.n)

best_parameters = if nrow(eligible) > 0
    DataFrame(
        k_transcription = [mean(eligible.k_transcription_value)],
        K = [mean(eligible.K_value)],
        k_degradation = [recovered_k_degradation],
        n = [recovered_n],
        selection = ["reference_threshold_centroid"],
        candidate_count = [nrow(eligible)],
    )
else
    best = first(risk_summary, 1)
    DataFrame(
        k_transcription = best.k_transcription_value,
        K = best.K_value,
        k_degradation = [recovered_k_degradation],
        n = [recovered_n],
        selection = ["minimum_q75_loss"],
        candidate_count = [1],
    )
end

mkpath(OUTPUT_DIR)
CSV.write(joinpath(OUTPUT_DIR, "observations.csv"), DataFrame(time = OBSERVATION_TIMES, B = observations))
CSV.write(joinpath(OUTPUT_DIR, "posterior_samples.csv"), posterior)
CSV.write(joinpath(OUTPUT_DIR, "thompson_samples.csv"), thompson_best)
CSV.write(joinpath(OUTPUT_DIR, "evaluation.csv"), evaluation_results)
CSV.write(joinpath(OUTPUT_DIR, "risk_summary.csv"), risk_summary)
CSV.write(joinpath(OUTPUT_DIR, "best_parameters.csv"), best_parameters)

open(joinpath(OUTPUT_DIR, "posterior_chain.jls"), "w") do io
    serialize(io, chain)
end

println("Best Repressilator design:")
show(stdout, MIME("text/plain"), best_parameters)
println("\nOutputs written to: ", OUTPUT_DIR)
