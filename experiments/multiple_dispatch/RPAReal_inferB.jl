"""
Example of a custom inference backend implemented with multiple dispatch.

This script uses the same RPA model, data, and `SimulationSpec` as
`RPAReal.jl`, but does not use Turing. `InferBSpec` is a new inference-spec
type and the method `run_inference(::Model, ::InferBSpec)` implements a simple
prior-draw/random-search inference algorithm.
"""

using UncertaintyOptimization
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEq
using CSV
using DataFrames
using Random
using Serialization
using SciMLBase: successful_retcode

"""
    InferBSpec(; simulation, data, n_samples=1000, noise_sigma=3.0, rng)

Settings for the example inference backend. The backend draws parameter values
from the model priors, simulates each draw, and ranks the draws by Gaussian
log likelihood.
"""
struct InferBSpec <: UncertaintyOptimization.InferenceSpec
    simulation::SimulationSpec
    data::Vector{Float64}
    n_samples::Int
    noise_sigma::Float64
    rng::AbstractRNG

    function InferBSpec(;
        simulation::SimulationSpec,
        data::AbstractVector{<:Real},
        n_samples::Integer = 1000,
        noise_sigma::Real = 3.0,
        rng::AbstractRNG = MersenneTwister(6),
    )
        n_samples > 0 || error("n_samples must be positive")
        noise_sigma > 0 || error("noise_sigma must be positive")

        new(
            simulation,
            Float64.(data),
            Int(n_samples),
            Float64(noise_sigma),
            rng,
        )
    end
end

"""Flatten the observed-state trajectories in simulation output order."""
function infer_b_predictions(sols)
    return reduce(vcat, (vec(Array(sol)) for sol in sols); init = Float64[])
end

"""
    run_inference(model, spec::InferBSpec)

Custom inference implementation selected by multiple dispatch. It reuses the
standard model setup and `simulate!` implementation, while replacing Turing's
sampler and likelihood model with a simple prior-draw search.
"""
function UncertaintyOptimization.run_inference(model::Model, spec::InferBSpec)
    @info "Running custom InferB prior-draw inference"

    # This is the same setup performed by the Turing backend. It builds the
    # cached ODE problem, parameter setters, tunable symbols, and priors.
    UncertaintyOptimization.setup_model_for_inference(model, spec)
    priors = UncertaintyOptimization.make_priors(model)

    samples = Matrix{Float64}(undef, spec.n_samples, length(priors))
    losses = fill(Inf, spec.n_samples)
    loglikelihoods = fill(-Inf, spec.n_samples)

    for draw_index in 1:spec.n_samples
        draw = [rand(spec.rng, prior) for prior in priors]
        samples[draw_index, :] .= draw

        sols = try
            # This is the shared simulation path. The proposed values are in
            # model.tunable_symbols order, just as they are for Turing.
            simulate!(
                model,
                spec.simulation;
                sampled_uncertain_params = draw,
            )
        catch error
            @debug "Simulation failed for custom inference draw" exception = (error, catch_backtrace())
            nothing
        end

        if isnothing(sols) || any(sol -> !successful_retcode(sol), sols)
            continue
        end

        predicted = infer_b_predictions(sols)
        if length(predicted) != length(spec.data) || !all(isfinite, predicted)
            continue
        end

        residuals = predicted .- spec.data
        loss = sum(abs2, residuals)
        loglikelihood = -0.5 * loss / spec.noise_sigma^2 -
                        length(spec.data) * log(spec.noise_sigma * sqrt(2pi))

        losses[draw_index] = loss
        loglikelihoods[draw_index] = loglikelihood
    end

    results = DataFrame(samples, :auto)
    rename!(results, collect(model.tunable_symbols))
    results.loss = losses
    results.loglikelihood = loglikelihoods
    results.is_best = results.loglikelihood .== maximum(results.loglikelihood)
    sort!(results, :loglikelihood, rev = true)

    return results
end


# Load and compile the same model used by RPAReal.jl.
RPA_model = load_model("./test/test-data/RPA_real/cluster.yml")
@mtkcompile sys = System(RPA_model.equations, t)
model = Model(RPA_model, sys)

# Keep the simulation definition identical to the Turing example.
tspan = ((0.0, 10.0), (0.0, 5.45))
data_frame = CSV.read(
    joinpath(@__DIR__, "reference", "RPA_real_data.csv"),
    DataFrame;
    normalizenames = true,
    stripwhitespace = true,
)
data_selected = vcat(
    data_frame.experession20,
    data_frame.experession100,
    data_frame.expression1000,
) .- 17.6

sim_spec = SimulationSpec(
    t_obs = data_frame.time,
    obs_state = :A,
    initial_conditions = (24.0, 350.0),
    tspan = tspan,
    solver = AutoTsit5(Rosenbrock23(autodiff = false)),
    solver_opts = (dtmin = 1e-9,),
)

infer_b_spec = InferBSpec(
    simulation = sim_spec,
    data = data_selected,
    n_samples = 3000,
    noise_sigma = 3.0,
    rng = MersenneTwister(6),
)

# Julia selects the custom method above because the second argument is
# `InferBSpec`, rather than `TuringSpec`.
results = run_inference(model, infer_b_spec)

open(joinpath(@__DIR__, "mtk_a15_cluster_inferB_seed6.jls"), "w") do io
    serialize(io, results)
end
