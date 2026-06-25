#=
For one posterior sample, the script performs a grid 
search over scaling factors for design parameters,
runs a warmup simulation, then evaluates how close 
the warmup state and final post-warmup 
prediction are to a target value.
=#
using Revise
using UncertaintyOptimization
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D;
using OrdinaryDiffEq
using CSV, Tables
using SymbolicIndexingInterface
using Serialization
using CSV, Tables
using Plots
using DataFrames
using Random
using Statistics
using StatsBase: countmap


RPA_model = load_model("./test/test-data/RPA_real/opt.yml")
@mtkcompile sys = System(RPA_model.equations, t)
model = Model(RPA_model, sys)

time = CSV.File(joinpath(@__DIR__, "RPA_real_data/time_points.csv")).time
sim_spec = SimulationSpec(
    t_obs = time,
    obs_state = :A,
    initial_conditions = (24.0, 350.0),
    tspan = (0.0, 10.0),
    solver = Rosenbrock23(),
    solver_opts = (dtmin = 1e-9,),
)

# load the precomputed posterior from a single chain for now
posterior = open(string(@__DIR__)*"/reference/rpareal_chain_reference.jls", "r") do io
        # extract and convert into a DataFrame
        posterior_sample = deserialize(io)[1000:2000]
        posterior_df = DataFrame(posterior_sample)
        # rng = MersenneTwister(4)
        # draw_index = rand(rng, 1:nrow(posterior_df))
        # posterior_df[draw_index:draw_index, :]
end

# loss function for the evaluation of the design parameters
function loss(warmup_sol, predicted_sol; sys=nothing)
    isnothing(sys) && error("loss requires the model system as `sys` to locate state A.")

    A_idx = findfirst(isequal(getproperty(sys, :A)), unknowns(sys))
    isnothing(A_idx) && error("State A was not found in the model unknown order.")

    background_fluorescence = 17.6
    adjusted_predicted = Array(predicted_sol)[end] + background_fluorescence
    warmup = warmup_sol[A_idx, end] + background_fluorescence
    
    target = 50
    (((warmup - target).^2) + (adjusted_predicted - target).^2) / 2
end


## thompson sampling 
thompson_scan = CartesianScanner(
    simulation = sim_spec,
    scan = [
        (symbol = :kx2, values = LinRange(0.01, 3, 100), kind = :scale),
    ],
    loss = loss,
)

thompson_samples = run_scan(posterior, thompson_scan, model)
thompson_best = thompson_samples[thompson_samples.is_best, :]

CSV.write("thompson_samples.csv", thompson_best)


## count how often each scaler occurs
kx2_scaler_count = countmap(thompson_best.kx2_scaler)

eval_scan = CartesianScanner(
    simulation = sim_spec,
    scan = [(symbol = :kx2, values = collect(keys(kx2_scaler_count)), kind = :scale)],
    loss = loss,
)

eval_results = run_scan(posterior, eval_scan, model)

evaluation = combine(groupby(eval_results, :kx2_scaler)) do eval_group
    # extract all the losses across posterior for this kx2 scaler
    kx2_scaler = first(eval_group.kx2_scaler)
    losses = eval_group.loss
    thompson_count = kx2_scaler_count[kx2_scaler]
    expanded_losses = repeat(losses, thompson_count)

    return (
        median_loss = median(expanded_losses),
        q75_loss = quantile(expanded_losses, 0.75),
        std_loss = std(expanded_losses),
        thompson_count = thompson_count,
    )
end

sort!(evaluation, :kx2_scaler)
CSV.write("evaluation.csv", evaluation)

evaluation_plot = Plots.scatter(
    evaluation.median_loss,
    evaluation.q75_loss,
    xlabel = "Median Residue",
    ylabel = "Quantile Residue",
    title = "Evaluation of Thompson Samples",
    label = "Evaluation",
    color = "blue",
)

indices = findall((evaluation.median_loss .< 344) .& (evaluation.q75_loss .< 387))
good_values = evaluation.kx2_scaler[indices]

Plots.scatter!(
    evaluation_plot,
    evaluation.median_loss[indices],
    evaluation.q75_loss[indices],
    xlabel = "Median Residue",
    ylabel = "Quantile Residue",
    title = "Evaluation of Thompson Samples",
    label = "Good Values",
    color = "red",
)

CSV.write("indices.csv", DataFrame(index = indices))
CSV.write("medians.csv", DataFrame(median_loss = evaluation.median_loss))
CSV.write("quantiles.csv", DataFrame(q75_loss = evaluation.q75_loss))
CSV.write("good_values.csv", DataFrame(kx2_scaler = good_values))
Plots.savefig(evaluation_plot, "evaluation_scatter.png")
