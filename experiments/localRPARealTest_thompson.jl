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
posterior_kx2 = posterior.kx2[thompson_best.iteration]

summary_best = DataFrame(
    posterior_index = thompson_best.iteration,
    posterior_kx2 = posterior_kx2,
    kx2_scaler = thompson_best.kx2_scaler,
    best_kx2 = thompson_best.kx2_value,
    best_loss = thompson_best.best_loss,
)

CSV.write("thompson_samples.csv", summary_best)


## evaluate now the unique kx2 scalers for each posterior
scaler_counts = combine(
    groupby(summary_best, :kx2_scaler),
    nrow => :thompson_count,
)
sort!(scaler_counts, :kx2_scaler)
scaler_counts.thompson_weight = scaler_counts.thompson_count ./ nrow(summary_best)

candidate_scalers = scaler_counts.kx2_scaler

eval_scan = CartesianScanner(
    simulation = sim_spec,
    scan = [(symbol = :kx2, values = candidate_scalers, kind = :scale)],
    loss = loss,
)

eval_results = run_scan(posterior, eval_scan, model)
count_by_scaler = Dict(scaler_counts.kx2_scaler .=> scaler_counts.thompson_count)
weight_by_scaler = Dict(scaler_counts.kx2_scaler .=> scaler_counts.thompson_weight)

evaluation = DataFrame(
    kx2_scaler = Float64[],
    median_loss = Float64[],
    q75_loss = Float64[],
    std_loss = Float64[],
    thompson_count = Int[],
    thompson_weight = Float64[],
)

for eval_group in groupby(eval_results, :kx2_scaler)
    # extract all the losses across posterior for this kx2 scaler
    kx2_scaler = first(eval_group.kx2_scaler)
    losses = eval_group.loss
    thompson_count = count_by_scaler[kx2_scaler]
    expanded_losses = repeat(losses, thompson_count)

    push!(evaluation, (
        kx2_scaler = kx2_scaler,
        median_loss = median(expanded_losses),
        q75_loss = quantile(expanded_losses, 0.75),
        std_loss = std(expanded_losses),
        thompson_count = thompson_count,
        thompson_weight = weight_by_scaler[kx2_scaler],
    ))
end

sort!(evaluation, :kx2_scaler)
CSV.write("evaluation.csv", evaluation)

expanded_indices = Int[]
for row_index in 1:nrow(evaluation)
    append!(expanded_indices, fill(row_index, evaluation.thompson_count[row_index]))
end
expanded_evaluation = evaluation[expanded_indices, :]
CSV.write("evaluation_expanded.csv", expanded_evaluation)


med_res = expanded_evaluation.median_loss
quant_res = expanded_evaluation.q75_loss
std_res = expanded_evaluation.std_loss

evaluation_plot = Plots.scatter(
    med_res,
    quant_res,
    xlabel = "Median Residue",
    ylabel = "Quantile Residue",
    title = "Evaluation of Thompson Samples",
    label = "Evaluation",
    color = "blue",
)

indices = findall((med_res .< 344) .& (quant_res .< 387))
good_values = expanded_evaluation.kx2_scaler[indices]
centroid = isempty(good_values) ? missing : mean(good_values)

Plots.scatter!(
    evaluation_plot,
    med_res[indices],
    quant_res[indices],
    xlabel = "Median Residue",
    ylabel = "Quantile Residue",
    title = "Evaluation of Thompson Samples",
    label = "Good Values",
    color = "red",
)

CSV.write("indices.csv", DataFrame(index = indices))
CSV.write("medians.csv", DataFrame(median_loss = med_res))
CSV.write("quantiles.csv", DataFrame(q75_loss = quant_res))
CSV.write("good_values.csv", DataFrame(kx2_scaler = good_values))
Plots.savefig(evaluation_plot, "evaluation_scatter.png")
