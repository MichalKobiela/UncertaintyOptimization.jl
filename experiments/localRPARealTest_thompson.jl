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
thompson_scan = CartesianSampler(
    simulation = sim_spec,
    scan = [
        (symbol = :kx2, values = LinRange(0.01, 3, 100), kind = :scale),
    ],
    loss = loss,
)

thompson_samples = run_scan(posterior, thompson_scan, model)

summary_best = DataFrame(
    posterior_index = [r.sample_index for r in thompson_samples],
    posterior_kx2 = [r.sample.kx2 for r in thompson_samples],
    kx2_scaler = [only(r.best_values).value for r in thompson_samples],
    best_kx2 = [r.sample.kx2 * only(r.best_values).value for r in thompson_samples],
    best_loss = [r.best_loss for r in thompson_samples],
)

CSV.write("thompson_samples.csv", summary_best)


## evaluate now the kx2 values for each posterior
candidate_kx2 = sort(unique(summary_best.best_kx2))

eval_scan = CartesianSampler(
    simulation = sim_spec,
    scan = [(symbol = :kx2, values = candidate_kx2, kind = :value)],
    loss = loss,
)

eval_results = run_scan(posterior, eval_scan, model)

evaluation = DataFrame(
    kx2 = candidate_kx2,
    median_loss = Float64[],
    q75_loss = Float64[],
    std_loss = Float64[],
)

for (i, kx2) in enumerate(candidate_kx2)
    # extract all the losses across posterior for this kx2 
    losses = [r.losses[i] for r in eval_results]
    push!(evaluation, (
        kx2 = kx2,
        median_loss = median(losses),
        q75_loss = quantile(losses, 0.75),
        std_loss = std(losses),
    ))
end

CSV.write("evaluation.csv", evaluation)

med_res = evaluation.median_loss
quant_res = evaluation.q75_loss
std_res = evaluation.std_loss

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
good_values = evaluation.kx2[indices]
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
CSV.write("good_values.csv", DataFrame(kx2 = good_values))
Plots.savefig(evaluation_plot, "evaluation_scatter.png")
