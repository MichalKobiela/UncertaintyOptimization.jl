"""
For each posterior sample, the script performs a 1D grid 
search over a scaling factor for one parameter, 
runs a warmup simulation, then evaluates how close 
the warmup state and final post-warmup 
prediction are to a target value.
"""
using Revise
using UncertaintyOptimization
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D;
using OrdinaryDiffEq
using CSV, Tables
using Turing
using SciMLBase: VectorOfArray
using SymbolicIndexingInterface
using Random
using PreallocationTools
using Serialization
using CSV, Tables
using Plots
using DataFrames
# using BenchmarkTools
using Profile
# using ProfileView
# using StatProfilerHTML
using StatsPlots
# using SciMLSensitivity


Random.seed!(0);

"""
This is the thompson sampling procedure. 
"""

# we still need the model
RPA_model = load_model("./test/test-data/RPA_real/opt.yml")
@mtkcompile sys = System(RPA_model.equations, t)
model = Model(RPA_model, sys)

# CSV.write(".//experiments//RPA_real_data//rpa_ode1.csv", Tables.table(sol.u))
time = CSV.read(string(@__DIR__)*"/RPA_real_data/time_points.csv", 
        DataFrame)[!,1]
background_fluorescence = 17.6


# load the precomputed posterior from a single chain for now
posterior = open(string(@__DIR__)*"/reference/rpareal_chain_reference.jls", "r") do io
        # convert into a DataFrame
        posterior_sample = deserialize(io)[1000:2000]
        DataFrame(posterior_sample)
end

# thompson sampling, grid search loss function
# for each posterior sample check what is the best kx2 scaling factor
# now these will be reserved keywords, 
# so warmup_sol and predicted_sol will contain the data necessary  
function loss(warmup_pars, predicted_sol)
    # define the loss function using the outputs from warmup and predicted

    # we previously only saved one state
    stateA = 1

    adjusted_predicted = predicted_sol[stateA, end] + background_fluorescence
    warmup = warmup_sol[stateA, end] + background_fluorescence
    
    target = 50
    (((warmup - target).^2) + (adjusted_predicted - target).^2) / 2
end

# so this is what will go into a grid spec
# and we will have to pass the loss funtion to it? 
# so parameters:
# - posterior
# - grid
# - loss function
thompson_samples = []
for row in eachrow(posterior)
    # the fixed values should come from the setup, 
    # whereas the tunables should be used for evaluation

    # optimize loss
    values = collect(LinRange(0.01, 3, 100))
    loss_post = x -> loss(row, x)
    losses = loss_post.(values)
    min_loss_index = argmin(losses)

    push!(thompson_samples, values[min_loss_index])
end

spec = GridScan(
    t_obs = time,
    # TODO - ideally this would not be operating on idx, but on names
    obs_state_idx = 1,
    initial_conditions = (24.0, 350.0),
    tspan = (0.0, 10.0),
    solver = Rosenbrock23(),
    solver_opts = (dtmin=1e-12, ),
)

run_

# chain2 = run_inference(model, spec)
# f = open(string(@__DIR__)*"/posterior_samples_large_range_1_chain2_c11_r1.jls", "w")
# serialize(f, chain2)
# close(f)

# Profile.clear()
# @profilehtml chain2 = run_inference(model, spec)

# f = open(string(@__DIR__)*"/posterior_samples_large_range_1_c2_r1.jls", "w")
# serialize(f, chain2)
# close(f)


# posterior_samples = sample(chain[[:beta_RA, :beta_BA, :beta_BB, :beta_AB]], 1000; replace=false)
# samples = Array(posterior_samples)

# -------------Compare to the previous data-----------------------------------

# # Load original posterior samples
# og_posterior_file = "./test/test-data/posterior_samples_og.csv"
# og_posterior_data = CSV.File(og_posterior_file; header=false) |> Tables.matrix

# # The original CSV has columns in THIS specific order
# column_order = [:beta_RA, :beta_BA, :beta_AB, :beta_BB]

# # Build dict mapping parameter name -> (min, max) from original
# param_ranges = Dict{Symbol, Tuple{Float64, Float64}}()

# for (i, name) in enumerate(column_order)
#      col_values = og_posterior_data[:, i]              # extract the column
#      min_val = minimum(col_values)
#      max_val = maximum(col_values)
#      param_ranges[name] = (min_val, max_val)
#      println("Column: $name  |  min: $min_val  max: $max_val")
#  end

#  param_ranges = Dict{Symbol, Tuple{Float64, Float64}}()

# for (i, name) in enumerate(column_order)
#      col_values = samples[:, i]              # extract the column
#      min_val = minimum(col_values)
#      max_val = maximum(col_values)
#      param_ranges[name] = (min_val, max_val)
#      println("Column: $name  |  min: $min_val  max: $max_val")
#  end

# CSV.write(".//experiments//RPA_real_data//posterior_samples.csv",  Tables.table(samples), writeheader=true)
