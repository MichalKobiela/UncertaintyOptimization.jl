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
Local testing script for the RPA model as in the paper and repo here: https://github.com/MichalKobiela/uncertainty-circ-opt/blob/main/RPARealData
"""

# Load model
RPA_model = load_model("./test/test-data/RPA_real/opt.yml")

# Compile the system once
@mtkcompile sys = System(RPA_model.equations, t)
# TODO - consider using @named with structural simplify for 100% certainty
model = Model(RPA_model, sys)

# Define simulation parameters
init_cond = (24.0, 350.0) # Initial values for y1 and y2
tspan = (0.0, 10.0)
        
# Run simulation
# sols = simulate!(model, init_cond, tspan; multiparam_length=3)
# for sol in sols
#     p = Plots.plot(sol)
#     display(p)
# end

# CSV.write(".//experiments//RPA_real_data//rpa_ode1.csv", Tables.table(sol.u))
time = CSV.read(string(@__DIR__)*"/RPA_real_data/time_points.csv", 
        DataFrame)[!,1]
data = Matrix(CSV.read(string(@__DIR__)*"/RPA_real_data/data.csv", 
        DataFrame))
background_fluorescence = 17.6
data = data .- background_fluorescence
# select specific modelled data
data_subset = vcat(data[:,2], data[:,5], data[:,9])

rosen_opts = (rtol=1e-5, atol=1e-7, maxiters=1_000_000)

# Run inference
# nuts = NUTS(0.65, init_ϵ = 0.001; adtype = AutoZygote())
nuts = NUTS(0.5, init_ϵ = 0.003)
spec = TuringSpec(
    data = data_subset,
    t_obs = time,
    obs_state_idx = 1,
    initial_conditions = (24.0, 350.0),
    tspan = (0.0, 10.0),
    # uncertain_param_values = params,
    noise_prior = InverseGamma(2,3),
    sampler = nuts, # MH()
    n_samples = 3000,
    n_chains = 1,
    # abstol and reltol? 
    solver = Rosenbrock23(), # AutoTsit5(Rosenbrock23()), # AutoTsit5(Rosenbrock23(autodiff=false)), 
    solver_opts = (dtmin=1e-12, ),
)


chain = run_inference(model, spec)

# Profile.init(; n=10^7, delay=0.001)
# Profile.clear_malloc_data()
# Profile.clear()
# @profile chain = run_inference(model, spec)
# data, lidict = Profile.retrieve()
# serialize("profile_results_1thread.jlprof", (data, lidict))


# plot(chain)
f = open(string(@__DIR__)*"/posterior_try67_nuts0.5_RB23_j12.1_noAutoDiffFalse.jls", "w")
serialize(f, chain)
close(f)

# sample 3 k samples

# thompson sampling, grid search loss function
# for each posterior sample check what is the best kx2 scaling factor
function loss(warmup_sol, predicted_sol)
    # define the loss function using the outputs from warmup and predicted
    adjusted_predicted = predicted_sol[stateA, end] + background_fluorescence
    warmup = warmup_sol[stateA, end]
    
    target = 50
    (((warmup - target).^2) + (adjusted_predicted - target).^2) / 2
end

# 
# run_design(model, chains, loss)


# EP: there must be a grid loss search function, there must be a optimisation problem (find the minimum)
# - global search (to cover both grid and gradient)
# EP: use a spec to also support user defined spec / function
# EP: pytorch, check how they abstracted optimisers, 
# Spec - spec inheritance? 



# function find_best 
#     for scaling_factor:
#         run_loss_function
#         # problem: you cannot use simulate function, unless you make it more complex and add returning the warmup in it. 
#         # also right now it runs 3 copies. But in the final test cuma is different, and we again
# end



# scan
# kx2 modifier, in the yaml, scan multipliers
# should we do this as part of the design? like in yaml: design: grid



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
