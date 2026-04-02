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


Random.seed!(0);

"""
Local testing script for the RPA model as in the paper and repo here: https://github.com/MichalKobiela/uncertainty-circ-opt/blob/main/RPARealData
"""

# Load model
RPA_model = load_model("./test/test-data/RPA_real/RPA_real_opt.yml")

# Compile the system once
@mtkcompile sys = System(RPA_model.equations, t)
# TODO - consider using @named with structural simplify for 100% certainty
model = Model(RPA_model, sys)

# Define simulation parameters
init_cond = (24.0, 350.0) # Initial values for y1 and y2
tspan = (0.0, 10.0)
        
# Run simulation
# sols = simulate!(model, init_cond, tspan)
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

# Run inference
# nuts = NUTS(0.65, init_ϵ = 0.001; adtype = AutoZygote())
nuts = NUTS(0.65, init_ϵ = 0.001)
spec = TuringSpec(
    data = data_subset,
    t_obs = time,
    obs_state_idx = 1,
    initial_conditions = (24.0, 350.0),
    tspan = (0.0, 10.0),
    # uncertain_param_values = params,
    noise_prior = InverseGamma(2,3),
    sampler = nuts, # MH()
    n_samples = 3,
    n_chains = 1,
    solver = Rosenbrock23(),
    solver_opts = (dtmin=1e-12, dense=false),
)


chain = run_inference(model, spec)

# Profile.init(; n=10^7, delay=0.001)
# Profile.clear_malloc_data()
# Profile.clear()
# @profile chain = run_inference(model, spec)
# data, lidict = Profile.retrieve()
# serialize("profile_results_1thread.jlprof", (data, lidict))


plot(chain)
f = open(string(@__DIR__)*"/posterior_try29.jls", "w")
serialize(f, chain)
close(f)

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
