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


Random.seed!(0);

"""
Local testing script for the RPA model as in the paper and repo here: https://github.com/MichalKobiela/uncertainty-circ-opt/blob/main/RPARealData
"""

# Load model
RPA_model = load_model_from_yaml("./test/test-data/RPA_real_opt.yml")

# Compile the system once
@mtkcompile sys = System(RPA_model.equations, t)

model = Model(RPA_model, sys)

# Define simulation parameters
init_cond = [24.0, 350.0] # Initial values for y1 and y2

# Ground truth values (must match original)
params = Dict(
    :cuma => 2e-6,
)
  
tspan = (0.0, 10.0)
        
# Run simulation
sol = simulate!(model, init_cond, Dict(:cuma => 2e-6,), tspan)
sol = simulate!(model, sol[:,end], Dict(:cuma => 2e-5,), tspan)
sol = simulate!(model, sol[:,end], Dict(:cuma => 0.001,), tspan)

Plots.plot(sol)

# CSV.write(".//experiments//RPA_real_data//rpa_ode1.csv", Tables.table(sol.u))

# # Generate noisy observations
# t_obs = collect(range(1, stop = 90, length = 30)) 
# randomized = VectorOfArray([sol(t_obs[i])[1] + 1*randn() for i in eachindex(t_obs)])
# data = convert(Array, randomized)

 # Run inference
#  spec = BayesianSpec(
#       data = data,
#       t_obs = t_obs,
#       obs_state_idx = 1,
#       initial_conditions = [1.0, 1.0],
#       tspan = (0.0, 100.0),
#       uncertain_param_values = params,
#       noise_prior = InverseGamma(2,3),
#       sampler = NUTS(0.65),
#       n_samples = 1000,
#       n_chains = 3,
#       solver = Euler(),
#       dt = 0.01
#   )

#  @time chain = run_inference(model, spec)


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
