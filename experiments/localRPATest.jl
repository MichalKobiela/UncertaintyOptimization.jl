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
Local testing script for the RPA model as in the paper and repo here: https://github.com/MichalKobiela/uncertainty-circ-opt/blob/main/RPA/Inference/mcmc.jl

    
    In the original inference procedure for the RPA model it performs Bayesian parameter inference for an ODE model. 
        
    First, it takes some initial conditions, ground truth parameters and an ODEProblem to solve the system and compute 
    some trajectories saving them to sol_true.csv. 
    
    Next, the script then creates noisy observations to simulate experimental data with meausurement noise and saves to data_true.csv. 
    
    A Bayesian hierarchical model is then defined with priors for a subset of parameters that they want to treat as uncertain. 
    Monte Carlo sampling is run to create posterior samples. 
    
    In the subsequent script then randomly selects 1,000 samples from the posterior and saves to posterior_samples.csv.
    
    The posterior samples can then be used later for design.

    The difference in this testing script is the use of the ModelLoader module. The aim is to see if a model defined in this way can
    be used to generated similar results to the original paper.

"""
function print_sys_parameters(sys)
    pars = parameters(sys)  # get all parameters in the system
    param_dict = Dict{Symbol, Float64}()

    println("Model parameters:")
    for p in pars
        val = getproperty(sys, p.name)
        param_dict[p.name] = Float64(val)  # ensure numeric
        println("  $(p.name) = $(val)")
    end

    return param_dict  # also return as a Dict if needed
end


# Load model
RPA_model = load_model_from_yaml("./test/test-data/test_RPA.yml")

# Compile the system once
@mtkcompile sys = System(RPA_model.equations, t)

model = Model(RPA_model, sys)

# Define simulation parameters
init_cond = [1.0, 1.0]
        
# Ground truth values (must match original)
params = Dict(
    :beta_RA => 0.1,
    :beta_AB => 0.001,
    :beta_BA => 0.01,
    :beta_BB => 0.001
)
  
tspan = (0.0, 100.0)  # Change to 100.0 for full validation
        
# Run simulation
sol = simulate!(model, init_cond, params, tspan)

CSV.write(".//experiments//RPA_data//rpa_sol_true.csv", Tables.table(sol.u))

# Generate noisy observations
t_obs = collect(range(1, stop = 90, length = 30))  # Change to range(1, 90, 30) for full validation
randomized = VectorOfArray([sol(t_obs[i])[1] + 1*randn() for i in eachindex(t_obs)])
data = convert(Array, randomized)

setup_simulation!(model,
                  t_obs,
                  1,
                  init_cond,
                  params,
                  tspan;
                  solver=Euler(),
                  dt=0.01)

lbc_func = model.buffer_func
setter_p! = setp(model.sys, model.uncertain_params)

@model function fit(data, prob, lbc_func, setter_p!)

    σ ~ InverseGamma(2, 3)
    beta_RA ~ truncated(Uniform(0.0, 1.0), lower=0.0)
    beta_AB ~ truncated(Uniform(0.0, 1.0), lower=0.0)
    beta_BA ~ truncated(Uniform(0.0, 1.0), lower=0.0)
    beta_BB ~ truncated(Uniform(0.0, 1.0), lower=0.0)

    # Combine parameters
    p_vec = [beta_RA, beta_AB, beta_BA, beta_BB]

    # Create fast buffer for these parameter values
    new_p = lbc_func(p_vec)
    setter_p!(new_p, p_vec)
    prob_tmp = remake(prob; p=new_p)

    # Change to array because we are working with the ODESystem
    predicted = Array(solve(prob_tmp, Euler(); dt=0.01, saveat=t_obs, save_idxs=1))
    data ~ MvNormal(predicted, σ^2 * I(length(data)))

    return nothing
end

model2 = fit(data, model.prob, lbc_func, setter_p!)

@time chain = sample(model2, NUTS(0.65), MCMCThreads(), 1000, 3; progress=true)

posterior_samples = sample(chain[[:beta_RA, :beta_AB, :beta_BA, :beta_BB]], 1000; replace=false)
samples = Array(posterior_samples)
# -------------Compare to the previous data-----------------------------------

# Load original posterior samples
og_posterior_file = "./test/test-data/posterior_samples_og.csv"
og_posterior_data = CSV.File(og_posterior_file; header=false) |> Tables.matrix

# The original CSV has columns in THIS specific order
column_order = [:beta_RA, :beta_BA, :beta_AB, :beta_BB]

# Build dict mapping parameter name -> (min, max) from original
param_ranges = Dict{Symbol, Tuple{Float64, Float64}}()

for (i, name) in enumerate(column_order)
     col_values = og_posterior_data[:, i]              # extract the column
     min_val = minimum(col_values)
     max_val = maximum(col_values)
     param_ranges[name] = (min_val, max_val)
     println("Column: $name  |  min: $min_val  max: $max_val")
 end

 param_ranges = Dict{Symbol, Tuple{Float64, Float64}}()

for (i, name) in enumerate(column_order)
     col_values = samples[:, i]              # extract the column
     min_val = minimum(col_values)
     max_val = maximum(col_values)
     param_ranges[name] = (min_val, max_val)
     println("Column: $name  |  min: $min_val  max: $max_val")
 end


# # Run inference
# spec = BayesianSpec(
#      data = data,
#      t_obs = t_obs,
#      obs_state_idx = 1,
#      initial_conditions = [1.0, 1.0],
#      tspan = (0.0, 2.0),  # Change to 100.0 for full validation
#      uncertain_param_values = params,
#      noise_prior = InverseGamma(2,3),
#      sampler = NUTS(0.65),
#      n_samples = 1000,
#      n_chains = 3,
#      solver = Euler(),
#      dt = 0.01
#  )

# @time chain = run_inference(model, spec)

# println("Chain variable names: ", names(chain))

# # Convert chain to DataFrame
# posterior_df = DataFrame(chain)

# # Clean up column names
# #rename!(posterior_df, [name => Symbol(replace(string(name), r"^:" => "")) for name in names(posterior_df)])

# #cols_of_interest = [:("p_vec[1]"), :("p_vec[2]"), :("p_vec[3]"), :("p_vec[4]")]
# cols_of_interest = [:beta_RA, :beta_BA, :beta_BB, :beta_AB]
# param_ranges = Dict{Symbol, Tuple{Float64, Float64}}()

# for name in cols_of_interest
#     col_values = posterior_df[!, name]   # extract column by name
#     param_ranges[name] = (minimum(col_values), maximum(col_values))
#     println("Column: $name | min: $(minimum(col_values)) max: $(maximum(col_values))")
# end

# param_ranges = Dict{Symbol, Tuple{Float64, Float64}}()

# for (i, name) in enumerate(their_csv_order)
#     col_values = og_posterior_data[:, i]              # extract the column
#     min_val = minimum(col_values)
#     max_val = maximum(col_values)
#     param_ranges[name] = (min_val, max_val)
#     println("Column: $name  |  min: $min_val  max: $max_val")
# end

# #println("Cleaned column names: ", names(posterior_df))

# # Filter to uncertain parameters
# #uncertain_param_cols = filter(name -> Symbol(name) in model.uncertain_params, names(posterior_df))

# #println("\nPosterior Statistics:")
# #println("="^60)
# #println("Note: Comparing by parameter NAME, not column position")
# #println("Your sampling order: ", model.uncertain_params)
# #println("Original CSV order: ", their_csv_order)
# #println("="^60)

# # # Extract ranges for YOUR posteriors (by name)
# # posterior_ranges = Dict{Symbol, Tuple{Float64, Float64}}()
# # for name in uncertain_param_cols
# #     col = posterior_df[!, Symbol(name)]
# #     posterior_ranges[Symbol(name)] = (minimum(col), maximum(col))
# # end

# # println(posterior_ranges)

# # # Compare by parameter NAME (order-independent)
# # for name in their_csv_order  # Iterate in their order for consistent display
# #     if haskey(param_ranges, name) && haskey(posterior_ranges, name)
# #         og_min, og_max = param_ranges[name]
# #         post_min, post_max = posterior_ranges[name]
        
# #         println("\nParameter: $name")
# #         println("  Original range:   [$og_min, $og_max]")
# #         println("  Posterior range:  [$post_min, $post_max]")
# #         println("  Width comparison: $(og_max - og_min) vs $(post_max - post_min)")
        
#         # Add a match quality indicator
#         ratio = (post_max - post_min) / (og_max - og_min)
#         if 0.7 < ratio < 1.3
#             println("  ✅ Good match (within 30%)")
#         elseif ratio > 2.0 || ratio < 0.5
#             println("  ⚠️  Poor match (>2x difference)")
#         else
#             println("  ⚡ Moderate match")
#         end
#     else
#         println("\n⚠️  Parameter $name not found in both datasets")
#     end
# end

# # Save outputs - order doesn't matter here, names are preserved
# samples = Matrix(posterior_df[!, uncertain_param_cols])

# serialize(".//experiments//RPA_data//posterior_chains_new.jls", chain)
# serialize(".//experiments//RPA_data//posterior_samples_new.jls", samples)

# CSV.write(
#     ".//experiments//RPA_data//posterior_samples.csv",
#     DataFrame(samples, Symbol.(uncertain_param_cols)),
#     writeheader=true
# )

# #println("\n" * "="^60)
# #println("✅ Inference complete. Results saved.")
# #println("="^60)

#posterior_samples = sample(chain[[:("p_vec[1]"), :("p_vec[2]"), :("p_vec[3]"), :("p_vec[4]")]], 1000; replace=false)
#samples = Array(posterior_samples)

#println(posterior_samples)

# f = open(".//experiments//RPA_data//posterior_chains.jls", "w")
# serialize(f, chain)
# close(f)

# f = open(".//experiments//RPA_data//posterior_chains.jls", "r")
# chain = deserialize(f)
# close(f)

# f = open(".//experiments//RPA_data//posterior_samples.jls", "w")
# serialize(f, samples)
# close(f)

CSV.write(".//experiments//RPA_data//posterior_samples.csv",  Tables.table(samples), writeheader=false)