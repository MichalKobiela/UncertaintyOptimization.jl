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

RPA_model = load_model_from_yaml("./test/test-data/test_RPA.yml")

# Compile the system once
@mtkcompile sys = System(RPA_model.equations, t)

model = Model(RPA_model, sys)

# Define simulation parameters
init_cond = [1.0, 1.0]  # Initial conditions for [A, B]
        
# Parameters to simulate with (ground truth values)
params = Dict(
        :beta_RA => 0.1,
        :beta_AB => 0.001,
        :beta_BA => 0.01,
        :beta_BB => 0.001
    )
        
tspan = (0.0, 100.0)  # Simulate from t=0 to t=100
        
# Run simulation
sol = simulate!(model, init_cond, params, tspan)

t_obs = collect(range(1, stop = 90, length = 30))
randomized = VectorOfArray([sol(t_obs[i])[1] + 1*randn() for i in eachindex(t_obs)])
data = convert(Array, randomized)

spec = BayesianSpec(
    data = data,
    t_obs = collect(t_obs),
    initial_conditions = [1.0, 1.0],
    tspan = (0.0, 100.0),
    noise_prior = InverseGamma(2,3),
    sampler = NUTS(0.65),
    n_samples = 1000,
    n_chains = 3,
    solver = Euler(),
    dt = 0.01
)

chain = run_inference(model, spec)

# --- Save solution to CSV ---
#CSV.write(".//experiments//RPA_data//rpa_sol_true.csv", Tables.table(sol.u), writeheader=false)

# --- Plot the solution ---
#plot(sol, xlabel="Time", ylabel="States", title="Simulation Results")
#savefig("./test/test-plots/simulation_plot.png")


# # --- Make some noisy observations and save---
# t_obs = collect(range(1, stop = 90, length = 30))
# randomized = VectorOfArray([sol(t_obs[i])[1] + 1*randn() for i in eachindex(t_obs)])
# data = convert(Array, randomized)
# CSV.write(".//experiments//RPA_data//rpa_data_true.csv", Tables.table(data), writeheader=false)

# # --- Create an uncertain dictionary - could take this from the Model Definition later
# uncertain_syms = [
#     sys.beta_RA,
#     sys.beta_AB,
#     sys.beta_BA,
#     sys.beta_BB]

# ]

# # Create a lazy cache that remakes only buffers, not the entire problem
# lbc_func = (p) -> remake_buffer(sys, rpa_prob.p, Dict(zip(uncertain_syms, p)))
# # Create the parameter setter
# setter_p! = setp(sys, uncertain_syms)

# @model function fit(data, prob, lbc_func, setter_p!)

#     σ ~ InverseGamma(2, 3)
#     beta_RA ~ truncated(Uniform(0.0, 1.0), lower=0.0)
#     beta_AB ~ truncated(Uniform(0.0, 1.0), lower=0.0)
#     beta_BA ~ truncated(Uniform(0.0, 1.0), lower=0.0)
#     beta_BB ~ truncated(Uniform(0.0, 1.0), lower=0.0)

#     # Combine parameters
#     p_vec = [beta_RA, beta_AB, beta_BA, beta_BB]

#     # Create fast buffer for these parameter values
#     new_p = lbc_func(p_vec)
#     setter_p!(new_p, p_vec)
#     prob_tmp = remake(prob; p=new_p)

#     # Change to array because we are working with the ODESystem
#     predicted = Array(solve(prob_tmp, Euler(); dt=0.01, saveat=t_obs, save_idxs=1))
#     data ~ MvNormal(predicted, σ^2 * I(length(data)))

#     return nothing
# end

# model2 = fit(data, rpa_prob, lbc_func, setter_p!)

# @time chain = sample(model2, NUTS(0.65), MCMCThreads(), 1000, 3; progress=false)


# posterior_samples = sample(chain[[:beta_RA, :beta_AB, :beta_BA, :beta_BB]], 1000; replace=false)
# samples = Array(posterior_samples)

# f = open(".//experiments//RPA_data//posterior_chains.jls", "w")
# serialize(f, chain)
# close(f)

# f = open(".//experiments//RPA_data//posterior_chains.jls", "r")
# chain = deserialize(f)
# close(f)

# f = open(".//experiments//RPA_data//posterior_samples.jls", "w")
# serialize(f, samples)
# close(f)

# CSV.write(".//experiments//RPA_data//posterior_samples.csv",  Tables.table(samples), writeheader=false)