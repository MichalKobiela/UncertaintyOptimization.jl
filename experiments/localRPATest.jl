using Revise
using UncertaintyOptimization
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D;
using OrdinaryDiffEq
using CSV, Tables
using Random
Random.seed!(0);
using SciMLBase: VectorOfArray

"""
Local testing script for the RPA model as in the paper and repo here: https://github.com/MichalKobiela/uncertainty-circ-opt/blob/main/RPA/Inference/mcmc.jl

    
    In the original inference procedure for the RPA model it performs Bayesian parameter inference for an ODE model. 
        
    First, it takes some initial conditions, ground truth parameters and an ODEProblem to solve the system and compute 
    some trajectories saving them to sol_true.csv. 
    
    Next, the script then creates noisy observations to simulate experimental data with meausurement noise and saves to data_true.csv. 
    
    A Bayesian hierarchical model is then defined with priors for a subset of parameters that they want to treat as uncertain. 
    Monte Carlo sampling is run to create posterior samples. 
    
    In the subsequent script then randomly selects 1,000 samples from the posterior and saves to posterior_samples.csv.
    
    The main idea is to create some simulation of noisy experimental data and see if the inference method can correctly estimate the
    true parameter values. Because each parameter is a posterior distrbution not just a single estimate the method also quantifies 
    uncertainty in the inferred parameters.
    
    The posterior samples can then be used later for design.

    The difference in this testing script is the use of the ModelLoader module. The aim is to see if a model defined in this way can
    be used to generated similar results to the original paper.

"""


RPA_model = load_model_from_yaml("./test/test-data/test_RPA.yml")

# Compile the system once
@mtkcompile sys = System(RPA_model.equations, t)

# --- Initial conditions ---
init_cond = [1.0, 1.0]

# --- Ground-truth parameters ---
ground_truth = Dict(
    :beta_RA => 0.1,
    :beta_AB => 0.001,
    :beta_BA => 0.01,
    :beta_BB => 0.001
)

# --- Create mapping of unknowns to initial conditions ---
u_map = Dict(unknowns(sys) .=> init_cond)

# --- Merge with all known parameters ---
p_all = Dict(p.symbol => p.value for p in values(RPA_model.parameters) if p.value !== nothing)
p_map = merge(u_map, p_all, ground_truth)

# --- Time span ---
tspan = (0.0, 100.0)

# --- Create the ODEProblem ---
rpa_prob = ODEProblem(sys, p_map, tspan)

# --- Solve ---
sol = solve(rpa_prob, Euler(), dt = 0.01)

# --- Save solution to CSV ---
CSV.write(".//experiments//RPA_data//rpa_sol_true.csv", Tables.table(sol.u), writeheader=false)


# --- Make some noisy operations ---
t_obs = collect(range(1, stop = 90, length = 30))
randomized = VectorOfArray([sol(t_obs[i])[1] + 1*randn() for i in eachindex(t_obs)])
data = convert(Array, randomized)

CSV.write(".//experiments//RPA_data//rpa_data_true.csv", Tables.table(data), writeheader=false)