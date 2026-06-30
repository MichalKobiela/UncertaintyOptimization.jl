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
using Serialization
using CSV, Tables
using Plots
using DataFrames

sampling_size = 1000
# debug
# sampling_size = 50

Random.seed!(0);

"""
Local testing script for the RPA model as in the paper and repo here: https://github.com/MichalKobiela/uncertainty-circ-opt/blob/main/RPA/Inference/mcmc.jl

    
    In the original inference procedure for the RPA model it performs Turing parameter inference for an ODE model. 
        
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

# Load model
RPA_model = load_model(normpath(joinpath(@__DIR__, "..", "test", "test-data", "test_RPA.yml")))

# Compile the system once
@mtkcompile sys = System(RPA_model.equations, t)

model = Model(RPA_model, sys)

# Define simulation parameters
init_cond = (1.0, 1.0)
        
# Ground truth values (must match original)
params = Dict(
    :beta_RA => 0.1,
    :beta_AB => 0.001,
    :beta_BA => 0.01,
    :beta_BB => 0.001
)
  
tspan = (0.0, 100.0)
        
# Run simulation
sol = only(simulate!(model, init_cond, params, tspan))
sol_df = DataFrame(permutedims(Array(sol)), [:A, :B])
insertcols!(sol_df, 1, :t => sol.t)
CSV.write("rpa_sol_true.csv", sol_df)

# Generate noisy observations
t_obs = collect(range(1, stop = 90, length = 30)) 
randomized = VectorOfArray([sol(t_obs[i])[1] + 1*randn() for i in eachindex(t_obs)])
data = convert(Array, randomized)

# Run inference
sim_spec = SimulationSpec(
    t_obs = t_obs,
    obs_state = :A,
    initial_conditions = (1.0, 1.0),
    tspan = (0.0, 100.0),
    uncertain_param_values = params,
    solver = Euler(),
    solver_opts = (dt = 0.01,),
)

spec = TuringSpec(
    simulation = sim_spec,
    data = data,
    noise_prior = InverseGamma(2,3), 
    noise_initial = 3.0, 
    sampler = NUTS(0.5),
    n_samples = sampling_size,
    n_chains = 3,
)

@time chain = run_inference(model, spec)

betas = [:beta_RA, :beta_BA, :beta_BB, :beta_AB]
posterior_samples = sample(chain[betas], sampling_size; replace=false)
samples = Array(posterior_samples)


# Test if posterior draws distribution recovers the ground truth values
function ci_contains_truth(chain, p::Symbol, truth::Real; level=0.95)
    α = 1 - level
    posterior_draws = vec(Array(chain[p]))                 
    lo, hi = quantile(posterior_draws, (α/2, 1-α/2))        
    μ = mean(posterior_draws)
    return (param=p, mean=μ, lo=lo, hi=hi, truth=Float64(truth), in_CI=(lo <= truth <= hi))
end

truths = [params[beta] for beta in betas]
rows = ci_contains_truth.(Ref(chain), betas, truths; level=0.95)

for row in rows
    println("Result: $(row)")
end

CSV.write(".//experiments//RPA_data//posterior_samples.csv",  Tables.table(samples), writeheader=true)
