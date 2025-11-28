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

@time chain = run_inference(model, spec)


function extract_uncertain_posteriors(chain::Chains; n_samples::Int=1000, rng::AbstractRNG=Random.GLOBAL_RNG)
    names_in_chain = names(chain)
    uncertain_params = filter(n -> occursin("uncertain_param", string(n)), names_in_chain)
    sampled_chain = sample(chain[uncertain_params], n_samples, replace=false)
    samples_array = Array(sampled_chain)
    clean_names = Symbol.(replace.(string.(uncertain_params), r"uncertain_param\[:(.+)\]" => s"\1"))
    return DataFrame(samples_array, Symbol.(clean_names))
end

posterior_df = extract_uncertain_posteriors(chain; n_samples=1000)  # returns DataFrame
samples = Array(posterior_df)  # convert to plain array if desired

# --- Save the full chain ---
f = open(".//experiments//RPA_data//posterior_chains_new.jls", "w")
serialize(f, chain)
close(f)

# --- Load the chain back (if needed) ---
f = open(".//experiments//RPA_data//posterior_chains_new.jls", "r")
chain = deserialize(f)
close(f)

# --- Save posterior samples separately ---
f = open(".//experiments//RPA_data//posterior_samples_new.jls", "w")
serialize(f, samples)
close(f)

CSV.write(".//experiments//RPA_data//posterior_samples_new.csv",  Tables.table(samples), writeheader=false)