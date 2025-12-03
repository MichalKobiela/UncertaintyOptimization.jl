using Distributions

"""
    InferenceSpec

This is the base specification for any inference algorithm.
This abstract type allows algorithm-specific specs. And makes it trivial to add new inference processes


Fields common to all algorithms

- `data::Vector{Float64}`: Observed data
- `t_obs::Vector{Float64}`: Observation times
- `obs_state_idx::Int`: Which state is observed
- `initial_conditions::Vector{Float64}`: Starting state
- `tspan::Tuple{Float64,Float64}`: Time span
- `fixed_params::Dict`: Additional fixed parameters


"""

abstract type InferenceSpec end

# =========================================================================
# BAYESIAN SPEC: BAYESIAN-specific settings
# =========================================================================

"""
        BayesianSpec
Uses all fields from InferenceSpec, plus:

- `noise_prior::Distribution`: Prior for observation noise
- `sampler::Any`: Turing sampler (NUTS, HMC, etc.)
- `n_samples::Int`: Samples per chain
- `n_chains::Int`: Number of chains
- `sampling_method::Any`: Threading method
- `solver`: solver
- `dt::Float64`: Time step

 Example
```julia
spec = BayesianSpec(
    data = observations,
    t_obs = times,
    obs_state_idx = 1,
    initial_conditions = [1.0, 1.0],
    tspan = (0.0, 100.0),
    noise_prior = InverseGamma(2, 3),
    sampler = NUTS(0.65),
    n_samples = 1000,
    n_chains = 3
)
```
"""

struct BayesianSpec <: InferenceSpec
    # Common fields
    data::Vector{Float64}
    t_obs::Vector{Float64}
    obs_state_idx::Int
    initial_conditions::Vector{Float64}
    tspan::Tuple{Float64, Float64}
    uncertain_param_values::Dict

    # Bayesian-specific fields
    noise_prior::Distribution
    sampler::Any
    n_samples::Int
    n_chains::Int
    sampling_method::Any
    solver::Any
    dt::Float64
    
    # Constructor with defaults
    function BayesianSpec(;
                      data::Vector{Float64},
                      t_obs::Vector{Float64},
                      obs_state_idx::Int=1,
                      initial_conditions::Vector{Float64},
                      tspan::Tuple{Float64, Float64},
                      uncertain_param_values::Dict=Dict(),
                      noise_prior::Distribution=InverseGamma(2, 3),
                      sampler=NUTS(0.65),
                      n_samples::Int=1000,
                      n_chains::Int=3,
                      sampling_method=MCMCThreads(),
                      solver=Euler(),
                      dt::Float64=0.01)
        
        # Validation
        if length(data) != length(t_obs)
            error("❌ Data and time vectors must have same length")
        end
        
        if n_samples < 1
            error("❌ n_samples must be positive")
        end
        
        new(data, t_obs, obs_state_idx, initial_conditions, tspan,
            uncertain_param_values, noise_prior, sampler, n_samples, n_chains,
            sampling_method, solver, dt)
    end
end