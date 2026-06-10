using Distributions

"""
    InferenceSpec

This is the base specification for any inference algorithm.
This abstract type allows algorithm-specific specs. And makes it trivial to add new inference processes


Fields common to all algorithms

- `data::Vector{Float64}`: Observed data
- `simulation::SimulationSpec`: Shared simulation settings


"""

abstract type InferenceSpec end

# =========================================================================
# TURING SPEC: TURING-specific settings
# =========================================================================

"""
        TuringSpec
Uses all fields from InferenceSpec, plus:

- `noise_prior::Distribution`: Prior for observation noise
- `noise_initial::Float64`: Initial value for observation noise
- `sampler::Any`: Turing sampler (NUTS, HMC, etc.)
- `n_samples::Int`: Samples per chain
- `n_chains::Int`: Number of chains
- `sampling_method::Any`: Threading method

 Example
```julia
sim_spec = SimulationSpec(
    t_obs = times,
    obs_state = :X,
    initial_conditions = (1.0, 1.0),
    tspan = (0.0, 100.0),
)

spec = TuringSpec(
    simulation = sim_spec,
    data = observations,
    noise_prior = InverseGamma(2, 3),
    noise_initial = 3.0,
    sampler = NUTS(0.65),
    n_samples = 3000,
    n_chains = 1,
)
```
"""

struct TuringSpec <: InferenceSpec
    # Shared simulation settings
    simulation::SimulationSpec

    # Observed data
    data::Union{Vector{Float64}, Matrix{Float64}}

    # Bayesian-specific fields
    noise_prior::Distribution
    noise_initial::Float64
    sampler::Any
    n_samples::Int
    n_chains::Int
    sampling_method::Any
    
    # Constructor with defaults
    function TuringSpec(;
                      simulation::SimulationSpec,
                      data::Union{Vector{Float64}, Matrix{Float64}},
                      noise_prior::Distribution=InverseGamma(2, 3),
                      noise_initial::Real=3.0,
                      sampler=NUTS(0.65),
                      n_samples::Int=3000,
                      n_chains::Int=1,
                      sampling_method=MCMCSerial(),
                      )
        
        # Validation TODO
        # if size(data,1) != size(t_obs,1)
        #     throw(DimensionMismatch("❌ Data $(size(data)) and time $(size(t_obs)) vectors must have same length"))
        # end
        
        if n_samples < 1
            error("❌ n_samples must be positive")
        end

        if noise_initial <= 0
            error("❌ noise_initial must be positive")
        end

        new(simulation, data, noise_prior, Float64(noise_initial), sampler, n_samples, n_chains, sampling_method)
    end
end
