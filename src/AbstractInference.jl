using Distributions

"""
    InferenceSpec

Base type for inference specifications.

Concrete subtypes describe how posterior samples should be produced from a
`Model` and a `SimulationSpec`. Dispatch on this type lets `run_inference`
select an implementation without changing the model-loading or simulation
stages.
"""
abstract type InferenceSpec end

"""
    TuringSpec(; simulation, data, noise_prior=InverseGamma(2, 3),
               noise_initial=3.0, sampler=NUTS(0.65), n_samples=3000,
               n_chains=1, sampling_method=MCMCSerial())

Settings for Bayesian inference with Turing.

This spec represents the inference stage of the workflow. `simulation` defines
how the model is solved for each proposed parameter draw; `data` contains the
observations to compare against those solves; and the remaining fields configure
the observation noise model and Turing sampler.

`data` may be a vector or matrix, but it is flattened internally. Its length
must match the saved simulation layout:

```julia
length(t_obs) * observed_state_count(simulation) * number_of_production_solves
```

Uncertain parameters are discovered from YAML parameters with role
`:uncertain`. Their priors are read from the YAML `prior` metadata, while
`noise_prior` controls the observation noise `sigma`.
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
