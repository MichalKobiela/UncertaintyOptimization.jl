using Distributions

"""
    InferenceSpec

Base type for inference specifications.
"""
abstract type InferenceSpec end

"""
    TuringSpec(; simulation, data, noise_prior, noise_initial, sampler, n_samples, n_chains)

Settings for Bayesian inference with Turing.
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
