
"""
    Adaptor for the Inference process.

The spec provided to the run_inference function determines which method is called.

Each file in algorithms will have a separate inference algorithm making adding new observations
trivial.

Only need to call run_inference(model::Model, spec::AbstractInference)

Common functions for all inference procedures go here:

1. Set up a simulation
2. Solve for uncertain params

"""
# Fallback function
function run_inference(model::Model, spec::InferenceSpec)
    error("No inference implementation for $(typeof(spec)).")
end

function setup_model_for_inference(model::Model, spec::InferenceSpec)
    # Setup model simulation
     setup_simulation!(
        model,
        spec.t_obs,                 
        spec.obs_state_idx,         
        spec.initial_conditions,    
        spec.fixed_params,          
        spec.tspan;                 
        solver = spec.solver,       
        dt     = spec.dt            
    )
    
    return nothing
end

# """
#     InferenceProblem

# Wraps a Model and adds inference-specific functionality.

# # Fields
# - `model::Model`: The Model object to perform inference on
# - `data::Union{Nothing, Vector{Float64}}`: Observed data
# - `t_obs::Union{Nothing, Vector{Float64}}`: Time points of observations
# - `obs_state_idx::Int`: Which state variable is observed (default: 1)
# - `noise_prior::Distribution`: Prior for observation noise
# - `chain::Union{Nothing, Any}`: chain results (stored after sampling)

# # Example
# ```julia
# model = Model(model_def, sys)
# inf_prob = InferenceProblem(model)
# ```

# """

# # -------------------------------------------------------------------------
# # Struct Definitions
# # -------------------------------------------------------------------------


# mutable struct InferenceProblem

#     model::Model # The model object to perform inference on
#     data::Union{Nothing, Vector{Float64}} # Observed data
#     t_obs::Union{Nothing, Vector{Float64}} # Time points of observations
#     state_idx::Int # Which state variable is observed (defualt: 1)
#     noise_prior::Distribution # Prior for observation noise
    
#     # Results storage
#     chain::Union{Nothing, Any}

#     # Constructor
#     function InferenceProblem(model::Model;
#                              obs_state_idx::Int=1,
#                              noise_prior::Distribution=InverseGamma(2, 3))
#         # Defaults:
#         # - obs_state_idx=1: Observe the first state variable by default
#         # - InverseGamma(2,3): Standard weakly informative prior for noise variance
        
#         new(model, nothing, nothing, obs_state_idx, noise_prior, nothing)
#     end
# end

# # =========================================================================
# # Query Methods
# # =========================================================================

# """
#     get_model(inf_prob::InferenceProblem) -> Model

# Get the underlying Model object.

# # Example
# ```julia
# model = get_model(inf_prob)
# simulate!(model, init_cond, params, tspan)
# plot_all_states(model)
# ```
# """

# function get_model(inf_prob::InferenceProblem)
#     # Just so we can still use all Model methods (simulate!, plot_state, etc.) while working with InferenceProblem.
#     return inf_prob.model
# end

# # =========================================================================
# # Data Methods
# # =========================================================================

# """
#     set_data!(inf_prob::InferenceProblem, 
#               data::Vector{Float64}, 
#               t_obs::Vector{Float64})

# Set the observation data for inference.

# Stores the observed data and corresponding time points in the InferenceProblem.

# # Arguments
# - `inf_prob`: The InferenceProblem to update
# - `data`: Vector of observed values
# - `t_obs`: Vector of time points when observations were made

# # Validation:
# - Checks that data and time vectors have the same length
# - Checks that time points are within reasonable range

# # Example
# ```julia
# # Generate synthetic observations
# t_obs = [0.0, 1.0, 2.0, 5.0, 10.0]
# data = [1.2, 1.5, 1.8, 2.1, 2.3]

# set_data!(inf_prob, data, t_obs)
# ```
# """

# function set_data!(inf_prob::InferenceProblem,
#                     data::Vector{Float64},
#                     t_obs::Vector{Float64})

#     # Do some checks first
#     if length(data) != length(t_obs)
#         error("❌ Data and time vectors must have the same length. " *
#               "Got data: $(length(data)), times: $(length(t_obs))")
#     end
#     if length(data) == 0
#         error("❌ Cannot set empty data. Need at least one observation.")
#     end
#     if any(!isfinite, data)
#         error("❌ Data contains NaN or Inf values")
#     end

#     inf_prob.data = data
#     inf_prob.t_obs = t_obs
    
# end