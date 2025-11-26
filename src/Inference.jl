using Turing
using Distributions
using PreallocationTools
using SymbolicIndexingInterface

"""
    InferenceProblem

Wraps a Model and adds inference-specific functionality for Bayesian parameter estimation.

# Fields
- `model::Model`: The Model object to perform inference on
- `data::Union{Nothing, Vector{Float64}}`: Observed data
- `t_obs::Union{Nothing, Vector{Float64}}`: Time points of observations
- `obs_state_idx::Int`: Which state variable is observed (default: 1)
- `noise_prior::Distribution`: Prior for observation noise
- `chain::Union{Nothing, Any}`: MCMC chain results (stored after sampling)

# Example
```julia
model = Model(model_def, sys)
inf_prob = InferenceProblem(model)
```

"""

# -------------------------------------------------------------------------
# Struct Definitions
# -------------------------------------------------------------------------


mutable struct InferenceProblem

    model::Model # The model object to perform inference on
    data::Union{Nothing, Vector{Float64}} # Observed data
    t_obs::Union{Nothing, Vector{Float64}} # Time points of observations
    state_idx::Int # Which state variable is observed (defualt: 1)
    noise_prior::Distribution # Prior for observation noise
    
    # Results storage
    chain::Union{Nothing, Any}

    # Constructor
    function InferenceProblem(model::Model;
                             obs_state_idx::Int=1,
                             noise_prior::Distribution=InverseGamma(2, 3))
        # Defaults:
        # - obs_state_idx=1: Observe the first state variable by default
        # - InverseGamma(2,3): Standard weakly informative prior for noise variance
        
        new(model, nothing, nothing, obs_state_idx, noise_prior, nothing)
    end
end

# =========================================================================
# Query Methods
# =========================================================================

"""
    get_model(inf_prob::InferenceProblem) -> Model

Get the underlying Model object.

# Why this is useful:
You can still use all Model methods (simulate!, plot_state, etc.)
while working with InferenceProblem.

# Example
```julia
model = get_model(inf_prob)
simulate!(model, init_cond, params, tspan)
plot_all_states(model)
```
"""

function get_model(inf_prob::InferenceProblem)
    # Simply return the model - allows access to all Model methods
    return inf_prob.model
end