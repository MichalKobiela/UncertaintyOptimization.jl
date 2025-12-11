
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
        spec.uncertain_param_values,          
        spec.tspan;                 
        solver = spec.solver,       
        dt= spec.dt            
    )
    
    return nothing
end
