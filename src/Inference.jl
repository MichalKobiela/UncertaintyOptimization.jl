
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
    return setup_model_for_simulation(model, spec.simulation)
end

function setup_model_for_simulation(model::Model, simulation::SimulationSpec)
    setup_simulation!(
        model,
        simulation.initial_conditions,
        simulation.tspan;
    )
    
    return nothing
end
