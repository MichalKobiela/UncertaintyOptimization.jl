"""
    run_inference(model, spec)

Run inference using the implementation selected by `spec`.
"""
function run_inference(model::Model, spec::InferenceSpec)
    error("No inference implementation for $(typeof(spec)).")
end

"""
    setup_model_for_inference(model, spec)

Prepare a model for the simulation settings in an inference spec.
"""
function setup_model_for_inference(model::Model, spec::InferenceSpec)
    return setup_model_for_simulation(model, spec.simulation)
end

"""
    setup_model_for_simulation(model, simulation)

Build the reusable simulation problem for a model.
"""
function setup_model_for_simulation(model::Model, simulation::SimulationSpec)
    setup_simulation!(
        model,
        simulation.initial_conditions,
        simulation.tspan;
    )
    
    return nothing
end
