"""
    run_inference(model, spec)

Run an inference stage using the implementation selected by `spec`.

`model` should wrap a compiled ModelingToolkit system, and `spec` should be a
concrete `InferenceSpec` such as `TuringSpec`. The concrete method prepares the
model for simulation, repeatedly calls `simulate!` under the sampler, and
returns posterior samples in the format produced by that backend.

This fallback method exists to make unsupported inference specifications fail
with a clear error.
"""
function run_inference(model::Model, spec::InferenceSpec)
    error("No inference implementation for $(typeof(spec)).")
end

"""
    setup_model_for_inference(model, spec)

Prepare a model for the simulation settings in an inference spec.

Inference backends call this before sampling so the expensive ODE problem and
parameter setters are built once and then reused for each proposed posterior
draw.
"""
function setup_model_for_inference(model::Model, spec::InferenceSpec)
    return setup_model_for_simulation(model, spec.simulation)
end

"""
    setup_model_for_simulation(model, simulation)

Build the reusable simulation problem for a model.

This is a small bridge from spec-based workflows to `setup_simulation!`. It is
used by inference and scan stages whenever the `Model` has not already cached an
ODE problem for the requested `SimulationSpec`.
"""
function setup_model_for_simulation(model::Model, simulation::SimulationSpec)
    setup_simulation!(
        model,
        simulation.initial_conditions,
        simulation.tspan;
    )
    
    return nothing
end
