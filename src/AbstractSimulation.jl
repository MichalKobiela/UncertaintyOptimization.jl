"""
    SimulationSpec(; t_obs, obs_state, initial_conditions, tspan,
                   uncertain_param_values=Dict(), solver=Euler(),
                   solver_opts=NamedTuple())

Shared settings for the simulation/build stage.

`SimulationSpec` captures the parts of a run that are not part of the YAML model
definition: observation times, observed state or states, initial conditions, time
spans, solver, and solver keyword options. The same spec can be reused by
`TuringSpec` for inference and by `ThompsonGridSpec`/`CartesianScanner` for
Thompson-style scans and later evaluation.

Fields:

- `t_obs`: time points saved for observations and loss evaluation.
- `obs_state`: one state name or a collection of state names to observe.
- `initial_conditions`: initial values in the compiled system's unknown order.
- `tspan`: integration intervals. A single `(start, stop)` is normalized to
  `((start, stop), (start, stop))`. Use `((warmup_start, warmup_stop),
  (production_start, production_stop))` when warmup and production solves should
  use different intervals.
- `uncertain_param_values`: optional values for uncertain parameters.
- `solver`: OrdinaryDiffEq-compatible solver object.
- `solver_opts`: keyword options passed through to `solve`.
"""
struct SimulationSpec
    t_obs::Vector{Float64}
    obs_state::Union{Symbol, Vector{Symbol}}
    initial_conditions::Tuple{Vararg{Number}}
    tspan::Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}
    uncertain_param_values::Dict
    solver::Any
    solver_opts::NamedTuple

    function SimulationSpec(;
        t_obs::AbstractVector{<:Number},
        obs_state,
        initial_conditions::Tuple{Vararg{Number}},
        tspan,
        uncertain_param_values::Dict = Dict(),
        solver = Euler(),
        solver_opts::NamedTuple = NamedTuple(),
    )
        if isempty(t_obs)
            error("t_obs must not be empty")
        end

        states = _normalize_observed_states(obs_state)

        if isempty(initial_conditions)
            error("initial_conditions must not be empty")
        end

        normalized_tspan = _normalize_simulation_tspan(tspan)

        return new(
            Float64.(collect(t_obs)),
            length(states) == 1 ? only(states) : collect(states),
            initial_conditions,
            normalized_tspan,
            uncertain_param_values,
            solver,
            solver_opts,
        )
    end
end

function _normalize_observed_states(obs_state::Union{Symbol, AbstractString})
    state = Symbol(obs_state)
    if isempty(String(state))
        error("obs_state must not be empty")
    end

    return (state,)
end

function _normalize_observed_states(obs_state)
    if !(obs_state isa Union{AbstractVector, Tuple})
        error("obs_state must be a Symbol, string, or collection of Symbols/strings.")
    end

    if isempty(obs_state)
        error("obs_state must not be empty")
    end

    states = map(obs_state) do state
        if !(state isa Union{Symbol, AbstractString})
            error("obs_state entries must be Symbols or strings. Got $(typeof(state)).")
        end

        normalized = Symbol(state)
        if isempty(String(normalized))
            error("obs_state entries must not be empty")
        end

        normalized
    end

    return Tuple(states)
end

"""
    observed_states(simulation) -> Tuple{Vararg{Symbol}}

Return observed states as a tuple.

`SimulationSpec` accepts either a single state or multiple states. This helper
normalizes both cases for code that needs to iterate over observations.
"""
function observed_states(simulation::SimulationSpec)
    return simulation.obs_state isa Symbol ? (simulation.obs_state,) : Tuple(simulation.obs_state)
end

"""
    observed_state_count(simulation) -> Int

Return the number of observed states.
"""
function observed_state_count(simulation::SimulationSpec)
    return length(observed_states(simulation))
end

"""
    observed_state_index(sys, simulation) -> Int

Return the state index for a single observed state.

Use this when `simulation` observes exactly one state. For multi-state
observations, use `observed_state_indices`.
"""
function observed_state_index(sys, simulation::SimulationSpec)
    states = observed_states(simulation)
    if length(states) != 1
        error("SimulationSpec observes multiple states. Use observed_state_indices instead.")
    end

    return observed_state_index(sys, only(states))
end

"""
    observed_state_index(sys, obs_state) -> Int

Resolve an observed state symbol against a compiled ModelingToolkit system.
"""
function observed_state_index(sys, obs_state::Symbol)
    state = try
        getproperty(sys, obs_state)
    catch
        error("Observed state $obs_state was not found in the model system.")
    end

    index = findfirst(isequal(state), unknowns(sys))
    if isnothing(index)
        error("Observed state $obs_state was not found in the model unknown order.")
    end

    return index
end

"""
    observed_state_indices(sys, simulation) -> Vector{Int}

Resolve all observed state indices for a simulation.

The returned indices are in the same order as `observed_states(simulation)`.
"""
function observed_state_indices(sys, simulation::SimulationSpec)
    return [observed_state_index(sys, state) for state in observed_states(simulation)]
end

"""
    observed_state_save_idxs(sys, simulation)

Return `save_idxs` compatible with the observed states.

This returns an `Int` for a single observed state and a vector of indices for
multiple states, matching the forms accepted by SciML's `solve` keyword.
"""
function observed_state_save_idxs(sys, simulation::SimulationSpec)
    indices = observed_state_indices(sys, simulation)
    return length(indices) == 1 ? only(indices) : indices
end

function simulate!(model::Model, simulation::SimulationSpec; kwargs...)
    spec_kwargs = (
        parameters=simulation.uncertain_param_values,
        solver=simulation.solver,
        saveat=simulation.t_obs,
        save_idxs=observed_state_save_idxs(model.sys, simulation),
        solver_opts=simulation.solver_opts,
    )
    call_kwargs = merge(spec_kwargs, (; kwargs...))

    return simulate!(
        model,
        simulation.initial_conditions,
        simulation.tspan;
        call_kwargs...,
    )
end
