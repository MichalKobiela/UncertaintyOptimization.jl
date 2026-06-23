"""
    SimulationSpec

Shared simulation settings that can be reused by inference and scan specs.
"""
struct SimulationSpec
    t_obs::Vector{Float64}
    obs_state::Union{Symbol, Vector{Symbol}}
    initial_conditions::Tuple{Vararg{Number}}
    tspan::Tuple{Float64, Float64}
    uncertain_param_values::Dict
    solver::Any
    solver_opts::NamedTuple

    function SimulationSpec(;
        t_obs::AbstractVector{<:Number},
        obs_state,
        initial_conditions::Tuple{Vararg{Number}},
        tspan::Tuple{<:Number, <:Number},
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

        if tspan[1] >= tspan[2]
            error("tspan must be ordered as (start, stop) with start < stop")
        end

        return new(
            Float64.(collect(t_obs)),
            length(states) == 1 ? only(states) : collect(states),
            initial_conditions,
            (Float64(tspan[1]), Float64(tspan[2])),
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

Resolve an observed state symbol against a compiled system.
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
"""
function observed_state_indices(sys, simulation::SimulationSpec)
    return [observed_state_index(sys, state) for state in observed_states(simulation)]
end

"""
    observed_state_save_idxs(sys, simulation)

Return `save_idxs` compatible with the observed states.
"""
function observed_state_save_idxs(sys, simulation::SimulationSpec)
    indices = observed_state_indices(sys, simulation)
    return length(indices) == 1 ? only(indices) : indices
end
