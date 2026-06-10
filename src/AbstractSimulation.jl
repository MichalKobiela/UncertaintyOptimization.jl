"""
    SimulationSpec

Shared simulation settings that can be reused by inference and scan specs.
"""
struct SimulationSpec
    t_obs::Vector{Float64}
    obs_state::Symbol
    initial_conditions::Tuple{Vararg{Number}}
    tspan::Tuple{Float64, Float64}
    uncertain_param_values::Dict
    solver::Any
    solver_opts::NamedTuple

    function SimulationSpec(;
        t_obs::AbstractVector{<:Number},
        obs_state::Union{Symbol, AbstractString},
        initial_conditions::Tuple{Vararg{Number}},
        tspan::Tuple{<:Number, <:Number},
        uncertain_param_values::Dict = Dict(),
        solver = Euler(),
        solver_opts::NamedTuple = NamedTuple(),
    )
        if isempty(t_obs)
            error("t_obs must not be empty")
        end

        state = Symbol(obs_state)
        if isempty(String(state))
            error("obs_state must not be empty")
        end

        if isempty(initial_conditions)
            error("initial_conditions must not be empty")
        end

        if tspan[1] >= tspan[2]
            error("tspan must be ordered as (start, stop) with start < stop")
        end

        return new(
            Float64.(collect(t_obs)),
            state,
            initial_conditions,
            (Float64(tspan[1]), Float64(tspan[2])),
            uncertain_param_values,
            solver,
            solver_opts,
        )
    end
end

function observed_state_index(sys, simulation::SimulationSpec)
    return observed_state_index(sys, simulation.obs_state)
end

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
