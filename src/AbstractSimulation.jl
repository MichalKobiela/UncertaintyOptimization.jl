"""
    SimulationSpec

Shared simulation settings that can be reused by inference and scan specs.
"""
struct SimulationSpec
    t_obs::Vector{Float64}
    obs_state_idx::Int
    initial_conditions::Tuple{Vararg{Number}}
    tspan::Tuple{Float64, Float64}
    uncertain_param_values::Dict
    solver::Any
    solver_opts::NamedTuple

    function SimulationSpec(;
        t_obs::AbstractVector{<:Number},
        obs_state_idx::Int = 1,
        initial_conditions::Tuple{Vararg{Number}},
        tspan::Tuple{<:Number, <:Number},
        uncertain_param_values::Dict = Dict(),
        solver = Euler(),
        solver_opts::NamedTuple = NamedTuple(),
    )
        if isempty(t_obs)
            error("t_obs must not be empty")
        end

        if obs_state_idx < 1
            error("obs_state_idx must be positive")
        end

        if isempty(initial_conditions)
            error("initial_conditions must not be empty")
        end

        if tspan[1] >= tspan[2]
            error("tspan must be ordered as (start, stop) with start < stop")
        end

        return new(
            Float64.(collect(t_obs)),
            obs_state_idx,
            initial_conditions,
            (Float64(tspan[1]), Float64(tspan[2])),
            uncertain_param_values,
            solver,
            solver_opts,
        )
    end
end
