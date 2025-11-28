using ModelingToolkit
using OrdinaryDiffEq
using SymbolicIndexingInterface
"""
Model

A simulation wrapper for the ModelDefinition that creates and manages ODE Problems
- can extend to other problems later

Provides methods for simulation, parameter manipulation, and result visualisation

"""

# -------------------------------------------------------------------------
# Struct Definitions
# -------------------------------------------------------------------------

mutable struct Model
    model_def::ModelDefinition
    sys:: Any # Compiled ModellingToolkit system
    prob::Union{Nothing, ODEProblem} #ODE supported first
    sol::Union{Nothing, Any} # solution of the LAST simulation

    # fields for inference procedure
    param_setter:: Union{Nothing, Any}
    buffer_func::Union{Nothing, Function}
    uncertain_params::Union{Nothing, Vector}
    simulation_context::Union{Nothing, NamedTuple}

    # Constructor
    function Model(model_def::ModelDefinition, sys::Any)
        #Problem and solution are initially empty as they are created during simulation
        new(model_def, sys, nothing, nothing, nothing, nothing, nothing, nothing)

    end
end
# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

"""
    get_uncertain_parameters(model::Model) -> Vector{Symbol}

Get the parameter names marked as :uncertain in the model definition

"""
function get_uncertain_parameters(model::Model)
    uncertain = Symbol[]
    for (name, spec) in model.model_def.parameters
        if spec.role == :uncertain
            push!(uncertain, name)
        end
    end
    return uncertain
end

# -------------------------------------------------------------------------
# Inference hook
# -------------------------------------------------------------------------

function evaluate_model(model::Model, parameters::Vector{Float64})

    if model.simulation_context === nothing
        error("Model not prepared for simulation. Call setup_simulation!")
    end

    ctx = model.simulation_context

    # Reuse the already created problem and update
    new_p = model.buffer_func(parameters)
    model.param_setter(new_p, parameters)
    prob_new = remake(model.prob; p=new_p)

    sol = solve(prob_new, ctx.solver; 
                dt=ctx.dt, 
                saveat=ctx.t_obs, 
                save_idxs=ctx.obs_state_idx)
    
    return Array(sol)


end

# -------------------------------------------------------------------------
# Simulators
# -------------------------------------------------------------------------

"""
Simulate

    simulate!(model::Model, 
              initial_conditions::Vector{Float64},
              parameters::Dict,
              tspan::Tuple{Float64, Float64};
              solver=Tsit5(),
              dt::Float64=0.01,
              saveat=Float64[])

Runs a simple one off simulation and stores the results

# Arguments
- `model`: The Model object to simulate
- `initial_conditions`: Vector of initial values for each state variable
- `parameters`: Dict mapping parameter symbols to values
- `tspan`: Time span as (t_start, t_end)

# Keyword Arguments
- `solver`: ODE solver algorithm (default: Tsit5())
- `dt`: Time step for solver (default: 0.01)
- `saveat`: Specific time points to save (default: all points)

# Returns
- The solution object (also stored in model.sol)

"""

function simulate!(model::Model, 
                   initial_conditions::Vector{Float64},
                   parameters::Dict,
                   tspan::Tuple{Float64, Float64};
                   solver=Tsit5(),
                   dt::Float64=0.01,
                   saveat=Float64[])
    
    # If we have states [A ,B] and initial conditions [1.0, 1.0]
    # this creates Dict(A=> 1.0, and B=> 1.0)
    u0 = Dict(unknowns(model.sys) .=> initial_conditions)
    p_map = Dict(p.symbol => p.value for p in values(model.model_def.parameters) if p.value !== nothing)
    # Merge them all together - here user defined params will override existing
    all_params = merge(u0, p_map, parameters)

    # Currently supports ODE but can add a contiditional here based on what the
    # user specifies
    model.prob = ODEProblem(model.sys, all_params, tspan)

    # Solve the Problem
    # If saveat is empty, use dt as the save interval
    # Otherwise use the specific time points provided

    if isempty(saveat)
        model.sol = solve(model.prob, solver, dt=dt)
    else
        model.sol = solve(model.prob, solver, dt=dt, saveat=saveat)
    end

    return model.sol
end

"""
Prepares the model for simulation, created onced for many evaluations.

    setup_evaluation!(model::Model;
                      t_obs::Vector{Float64},
                      obs_state_idx::Int,
                      initial_conditions::Vector{Float64},
                      tspan::Tuple{Float64, Float64},
                      solver=Euler(),
                      dt::Float64=0.01)

"""

function setup_simulation!(model::Model,
                          t_obs::Vector{Float64},
                          obs_state_idx::Int,
                          initial_conditions::Vector{Float64},
                          parameters::Dict,
                          tspan::Tuple{Float64, Float64};
                          solver=Euler(),
                          dt::Float64=0.01)
    
    u0 = Dict(unknowns(model.sys) .=> initial_conditions)
    p_map = Dict(p.symbol => p.value for p in values(model.model_def.parameters) if p.value !== nothing)
    
    all_params = merge(u0, p_map, parameters)
    model.prob = ODEProblem(model.sys, all_params, tspan)
    
    uncertain_names = get_uncertain_parameters(model)
    model.uncertain_params = [getproperty(model.sys, name) for name in uncertain_names]
    model.param_setter = setp(model.sys, model.uncertain_params)
    model.buffer_func = (p) -> remake_buffer(
        model.sys, model.prob.p, Dict(zip(model.uncertain_params, p))
    )
    
    model.simulation_context = (
        t_obs = t_obs,
        obs_state_idx = obs_state_idx,
        solver = solver,
        dt = dt
    )
    
    println("✅ Model ready for simulation")
    return nothing
end