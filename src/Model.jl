using ModelingToolkit
using OrdinaryDiffEq

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
    buffer_func::Union{Nohing, Function}
    uncertain_params::Union{Nothing, Vector}

    # Constructor
    function Model(model_def::ModelDefinition, sys::Any)
        #Problem and solution are initially empty as they are created during simulation
        new(model_def, sys, nothing, nothing, nothing, nothing, nothing)

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
# Methods
# -------------------------------------------------------------------------



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

