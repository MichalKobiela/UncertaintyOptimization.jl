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

    # Constructor
    function Model(model_def::ModelDefinition, sys::Any)
        #Problem and solution are initially empty as they are created during simulation
        new(model_def, sys, nothing, nothing)

    end
end